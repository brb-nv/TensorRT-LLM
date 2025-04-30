from collections import defaultdict
from typing import List
import torch
import math

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (FlashInferAttention,
                                                   FlashInferAttentionMetadata)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def calculate_ref_result(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         num_heads: int,
                         num_kv_heads: int,
                         head_dim: int,
                         sequence_lengths: List[int],
                         attention_window_size: int):
    """
    use standard attention to calculate the reference result by iterating over each request
    q shape: (total_tokens, num_heads * head_dim)
    k shape: (total_tokens, num_kv_heads * head_dim)
    v shape: (total_tokens, num_kv_heads * head_dim)
    """
    num_requests = len(sequence_lengths)
    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    total_tokens = 0

    # Reshape inputs for reference calculation
    for i in range(num_requests):
        q_seq = q[total_tokens:total_tokens + sequence_lengths[i]]
        k_seq = k[total_tokens:total_tokens + sequence_lengths[i]]
        v_seq = v[total_tokens:total_tokens + sequence_lengths[i]]

        # Reshape to (seq_len, num_heads, head_dim)
        q_seq = q_seq.view(sequence_lengths[i], num_heads, head_dim)
        k_seq = k_seq.view(sequence_lengths[i], num_kv_heads, head_dim)
        v_seq = v_seq.view(sequence_lengths[i], num_kv_heads, head_dim)

        q_reshaped.append(q_seq.transpose(0,
                                          1))  # (num_heads, seq_len, head_dim)
        k_reshaped.append(k_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)
        v_reshaped.append(v_seq.transpose(
            0, 1))  # (num_kv_heads, seq_len, head_dim)

        total_tokens += sequence_lengths[i]

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(num_requests):
        q = q_reshaped[i]  # (num_heads, seq_len, head_dim)
        k = k_reshaped[i]  # (num_kv_heads, seq_len, head_dim)
        v = v_reshaped[i]  # (num_kv_heads, seq_len, head_dim)

        # Handle grouped-query attention if num_heads > num_kv_heads.
        if num_heads > num_kv_heads:
            num_kv_groups = num_heads // num_kv_heads
            k = repeat_kv(k.unsqueeze(0), num_kv_groups).squeeze(0)
            v = repeat_kv(v.unsqueeze(0), num_kv_groups).squeeze(0)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(head_dim)

        # For sliding window attention, we block tokens that are too far away from the current token.
        seq_len = q.shape[1]
        causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=q.device)
        for token_idx in range(seq_len):
            start = max(0, token_idx - attention_window_size + 1)
            causal_mask[token_idx, start: token_idx + 1] = 1

        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        # Apply softmax to get attention probabilities
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights,
                                   v)  # (num_heads, seq_len, head_dim)

        # Reshape back to (seq_len, num_heads*head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            sequence_lengths[i], num_heads * head_dim)
        ref_results.append(attn_output)

    ref_result = torch.cat(ref_results)
    return ref_result


class TestingFlashInferAttentionMetadata(FlashInferAttentionMetadata):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_times_planned = defaultdict(int)

    def get_num_plans(self, plan_params) -> int:
        return self._num_times_planned[plan_params]

    def _plan_with_params(self, plan_params):
        if self.needs_plan(plan_params):
            self._num_times_planned[plan_params] += 1
        return super()._plan_with_params(plan_params)


def test_flashinfer_swa():
    num_layers = 1
    num_heads = 1
    num_kv_heads = 1
    head_dim = 256
    dtype = torch.float16

    device = torch.device('cuda')

    # TODO: make these a part of the scenario?
    num_gens = 0
    context_sequence_lengths = [16]
    sequence_lengths = context_sequence_lengths + [1] * num_gens
    past_seen_tokens = [0]
    batch_size = num_gens + len(context_sequence_lengths)
    request_ids = list(range(batch_size))
    token_nums = (torch.tensor(sequence_lengths) +
                    torch.tensor(past_seen_tokens)).tolist()

    num_blocks = 16
    tokens_per_block = 128
    max_seq_len = tokens_per_block * num_blocks
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    if dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype for unit test")

    # TODO: Mention max_attention_window in kv_cache_config.
    kv_cache_config = KvCacheConfig(max_tokens=num_blocks *
                                    tokens_per_block)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    for i in range(kv_cache_manager.num_layers):
        buf = kv_cache_manager.get_buffers(i)
        if buf is not None:
            torch.nn.init.normal_(buf)
            del buf

    if isinstance(num_kv_heads, int):
        num_kv_heads = [num_kv_heads] * num_layers

    contexts_per_layer = []
    gens_per_layer = []

    for layer_idx in range(num_layers):
        kv_heads = num_kv_heads[layer_idx]
        if kv_heads is None:
            continue

        context_qs = [
            torch.randn(sequence_length,
                        num_heads * head_dim,
                        dtype=dtype,
                        device=device)
            for sequence_length in context_sequence_lengths
        ]

        context_ks = [
            torch.randn(sequence_length,
                        kv_heads * head_dim,
                        dtype=dtype,
                        device=device)
            for sequence_length in context_sequence_lengths
        ]
        context_vs = [
            torch.randn(sequence_length,
                        kv_heads * head_dim,
                        dtype=dtype,
                        device=device)
            for sequence_length in context_sequence_lengths
        ]

        contexts_per_layer.append((context_qs, context_ks, context_vs))

        gen_qs = [
            torch.randn(1, num_heads * head_dim, dtype=dtype, device=device)
            for _ in range(num_gens)
        ]

        gen_ks = [
            torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
            for _ in range(num_gens)
        ]

        gen_vs = [
            torch.randn(1, kv_heads * head_dim, dtype=dtype, device=device)
            for _ in range(num_gens)
        ]

        gens_per_layer.append((gen_qs, gen_ks, gen_vs))

    layers = [
        FlashInferAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=kv_heads,
        ) for layer_idx, kv_heads in enumerate(num_kv_heads)
        if kv_heads is not None
    ]

    # [context_1]
    results_1 = []

    seq_lens = torch.tensor(sequence_lengths).int()
    attn_metadata = TestingFlashInferAttentionMetadata(
        seq_lens=seq_lens,
        num_contexts=len(context_sequence_lengths),
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
        max_num_requests=1,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
    )

    attn_metadata.prepare()
    for attn_layer_idx, flashinfer_attn in enumerate(layers):
        context_qs, context_ks, context_vs = contexts_per_layer[
            attn_layer_idx]
        gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

        q = torch.cat((*context_qs, *gen_qs))
        k = torch.cat((*context_ks, *gen_ks))
        v = torch.cat((*context_vs, *gen_vs))

        result = flashinfer_attn.forward(q, k, v, attn_metadata)
        expected = calculate_ref_result(q, k, v, num_heads=1, num_kv_heads=1, head_dim=256, sequence_lengths=[16], attention_window_size=5)
        assert result.size()[0] == sum(context_sequence_lengths) + num_gens
        for idx in range(16):
            print(idx, ":" , torch.max(torch.abs(expected[idx] - result[idx])))
        diff = torch.abs(expected - result)
        print(f"max: {diff.max()}, min: {diff.min()}, mean: {diff.mean()}")


if __name__ == "__main__":
    torch.manual_seed(4)
    test_flashinfer_swa()
