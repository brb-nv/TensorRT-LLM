from collections import defaultdict

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (FlashInferAttention,
                                                   FlashInferAttentionMetadata)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


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
    num_gens = 2
    context_sequence_lengths = [3, 2]
    sequence_lengths = context_sequence_lengths + [1] * num_gens
    past_seen_tokens = [30, 40, 62, 75]
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

    # [context_1, context_2, gen_1, gen_2]
    results_1 = []

    seq_lens = torch.tensor(sequence_lengths).int()
    attn_metadata = TestingFlashInferAttentionMetadata(
        seq_lens=seq_lens,
        num_contexts=len(context_sequence_lengths),
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=past_seen_tokens),
        max_num_requests=4,
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

        result_1 = flashinfer_attn.forward(q, k, v, attn_metadata)
        assert result_1.size()[0] == sum(context_sequence_lengths) + num_gens

        # validate kv cache was updated expectedly
        cache_buf = kv_cache_manager.get_buffers(flashinfer_attn.layer_idx)
        assert cache_buf is not None
        num_kv_heads = cache_buf.size(-2)

        # validate contexts
        block_ids_per_seq = kv_cache_manager.get_batch_cache_indices(
            request_ids)
        for seq_id in range(len(context_sequence_lengths)):
            # get a contiguous copy of the cache for the sequence
            block_ids = block_ids_per_seq[seq_id]
            last_block_len = attn_metadata.paged_kv_last_page_len[seq_id]
            cached_kvs = torch.concat(cache_buf[block_ids, :].unbind(dim=0),
                                        dim=1)
            # only look at new tokens added
            cached_kvs = cached_kvs[:,
                                    past_seen_tokens[seq_id]:last_block_len]

            # compare to input kvs
            torch.testing.assert_close(
                cached_kvs[0].to(context_ks[seq_id].dtype),
                context_ks[seq_id].view(-1, num_kv_heads, head_dim))
            torch.testing.assert_close(
                cached_kvs[1].to(context_vs[seq_id].dtype),
                context_vs[seq_id].view(-1, num_kv_heads, head_dim))

        # validate generations (same way)
        for gen_seq_id in range(num_gens):
            seq_id = len(context_sequence_lengths) + gen_seq_id
            block_ids = block_ids_per_seq[seq_id]
            last_block_len = attn_metadata.paged_kv_last_page_len[seq_id]
            cached_kvs = torch.concat(
                cache_buf[block_ids, :].unbind(dim=0),
                dim=1)[:, past_seen_tokens[seq_id]:last_block_len]

            torch.testing.assert_close(
                cached_kvs[0],
                gen_ks[gen_seq_id].view(-1, num_kv_heads, head_dim))
            torch.testing.assert_close(
                cached_kvs[1],
                gen_vs[gen_seq_id].view(-1, num_kv_heads, head_dim))

        results_1.append(result_1)
        del cache_buf

    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 1

    # Make sure prepare() re-planned all params.
    attn_metadata.prepare()
    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 2

    # [context_1, gen_1]
    results_2 = []
    num_cached_tokens_per_seq = [
        j for j in [
            past_seen_tokens[0], past_seen_tokens[len(
                context_sequence_lengths)]
        ]
    ]

    seq_lens = torch.tensor([context_sequence_lengths[0], 1],
                            dtype=torch.int)
    attn_metadata = TestingFlashInferAttentionMetadata(
        seq_lens=seq_lens,
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq),
        max_num_requests=2,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=[0, 2],
    )

    attn_metadata.prepare()

    for attn_layer_idx, flashinfer_attn in enumerate(layers):
        context_qs, context_ks, context_vs = contexts_per_layer[
            attn_layer_idx]
        gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

        result_2 = flashinfer_attn.forward(
            torch.cat((context_qs[0], gen_qs[0])),
            torch.cat((context_ks[0], gen_ks[0])),
            torch.cat((context_vs[0], gen_vs[0])), attn_metadata)
        assert result_2.size()[0] == context_sequence_lengths[0] + 1
        results_2.append(result_2)

    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 1

    # Make sure prepare() re-planned all params.
    attn_metadata.prepare()
    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 2

    # [context_2, gen_2]
    results_3 = []
    num_cached_tokens_per_seq = [
        j for j in [
            past_seen_tokens[1], past_seen_tokens[
                len(context_sequence_lengths) + 1]
        ]
    ]

    seq_lens = torch.tensor([context_sequence_lengths[1], 1],
                            dtype=torch.int)
    attn_metadata = TestingFlashInferAttentionMetadata(
        seq_lens=seq_lens,
        num_contexts=1,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq),
        max_num_requests=2,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=[1, 3],
    )

    attn_metadata.prepare()
    for attn_layer_idx, flashinfer_attn in enumerate(layers):
        context_qs, context_ks, context_vs = contexts_per_layer[
            attn_layer_idx]
        gen_qs, gen_ks, gen_vs = gens_per_layer[attn_layer_idx]

        result_3 = flashinfer_attn.forward(
            torch.cat((context_qs[1], gen_qs[1])),
            torch.cat((context_ks[1], gen_ks[1])),
            torch.cat((context_vs[1], gen_vs[1])), attn_metadata)
        assert result_3.size()[0] == context_sequence_lengths[1] + 1
        results_3.append(result_3)

    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 1

    # Make sure prepare() re-planned all params.
    attn_metadata.prepare()
    for plan_params in attn_metadata._plan_params_to_wrappers.keys():
        assert attn_metadata.get_num_plans(plan_params) == 2

    # assert value

    for result_1, result_2, result_3 in zip(results_1, results_2,
                                            results_3):
        torch.testing.assert_close(
            torch.cat((
                result_1[:context_sequence_lengths[0] +
                            context_sequence_lengths[1], :],
                result_1[sum(context_sequence_lengths
                                ):sum(context_sequence_lengths) + 2],
            )),
            torch.cat((
                result_2[:context_sequence_lengths[0], :],
                result_3[:context_sequence_lengths[1], :],
                result_2[context_sequence_lengths[0]:, :],
                result_3[context_sequence_lengths[1]:, :],
            )))

    kv_cache_manager.shutdown()

if __name__ == "__main__":
    torch.manual_seed(4)
    test_flashinfer_swa()
