# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import sys
import time
import traceback
import weakref
from dataclasses import dataclass
from typing import List, Optional

_HELIX_DEBUG_DIR = "/home/bbuddharaju/scratch/TensorRT-LLM/helix_debug_logs"

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionMetadata,
    KVCacheParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# Set this to True (or set env HELIX_DISABLE_ROPE=1) to disable RoPE for debugging
DISABLE_ROPE_FOR_DEBUG = os.environ.get("HELIX_DISABLE_ROPE", "0") == "1"

# Values inspired by a small LLaMA-like model
@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 2
    num_kv_heads: int = 2
    head_dim: int = 128
    hidden_size: int = 256  # num_heads * head_dim
    rope_theta: float = 10000.0
    kv_cache_tokens_per_block: int = 32
    bias: bool = False
    batch: int = 8
    ctx_len: int = 1024
    ref_steps: int = 1
    # note: need to use fairly high tolerances because the softmax stats can
    # lose a lot of precision and we're using bf16 here
    atol: float = 1e-1
    rtol: float = 5e-2

    @property
    def max_position_embeddings(self) -> int:
        return self.ctx_len + 1


all_scenarios = [
    Scenario(batch=1, ctx_len=64),
    Scenario(batch=1, ctx_len=512),
    Scenario(batch=1, ctx_len=1024),
    Scenario(batch=1, ctx_len=2048),
    Scenario(batch=1, ctx_len=4096),
    Scenario(batch=1, ctx_len=8192),
    Scenario(batch=1, ctx_len=16384),
    Scenario(batch=1, ctx_len=32768),
    Scenario(batch=1, ctx_len=65536),
    Scenario(batch=1, ctx_len=131072),
    Scenario(batch=8, ctx_len=1024),
    Scenario(batch=8, ctx_len=2048),
    Scenario(batch=8, ctx_len=4096),
    Scenario(batch=8, ctx_len=8192),
    Scenario(batch=8, ctx_len=16384),
    Scenario(batch=8, ctx_len=32768),
    Scenario(batch=8, ctx_len=65536),
    Scenario(batch=16, ctx_len=1024),
    Scenario(batch=16, ctx_len=2048),
    Scenario(batch=16, ctx_len=4096),
    Scenario(batch=16, ctx_len=8192),
    Scenario(batch=16, ctx_len=16384),
    Scenario(batch=16, ctx_len=32768),
]

# Limit the number of test scenarios to avoid taking too long
test_scenarios = [
    all_scenarios[0],
    # all_scenarios[1],
    # all_scenarios[4],
    # all_scenarios[7],
    # all_scenarios[10],
    # all_scenarios[13],
    # all_scenarios[17],
    # all_scenarios[18],
]


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 2048
    num_attention_heads: int = 16
    rope_scaling: dict = None
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    model_type: str = "llama"


def _setup_kv_and_metadata(scenario: Scenario, mapping: Mapping, gen_steps: int):
    """Set up KVCacheManager and attn_metadata for standard MHA."""
    n_gpu = mapping.world_size
    assert scenario.ctx_len % n_gpu == 0
    ctx_len_per_gpu = scenario.ctx_len // n_gpu
    max_tokens = (
        (ctx_len_per_gpu + gen_steps + scenario.kv_cache_tokens_per_block - 1)
        // scenario.kv_cache_tokens_per_block
        * scenario.kv_cache_tokens_per_block
        * scenario.batch
    )
    kv_cache_manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        # Use SELF cache type for standard MHA (separate K and V caches)
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=scenario.num_layers,
        num_kv_heads=scenario.num_kv_heads,
        head_dim=scenario.head_dim,
        tokens_per_block=scenario.kv_cache_tokens_per_block,
        max_seq_len=ctx_len_per_gpu + gen_steps,
        max_batch_size=scenario.batch,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kv_cache_dtype)),
    )
    for req_id in range(scenario.batch):
        req = LlmRequest(
            request_id=req_id,
            max_new_tokens=1,
            input_tokens=[1] * ctx_len_per_gpu,
            sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
            is_streaming=False,
        )
        req.is_dummy_request = True
        req.paged_kv_block_ids = []
        beam_width = 1
        kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, beam_width, req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS
        req.prompt_len = ctx_len_per_gpu
        req.py_prompt_len = req.prompt_len
    attn_metadata = get_attention_backend("TRTLLM").Metadata(
        seq_lens=torch.tensor([ctx_len_per_gpu] * scenario.batch, dtype=torch.int),
        request_ids=list(range(scenario.batch)),
        max_num_requests=scenario.batch,
        num_contexts=scenario.batch,
        prompt_lens=[ctx_len_per_gpu] * scenario.batch,
        max_num_tokens=ctx_len_per_gpu,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in range(scenario.batch)],
        ),
        mapping=mapping,
    )
    attn_metadata.prepare()
    return kv_cache_manager, attn_metadata


def _generate_random_weights(attn: Attention):
    """Initialize Attention weights with random values."""
    for name, param in attn.named_parameters():
        if param.dtype.itemsize <= 1:
            t2 = torch.empty_like(param, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(t2)
            param.data.copy_(t2)
        else:
            torch.nn.init.kaiming_uniform_(param.data)


def _copy_to_cp(weights, param_name, dim, rank, world_size):
    w_dim_per_rank = weights[param_name].shape[dim] // world_size
    w_dim_start = rank * w_dim_per_rank
    w_dim_end = w_dim_start + w_dim_per_rank
    slices = [slice(None)] * weights[param_name].ndim
    slices[dim] = slice(w_dim_start, w_dim_end)
    weights[param_name] = weights[param_name][slices]


def _error_report(output, ref_output, atol, rtol, prefix):
    err = torch.abs(output - ref_output)
    ref_abs = torch.abs(ref_output)
    ref_abs[ref_abs == 0] = torch.finfo(ref_abs.dtype).smallest_normal
    rel_err = err / ref_abs
    max_err_idx = torch.unravel_index(torch.argmax(err - atol - rtol * ref_abs), err.shape)
    values_err = (output[max_err_idx].item(), ref_output[max_err_idx].item())
    max_abs_err_idx = torch.unravel_index(torch.argmax(err), err.shape)
    values_abs = (output[max_abs_err_idx].item(), ref_output[max_abs_err_idx].item())
    max_rel_err_idx = torch.unravel_index(torch.argmax(rel_err), rel_err.shape)
    values_rel = (output[max_rel_err_idx].item(), ref_output[max_rel_err_idx].item())
    max_abs_err = err[max_abs_err_idx].item()
    max_rel_err = rel_err[max_rel_err_idx].item()
    max_err_idx = [x.item() for x in max_err_idx]
    max_abs_err_idx = [x.item() for x in max_abs_err_idx]
    max_rel_err_idx = [x.item() for x in max_rel_err_idx]
    isclose = err < atol + rtol * ref_abs
    n_error = (~isclose).sum().item()
    print(
        f"{prefix}: {n_error} errors, max error index: {max_err_idx} "
        f"(test/ref values: {values_err}), max abs error index: {max_abs_err_idx} "
        f"(test/ref values: {values_abs}, err: {max_abs_err}), max rel error index: {max_rel_err_idx} "
        f"(test/ref values: {values_rel}, err: {max_rel_err}), atol: {atol}, rtol: {rtol}"
    )
    return n_error


def _run_attention_distributed(
    rank: int,
    world_size: int,
    scenario: Scenario,
    mapping: Mapping,
    test_params: tuple,
    ref_output: torch.Tensor,
    gen_steps: int,
    ref_pre_oproj: Optional[torch.Tensor] = None,
):
    input_ctx, input_gen, position_ids_ctx, weights, pos_embd_params = test_params
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )

    # DEBUG: Print input state
    print(f"[DEBUG] Rank {rank}: input_gen.shape={input_gen.shape}, "
          f"input_gen[0,:8]={input_gen[0,:8].tolist()}")
    print(f"[DEBUG] Rank {rank}: position_ids_gen={position_ids_gen.tolist()}")
    print(f"[DEBUG] Rank {rank}: mapping cp_size={mapping.cp_size}, "
          f"cp_rank={mapping.cp_rank}, tp_size={mapping.tp_size}, "
          f"rank={mapping.rank}")

    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    attn = Attention(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        config=config,
        enable_helix_test=True,
    ).cuda()

    # DEBUG: Print attention module config
    print(f"[DEBUG] Rank {rank}: Attention module - "
          f"num_heads={attn.num_heads}, num_kv_heads={attn.num_key_value_heads}, "
          f"num_heads_tp_cp={attn.num_heads_tp_cp}, head_dim={attn.head_dim}, "
          f"q_size={attn.q_size}, kv_size={attn.kv_size}, "
          f"rope_fusion={attn.rope_fusion}, "
          f"support_fused_qkv={attn.support_fused_qkv}, "
          f"cp_size={attn.cp_size}, tp_size={attn.tp_size}")

    # DEBUG: Print weight shapes before split
    print(f"[DEBUG] Rank {rank}: weights before split: "
          f"qkv_proj.weight={weights['qkv_proj.weight'].shape}, "
          f"o_proj.weight={weights['o_proj.weight'].shape}")

    # Split o_proj weight along input dimension for CP
    _copy_to_cp(weights, "o_proj.weight", 1, rank, world_size)

    # DEBUG: Print weight shapes after split
    print(f"[DEBUG] Rank {rank}: o_proj.weight after split={weights['o_proj.weight'].shape}")
    print(f"[DEBUG] Rank {rank}: expected o_proj shape={attn.o_proj.weight.shape}")
    print(f"[DEBUG] Rank {rank}: qkv_proj.weight[0,:8]={weights['qkv_proj.weight'][0,:8].tolist()}")
    print(f"[DEBUG] Rank {rank}: o_proj.weight[0,:8]={weights['o_proj.weight'][0,:8].tolist()}")

    attn.load_state_dict(weights)

    # Hook to capture distributed attention output before o_proj
    dist_pre_oproj = {}
    def _capture_dist_pre_oproj(module, args, kwargs=None):
        inp = args[0] if args else None
        if inp is not None:
            dist_pre_oproj['value'] = inp.detach().clone()
    dist_hook = attn.o_proj.register_forward_pre_hook(_capture_dist_pre_oproj)

    # Set up KVCacheManager and attn_metadata for distributed
    kv_cache_manager, attn_metadata = _setup_kv_and_metadata(scenario, mapping, gen_steps)
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)

    # Split inputs into chunks for each rank
    input_ctx_bs_rank = input_ctx_bs[:, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu, :]
    input_ctx_rank = input_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu, scenario.hidden_size
    ).contiguous()
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch, scenario.ctx_len)
    position_ids_ctx_bs_rank = position_ids_ctx_bs[
        :, rank * ctx_len_per_gpu : (rank + 1) * ctx_len_per_gpu
    ]
    position_ids_ctx_rank = position_ids_ctx_bs_rank.reshape(
        scenario.batch * ctx_len_per_gpu
    ).contiguous()
    # Context step
    print(f"[DEBUG] Rank {rank}: DIST context step, "
          f"input_ctx_rank.shape={input_ctx_rank.shape}, "
          f"position_ids_ctx_rank[:8]={position_ids_ctx_rank[:8].tolist()}, "
          f"position_ids_ctx_rank[-8:]={position_ids_ctx_rank[-8:].tolist()}, "
          f"mapping_for_attn_metadata: cp_size={mapping.cp_size}, cp_rank={mapping.cp_rank}")
    with model_extra_attrs(extra_attrs):
        attn(position_ids_ctx_rank, input_ctx_rank, attn_metadata)

    outputs = []
    start = time.time()

    # CUDA graph setup for timing
    use_cuda_graph = gen_steps > scenario.ref_steps
    graph = None
    graph_output = None
    start = time.time()

    for step in range(gen_steps):
        helix_is_inactive_rank = []
        for req_id in range(scenario.batch):
            kv_cache_manager.impl.add_token(req_id)
            # Assume last rank is active for all gen steps.
            if rank == world_size - 1:
                helix_is_inactive_rank.append(False)
                cache_add = step
            else:
                helix_is_inactive_rank.append(True)
                cache_add = 0
        cached_tokens_per_seq = [ctx_len_per_gpu + cache_add for _ in range(scenario.batch)]

        # DEBUG: Print generation step info
        if step == 0:
            print(f"[DEBUG] Rank {rank} step {step}: "
                  f"helix_is_inactive_rank={helix_is_inactive_rank}, "
                  f"cache_add={cache_add}, "
                  f"cached_tokens_per_seq={cached_tokens_per_seq}, "
                  f"ctx_len_per_gpu={ctx_len_per_gpu}, "
                  f"position_ids_gen={position_ids_gen.tolist()}, "
                  f"num_heads={scenario.num_heads}, "
                  f"num_kv_heads={scenario.num_kv_heads}, "
                  f"heads_per_rank={scenario.num_heads // world_size}")

        if step == 0:
            attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                request_ids=list(range(scenario.batch)),
                max_num_requests=scenario.batch,
                num_contexts=0,
                prompt_lens=[ctx_len_per_gpu] * scenario.batch,
                max_num_tokens=ctx_len_per_gpu,
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=cached_tokens_per_seq,
                ),
            )
            attn_metadata.enable_helix = True
            attn_metadata.helix_is_inactive_rank = torch.tensor(
                helix_is_inactive_rank, dtype=torch.bool, device="cuda"
            )
            attn_metadata.helix_is_inactive_rank_cpu = attn_metadata.helix_is_inactive_rank.to(
                device="cpu"
            ).pin_memory()
            attn_metadata.helix_position_offsets = torch.tensor(
                position_ids_gen, dtype=torch.int, device="cuda"
            )
            attn_metadata.helix_position_offsets_cpu = attn_metadata.helix_position_offsets.to(
                device="cpu"
            ).pin_memory()
        else:
            attn_metadata.kv_cache_params = KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_tokens_per_seq,
            )
            attn_metadata.helix_is_inactive_rank = torch.tensor(
                helix_is_inactive_rank, dtype=torch.bool, device="cuda"
            )
        attn_metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
        if not use_cuda_graph:
            with model_extra_attrs(extra_attrs):
                result = attn(position_ids_gen, input_gen, attn_metadata)
            if step < scenario.ref_steps:
                outputs.append(result)

            # Diagnostic: compare pre-oproj with reference
            if step == 0 and 'value' in dist_pre_oproj and ref_pre_oproj is not None:
                dist_attn_out = dist_pre_oproj['value']
                heads_per_rank = scenario.num_heads // world_size
                head_dim = scenario.head_dim
                local_dim = heads_per_rank * head_dim
                ref_slice = ref_pre_oproj[:, rank * local_dim : (rank + 1) * local_dim]
                print(f"\n{'='*80}")
                print(f"[DIAG] Rank {rank}: PRE-O_PROJ COMPARISON (step {step})")
                print(f"[DIAG] Rank {rank}: dist_attn_out shape={dist_attn_out.shape}, "
                      f"ref_slice shape={ref_slice.shape}")
                print(f"[DIAG] Rank {rank}: dist_attn_out[0,:16]={dist_attn_out[0,:16].tolist()}")
                print(f"[DIAG] Rank {rank}: ref_slice[0,:16]={ref_slice[0,:16].tolist()}")
                diff = (dist_attn_out - ref_slice).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"[DIAG] Rank {rank}: max_abs_diff={max_diff:.6f}, "
                      f"mean_abs_diff={mean_diff:.6f}")
                # Per-head comparison
                for h in range(min(heads_per_rank, 4)):
                    h_start = h * head_dim
                    h_end = (h + 1) * head_dim
                    h_diff = diff[0, h_start:h_end]
                    print(f"[DIAG] Rank {rank}: head {rank * heads_per_rank + h} (local {h}): "
                          f"max_diff={h_diff.max().item():.6f}, "
                          f"mean_diff={h_diff.mean().item():.6f}, "
                          f"dist[0:4]={dist_attn_out[0, h_start:h_start+4].tolist()}, "
                          f"ref[0:4]={ref_slice[0, h_start:h_start+4].tolist()}")
                print(f"{'='*80}\n")

            print(f"Rank {rank} {world_size}-GPU: result: {result[0, :8]} / {result[-1, -8:]}")
            position_ids_gen += 1
            continue

        # CUDA graph capture on first step when timing
        if step == 0:
            print(f"Rank {rank} {world_size}-GPU: Creating CUDA graph and capturing")
            attn_metadata = attn_metadata.create_cuda_graph_metadata(max_batch_size=scenario.batch)
            attn_metadata.prepare()
            extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)

            for _ in range(2):
                with model_extra_attrs(extra_attrs):
                    result = attn(position_ids_gen, input_gen, attn_metadata)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with model_extra_attrs(extra_attrs):
                    graph_output = attn(position_ids_gen, input_gen, attn_metadata)
            result = graph_output
        elif step == scenario.ref_steps:
            start = time.time()

        graph.replay()
        result = graph_output
        position_ids_gen += 1
        if step < scenario.ref_steps:
            outputs.append(result)

    end = time.time()
    if gen_steps == scenario.ref_steps:
        avg_gen_time = float("inf")
    else:
        avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
    throughput = scenario.batch / avg_gen_time
    print(
        f"Rank {rank} {world_size}-GPU: time taken for "
        f"{gen_steps - scenario.ref_steps} steps: "
        f"{end - start} s, throughput: {throughput} Attn/s"
    )
    output = torch.stack(outputs, dim=0)
    dist_hook.remove()
    kv_cache_manager.shutdown()

    # DEBUG: Save outputs for offline comparison
    os.makedirs(_HELIX_DEBUG_DIR, exist_ok=True)
    torch.save(output, f"{_HELIX_DEBUG_DIR}/rank{rank}_test_output.pt")
    torch.save(ref_output, f"{_HELIX_DEBUG_DIR}/rank{rank}_ref_output.pt")
    print(f"[DEBUG] Rank {rank}: test output shape={output.shape}, "
          f"ref output shape={ref_output.shape}")
    print(f"[DEBUG] Rank {rank}: test output[0,0,:16]={output[0,0,:16].tolist()}")
    print(f"[DEBUG] Rank {rank}: ref  output[0,0,:16]={ref_output[0,0,:16].tolist()}")
    print(f"[DEBUG] Rank {rank}: diff[0,0,:16]={(output[0,0,:16] - ref_output[0,0,:16]).tolist()}")

    # Detailed error analysis: check which dimension ranges have errors
    if output.shape == ref_output.shape:
        abs_diff = (output[0, 0] - ref_output[0, 0]).abs()
        hidden = output.shape[-1]
        quarter = hidden // 4
        for q_idx in range(4):
            q_start = q_idx * quarter
            q_end = (q_idx + 1) * quarter
            q_max = abs_diff[q_start:q_end].max().item()
            q_mean = abs_diff[q_start:q_end].mean().item()
            print(f"[DIAG] Rank {rank}: output dim range [{q_start}:{q_end}]: "
                  f"max_diff={q_max:.6f}, mean_diff={q_mean:.6f}")

    # Every rank should have the same output and checks against the reference
    atol, rtol = scenario.atol, scenario.rtol
    mismatch_count = 0
    for ref_step in range(scenario.ref_steps):
        for b in range(scenario.batch):
            mismatch_count += _error_report(
                output[ref_step, b],
                ref_output[ref_step, b],
                atol,
                rtol,
                f"Rank {rank} {world_size}-GPU step {ref_step}, batch {b}",
            )

    ratio_mismatch = mismatch_count / output.numel()
    print(
        f"Rank {rank} {world_size}-GPU: {mismatch_count}/{output.numel()} mismatches: {ratio_mismatch}"
    )
    return ratio_mismatch


@torch.inference_mode
def _full_test_multi_gpu(
    rank: int,
    world_size: int,
    scenario: Scenario,
    gen_steps: int,
    use_nccl_for_alltoall: bool = False,
    fifo_version: int = 2,
):
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=None,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
    )
    torch.manual_seed(42)
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    input_gen = torch.empty(
        scenario.batch, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    position_ids_ctx = torch.arange(scenario.ctx_len, dtype=torch.int, device="cuda").repeat(
        scenario.batch
    )
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )

    if DISABLE_ROPE_FOR_DEBUG:
        print(f"[DEBUG] Rank {rank}: *** ROPE IS DISABLED FOR DEBUGGING ***")
        pos_embd_params = None
    else:
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(rope_config),
            is_neox=True,
        )

    attn = Attention(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        enable_helix_test=True,
    ).cuda()

    # DEBUG: Print reference attention config
    print(f"[DEBUG] Rank {rank}: REF Attention module - "
          f"num_heads={attn.num_heads}, num_kv_heads={attn.num_key_value_heads}, "
          f"num_heads_tp_cp={attn.num_heads_tp_cp}, head_dim={attn.head_dim}, "
          f"q_size={attn.q_size}, kv_size={attn.kv_size}, "
          f"rope_fusion={attn.rope_fusion}, "
          f"support_fused_qkv={attn.support_fused_qkv}, "
          f"cp_size={attn.cp_size}, tp_size={attn.tp_size}, "
          f"has_cp_helix={attn.mapping.has_cp_helix()}")

    _generate_random_weights(attn)
    weights = attn.state_dict()

    # DEBUG: Print weight shapes and values
    print(f"[DEBUG] Rank {rank}: weights keys={list(weights.keys())}")
    for k_name, v_val in weights.items():
        print(f"[DEBUG] Rank {rank}: weight '{k_name}' shape={v_val.shape}, "
              f"first_vals={v_val.flatten()[:4].tolist()}")

    # Up to this point, all ranks should have the same tensors because the seed
    # is the same. Now we run the reference Attention on rank 0.
    if rank == 0:
        ref_mapping = Mapping(world_size=1, tp_size=1, rank=0)
        ref_kv_cache_manager, ref_attn_metadata = _setup_kv_and_metadata(
            scenario, ref_mapping, gen_steps
        )
        # Hook to capture attention output before o_proj (all heads)
        ref_pre_oproj = {}
        def _capture_pre_oproj(module, args, kwargs=None):
            # args[0] is the input tensor to o_proj
            inp = args[0] if args else (kwargs.get('input', None) if kwargs else None)
            if inp is not None:
                ref_pre_oproj['value'] = inp.detach().clone()
        hook_handle = attn.o_proj.register_forward_pre_hook(_capture_pre_oproj)

        # Context step
        print(f"[DEBUG] Rank {rank}: Running REF context step, "
              f"input_ctx.shape={input_ctx.shape}, "
              f"position_ids_ctx[:8]={position_ids_ctx[:8].tolist()}")
        attn(position_ids_ctx, input_ctx, ref_attn_metadata)
        ref_outputs = []
        start = time.time()

        use_cuda_graph = gen_steps > scenario.ref_steps
        graph = None
        graph_output = None

        for step in range(gen_steps):
            for req_id in range(scenario.batch):
                ref_kv_cache_manager.impl.add_token(req_id)
            if step == 0:
                ref_attn_metadata = get_attention_backend("TRTLLM").Metadata(
                    seq_lens=torch.tensor([1] * scenario.batch, dtype=torch.int),
                    request_ids=list(range(scenario.batch)),
                    max_num_requests=scenario.batch,
                    num_contexts=0,
                    prompt_lens=[scenario.ctx_len] * scenario.batch,
                    max_num_tokens=scenario.ctx_len,
                    kv_cache_manager=ref_kv_cache_manager,
                    kv_cache_params=KVCacheParams(
                        use_cache=True,
                        num_cached_tokens_per_seq=[
                            scenario.ctx_len + step for _ in range(scenario.batch)
                        ],
                    ),
                )
            else:
                ref_attn_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        scenario.ctx_len + step for _ in range(scenario.batch)
                    ],
                )
            ref_attn_metadata.prepare()

            if step == 0:
                print(f"[DEBUG] Rank {rank}: REF gen step {step}, "
                      f"position_ids_gen={position_ids_gen.tolist()}, "
                      f"input_gen[0,:8]={input_gen[0,:8].tolist()}, "
                      f"cached_tokens={scenario.ctx_len + step}")

            if not use_cuda_graph:
                result = attn(position_ids_gen, input_gen, ref_attn_metadata)
                if step < scenario.ref_steps:
                    ref_outputs.append(result)
                    # DEBUG: Save reference output and pre-o_proj attention output
                    os.makedirs(_HELIX_DEBUG_DIR, exist_ok=True)
                    torch.save(result, f"{_HELIX_DEBUG_DIR}/ref_output_step{step}.pt")
                    if 'value' in ref_pre_oproj:
                        ref_attn_out = ref_pre_oproj['value']
                        torch.save(ref_attn_out, f"{_HELIX_DEBUG_DIR}/ref_pre_oproj_step{step}.pt")
                        print(f"[DEBUG] Rank {rank}: REF pre-o_proj shape={ref_attn_out.shape}, "
                              f"values[0,:16]={ref_attn_out[0,:16].tolist()}")
                        print(f"[DEBUG] Rank {rank}: REF pre-o_proj first_half[0,:8]={ref_attn_out[0,:8].tolist()}, "
                              f"second_half[0,{ref_attn_out.shape[-1]//2}:{ref_attn_out.shape[-1]//2+8}]="
                              f"{ref_attn_out[0,ref_attn_out.shape[-1]//2:ref_attn_out.shape[-1]//2+8].tolist()}")
                print(f"Ref result: {result[0, :8]} / {result[-1, -8:]}")
                position_ids_gen += 1
                continue

            if step == 0:
                print("Creating CUDA graph and capturing")
                ref_attn_metadata = ref_attn_metadata.create_cuda_graph_metadata(
                    max_batch_size=scenario.batch
                )
                ref_attn_metadata.prepare()

                for _ in range(2):
                    result = attn(position_ids_gen, input_gen, ref_attn_metadata)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = attn(position_ids_gen, input_gen, ref_attn_metadata)
                result = graph_output
            elif step == scenario.ref_steps:
                start = time.time()
            graph.replay()
            result = graph_output
            position_ids_gen += 1
            if step < scenario.ref_steps:
                ref_outputs.append(result)
        end = time.time()
        if gen_steps == scenario.ref_steps:
            avg_gen_time = float("inf")
        else:
            avg_gen_time = (end - start) / (gen_steps - scenario.ref_steps)
        throughput = scenario.batch / avg_gen_time
        print(
            f"Time taken for {gen_steps - scenario.ref_steps} steps: "
            f"{end - start} s, throughput: {throughput} Attn/s"
        )
        ref_output = torch.stack(ref_outputs, dim=0)
        hook_handle.remove()

        # Save ref pre-oproj for comparison in distributed ranks
        ref_pre_oproj_tensor = ref_pre_oproj.get('value', None)
        if ref_pre_oproj_tensor is not None:
            print(f"[DEBUG] Rank {rank}: Saved REF pre-o_proj tensor, shape={ref_pre_oproj_tensor.shape}")
        ref_kv_cache_manager.shutdown()
    else:
        ref_output = torch.empty(
            scenario.ref_steps,
            scenario.batch,
            scenario.hidden_size,
            dtype=scenario.dtype,
            device="cuda",
        )
        ref_pre_oproj_tensor = None

    # Distributed mapping for helix
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    # Broadcast reference output from rank 0 to all ranks
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]

    # Broadcast ref pre-oproj tensor from rank 0 to all ranks
    if ref_pre_oproj_tensor is None:
        # Create placeholder on non-zero ranks
        ref_pre_oproj_tensor = torch.empty(
            scenario.batch, scenario.num_heads * scenario.head_dim,
            dtype=scenario.dtype, device="cuda"
        )
    ref_pre_oproj_all = cp_allgather(ref_pre_oproj_tensor, mapping=mapping, dim=0)
    ref_pre_oproj_tensor = ref_pre_oproj_all.view(world_size, *ref_pre_oproj_tensor.shape)[0]
    print(f"[DEBUG] Rank {rank}: ref_pre_oproj_tensor shape={ref_pre_oproj_tensor.shape}, "
          f"values[0,:8]={ref_pre_oproj_tensor[0,:8].tolist()}")

    test_params = (
        input_ctx,
        input_gen,
        position_ids_ctx,
        weights,
        pos_embd_params,
    )
    return _run_attention_distributed(
        rank, world_size, scenario, mapping, test_params, ref_output, gen_steps,
        ref_pre_oproj=ref_pre_oproj_tensor,
    )


def _run_single_rank(func, *args, **kwargs):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred. Original traceback is\n{tb}\n")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario", test_scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize("comms_medium", ["nccl", "fifo_v1", "fifo_v2"])
def test_mha_helix_distributed(
    scenario: Scenario,
    comms_medium: str,
    gen_steps: Optional[int] = None,
    max_mismatch_ratio: float = 0.02,
    mismatch_ratios: Optional[List[float]] = None,
):
    world_size = 2
    print(f"Testing with comms_medium={comms_medium}.")
    gen_steps = scenario.ref_steps if gen_steps is None else gen_steps

    if comms_medium == "nccl":
        use_nccl_for_alltoall = True
        fifo_version = 2
    elif comms_medium == "fifo_v1":
        use_nccl_for_alltoall = False
        fifo_version = 1
    elif comms_medium == "fifo_v2":
        use_nccl_for_alltoall = False
        fifo_version = 2
    else:
        raise ValueError(f"Unknown comms_medium: {comms_medium}")

    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            _run_single_rank,
            *zip(
                *[
                    (
                        _full_test_multi_gpu,
                        world_size,
                        scenario,
                        gen_steps,
                        use_nccl_for_alltoall,
                        fifo_version,
                    )
                ]
                * world_size
            ),
        )
        if mismatch_ratios is None:
            for ratio_mismatch in results:
                assert ratio_mismatch <= max_mismatch_ratio
        else:
            mismatch_ratios.extend(results)


if __name__ == "__main__":
    for comms_medium in ["fifo_v1", "fifo_v2", "nccl"]:
        print(f"\n{'=' * 60}")
        print(f"Testing with comms_medium={comms_medium}")
        print(f"{'=' * 60}\n")
        for scenario in all_scenarios[:8]:
            timing_steps = 256
            gen_steps = scenario.ref_steps + timing_steps
            print(f"Running scenario: {scenario} and timing {timing_steps} steps")
            mismatch_ratios = []
            test_mha_helix_distributed(
                scenario,
                comms_medium=comms_medium,
                gen_steps=gen_steps,
                mismatch_ratios=mismatch_ratios,
            )
            if any(mismatch > 0 for mismatch in mismatch_ratios):
                print(f"Numerical test failed with mismatch ratios: {mismatch_ratios}")
            else:
                print("Numerical test passed")
