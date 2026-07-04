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
"""Multi-LoRA delta builders for the routed TRTLLM-gen fused MoE (single node).

Ported from FlashInfer's flashinfer/fused_moe/moe_lora_delta.py (Apache-2.0),
but driving TRT-LLM's *native* BGMV ops (``torch.ops.trtllm.bgmv_moe_shrink`` /
``bgmv_moe_expand`` from cpp/tensorrt_llm/kernels/bgmvMoe) instead of FlashInfer.
No FlashInfer dependency.

These produce the per-(token, expert) LoRA deltas for the two layers of a SwiGLU
MoE expert FFN:

  * FC1 (gate_up GLU) -> ``bgmv_moe_gemm1_lora_delta`` -> ``[T, k, 2I]`` bf16, fed
    INTO the FP8 block-scale MoE runner as ``gemm1_lora_delta`` (added pre-SwiGLU;
    expand ``finalize=False`` -> per-pair, unweighted).
  * FC2 (down_proj) -> ``bgmv_moe_gemm2_lora_delta`` -> ``[T, H]``, ADDED to the MoE
    output (shrink ``per_pair_input=True`` over the gathered post-SwiGLU activation +
    expand ``finalize=True`` -> routing-weighted combine).

The LoRA weights are **caller-managed** and use TRT-LLM's per-adapter pointer
model (not FlashInfer's per-expert + adapter-stride model): the builders take
``[num_slices, max_loras]`` int64 base-pointer tables (``w_ptr``), where
``w_ptr[slice, lora_id]`` points to that adapter's contiguous
``[num_experts, rank, feat]`` bank. The BGMV kernel adds the per-expert offset
internally, so there is no ``lora_stride``. At runtime these pointers come
straight from the PEFT cache (``weight_pointers`` / ``h_b_ptrs``); :func:`fill_w_ptr`
builds them over a contiguous ``[max_loras, num_experts, *, *]`` bank for tests.
"""

from typing import Tuple

import torch


def fill_w_ptr(
    w_ptr: torch.Tensor,
    weights: torch.Tensor,
    slice_id: int,
) -> None:
    """Fill ``w_ptr[slice_id, 0:max_loras]`` with per-**adapter** base pointers.

    Per-adapter pointer model (matches TRT-LLM's PEFT layout, not FlashInfer's):
    ``weights`` is a ``[max_loras, num_experts, rank, feat]`` bank where each
    adapter's ``[num_experts, rank, feat]`` slice is contiguous, and
    ``w_ptr[slice_id, lora_id]`` points to that adapter's slice base. The BGMV
    kernel adds the per-expert offset (compile-time ``feat_in*feat_out``)
    internally, so no adapter stride is returned.

    At runtime the eager backend builds ``w_ptr`` directly from the PEFT cache's
    per-adapter base pointers (``weight_pointers`` / ``h_b_ptrs``) instead of
    calling this; this helper is mainly for tests over contiguous banks.
    """
    max_loras = weights.shape[0]
    base_ptr = weights.data_ptr()
    adapter_stride_bytes = weights.stride(0) * weights.element_size()
    arange = torch.arange(max_loras, dtype=torch.int64, device=weights.device)
    w_ptr[slice_id, :max_loras] = arange * adapter_stride_bytes + base_ptr


def _expanded_pairs(
    topk_ids: torch.Tensor, lora_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build token-major (``p = token*top_k + slot``) per-pair routing arrays.

    ``lora_ids`` stays per-token ``[T]`` (the kernels look up the adapter via the
    real token carried by ``sorted_token_ids``). Returns int64 ``token_per_pair``
    ``[P]`` and ``expert_per_pair`` ``[P]`` on ``topk_ids.device``.
    """
    del lora_ids  # only used by the caller; kept for signature parity/clarity
    T, k = topk_ids.shape
    device = topk_ids.device
    token_per_pair = torch.arange(
        T, device=device, dtype=torch.int64).repeat_interleave(k)
    expert_per_pair = topk_ids.reshape(-1).to(torch.int64)
    return token_per_pair, expert_per_pair


def bgmv_moe_gemm1_lora_delta(
    hidden_states: torch.Tensor,
    w_ptr_a: torch.Tensor,
    w_ptr_b: torch.Tensor,
    topk_ids: torch.Tensor,
    lora_ids: torch.Tensor,
    rank: int,
    intermediate_size: int,
    *,
    lora_dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FC1 (gate_up) LoRA delta in the ``[T, top_k, 2*I]`` layout consumed by the
    FP8 block-scale MoE runner's ``gemm1_lora_delta``.

    For each routed pair ``(token t, slot j)`` with expert ``e = topk_ids[t, j]``
    and adapter ``l = lora_ids[t]`` (skipped when ``l < 0``)::

        delta[t, j] = scale * concat(B_gate[l,e] @ (A_gate[l,e] @ x[t]),
                                     B_up[l,e]   @ (A_up[l,e]   @ x[t]))

    Unweighted and per-(token, slot) (added before the nonlinear SwiGLU).
    """
    assert w_ptr_a.shape[0] == 2 and w_ptr_b.shape[0] == 2, (
        "FC1 LoRA is a 2-slice (gate, up) GLU projection")
    T, _ = hidden_states.shape
    k = topk_ids.shape[1]
    P = T * k
    inter = intermediate_size
    device = hidden_states.device

    token_per_pair, expert_per_pair = _expanded_pairs(topk_ids, lora_ids)
    lora_idx = lora_ids.to(torch.int64)
    x = hidden_states.to(lora_dtype)

    # Shrink: x @ A -> [2, P, rank]. Per-token input read (default mode).
    shrink_out = torch.zeros(2, P, rank, dtype=lora_dtype, device=device)
    torch.ops.trtllm.bgmv_moe_shrink(shrink_out, x, w_ptr_a, token_per_pair,
                                     expert_per_pair, lora_idx, False)

    # Expand: shrink_out @ B -> [P, 2I], per-pair unweighted store; zeroed so
    # skipped pairs stay 0. topk_weights is ignored (finalize=False) but must be
    # a valid [P] float32 tensor.
    slice_start_loc = torch.tensor([0, inter], dtype=torch.int64, device=device)
    unit_w = torch.ones(P, dtype=torch.float32, device=device)
    y = torch.zeros(P, 2 * inter, dtype=torch.float32, device=device)
    torch.ops.trtllm.bgmv_moe_expand(y, shrink_out, w_ptr_b, token_per_pair,
                                     expert_per_pair, unit_w, lora_idx,
                                     slice_start_loc, inter, False)

    if scale != 1.0:
        y = y * scale
    return y.view(T, k, 2 * inter).to(out_dtype)


def bgmv_moe_gemm2_lora_delta(
    gemm1_activation_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    w_ptr_a: torch.Tensor,
    w_ptr_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_ids: torch.Tensor,
    rank: int,
    hidden_size: int,
    *,
    lora_dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FC2 (down_proj) LoRA delta, to be ADDED to the MoE output.

    Consumes the permuted post-SwiGLU activation returned by the FP8 block-scale
    MoE runner (called with ``gemm1_lora_delta`` set and ``do_finalize=True``).
    For each routed pair ``(token t, slot j)`` with expert ``e`` and adapter ``l``::

        delta[t] = scale * sum_j w[t,j] * (B_down[l,e] @ (A_down[l,e] @ a[t,j]))

    Weighted and combined over experts (added after FC2, post-combine).
    """
    assert w_ptr_a.shape[0] == 1 and w_ptr_b.shape[0] == 1, (
        "FC2 LoRA is a single down-projection slice")
    T, k = topk_ids.shape
    P = T * k
    inter = gemm1_activation_output.shape[1]
    hidden = hidden_size
    device = gemm1_activation_output.device

    token_per_pair, expert_per_pair = _expanded_pairs(topk_ids, lora_ids)
    lora_idx = lora_ids.to(torch.int64)

    # Gather permuted activation into expanded [P, I] order; inactive slots stay 0.
    perm = expanded_idx_to_permuted_idx.to(torch.int64)
    valid = perm >= 0
    a_exp = torch.zeros(P, inter, dtype=lora_dtype, device=device)
    a_exp[valid] = gemm1_activation_output[perm[valid]].to(lora_dtype)

    # Shrink: a_exp @ A_down -> [1, P, rank]. Per-pair input read.
    shrink_out = torch.zeros(1, P, rank, dtype=lora_dtype, device=device)
    torch.ops.trtllm.bgmv_moe_shrink(shrink_out, a_exp, w_ptr_a, token_per_pair,
                                     expert_per_pair, lora_idx, True)

    # Expand (finalize): shrink_out @ B_down -> [T, H], routing-weighted combine.
    slice_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    topk_w = topk_weights.reshape(P).to(torch.float32)
    y = torch.zeros(T, hidden, dtype=torch.float32, device=device)
    torch.ops.trtllm.bgmv_moe_expand(y, shrink_out, w_ptr_b, token_per_pair,
                                     expert_per_pair, topk_w, lora_idx,
                                     slice_start_loc, hidden, True)

    if scale != 1.0:
        y = y * scale
    return y.to(out_dtype)
