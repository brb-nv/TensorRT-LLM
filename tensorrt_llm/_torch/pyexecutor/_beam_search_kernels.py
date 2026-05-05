# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hand-written Triton kernels for beam-search sampling hot paths.

Currently provides ``beam_logprobs_prep_triton`` -- a single-kernel
replacement for the 9-kernel eager sequence inside
``beam_search_sampling_batch``'s ``bss.logprobs_prep`` block.

Replaces (per generation step):
    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs[finished, :] = -inf
    logprobs[finished, 0] = 0
    logprobs += cum_log_probs.unsqueeze(-1)

Avoids the ~5-10 us / call ``torch.compile`` dispatch overhead by issuing a
single ``triton_kernel[grid](...)`` launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1024}, num_warps=4),
        triton.Config({"BLOCK_V": 2048}, num_warps=4),
        triton.Config({"BLOCK_V": 2048}, num_warps=8),
        triton.Config({"BLOCK_V": 4096}, num_warps=8),
        triton.Config({"BLOCK_V": 8192}, num_warps=8),
    ],
    key=["V"],
)
@triton.jit
def _beam_logprobs_prep_kernel(
    LOGITS_PTR,
    FINISHED_PTR,
    CUM_PTR,
    OUT_PTR,
    V,
    LOGITS_STRIDE_B,
    LOGITS_STRIDE_K,
    OUT_STRIDE_B,
    OUT_STRIDE_K,
    FINISHED_STRIDE_B,
    CUM_STRIDE_B,
    BLOCK_V: tl.constexpr,
):
    """One program per (batch, beam) row of length V.

    Pass 1 (online softmax):  scan the row to obtain (max, sum_exp).
    Pass 2 (emit):            log_softmax, apply finished-beam mask
                              (forcing position 0 -> 0 and others -> -inf
                              for finished beams), add per-beam
                              ``cum_log_probs``, store.

    Numerics: pass 1 accumulates in float32 regardless of input dtype
    (matches ``torch.log_softmax``'s upcast-for-stability behavior).
    """
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Per-row scalars.
    is_finished = tl.load(FINISHED_PTR + pid_b * FINISHED_STRIDE_B + pid_k) != 0
    cum = tl.load(CUM_PTR + pid_b * CUM_STRIDE_B + pid_k).to(tl.float32)

    row_logits = LOGITS_PTR + pid_b * LOGITS_STRIDE_B + pid_k * LOGITS_STRIDE_K
    row_out = OUT_PTR + pid_b * OUT_STRIDE_B + pid_k * OUT_STRIDE_K

    NEG_INF = float("-inf")

    # Pass 1: online (max, sum_exp).
    max_val = NEG_INF
    sum_exp = tl.zeros([], dtype=tl.float32)

    n_chunks = tl.cdiv(V, BLOCK_V)
    for c in range(0, n_chunks):
        offs = c * BLOCK_V + tl.arange(0, BLOCK_V)
        m = offs < V
        x = tl.load(row_logits + offs, mask=m, other=NEG_INF).to(tl.float32)
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(max_val, block_max)
        # Rescale running sum to the new max, then add this block's contribution.
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        max_val = new_max

    log_norm = tl.log(sum_exp) + max_val  # log(sum_v exp(logits[v]))

    # Pass 2: emit log_softmax + finished mask + cum_log_probs.
    out_dtype = OUT_PTR.dtype.element_ty
    for c in range(0, n_chunks):
        offs = c * BLOCK_V + tl.arange(0, BLOCK_V)
        m = offs < V
        x = tl.load(row_logits + offs, mask=m, other=0.0).to(tl.float32)
        # log_softmax[v] = logits[v] - log_norm.
        lp = x - log_norm
        # Finished-beam value (before adding cum):
        #   v == 0  -> 0
        #   v >  0  -> -inf
        finished_val = tl.where(offs == 0, 0.0, NEG_INF)
        out = tl.where(is_finished, finished_val, lp) + cum
        tl.store(row_out + offs, out.to(out_dtype), mask=m)


def beam_logprobs_prep_triton(
    logits: torch.Tensor,
    finished_mask: torch.Tensor,
    cum_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Single-launch fusion of log_softmax + finished-beam mask +
    first-token reset + per-beam ``cum_log_probs`` broadcast-add.

    Args:
        logits:        ``[B, K_in, V]`` fp16/bf16/fp32, V-dim contiguous.
        finished_mask: ``[B, K_in]`` bool (or any int dtype where 0 means
                       "not finished", non-zero means "finished").
        cum_log_probs: ``[B, K_in]`` fp32 (matches ``BeamSearchStore``).

    Returns:
        ``[B, K_in, V]`` log probs in ``logits.dtype``.

    Equivalent to (eager reference):
        logprobs = log_softmax(logits, dim=-1)
        logprobs[finished, :] = -inf
        logprobs[finished, 0] = 0
        logprobs += cum_log_probs.unsqueeze(-1)
    """
    assert logits.is_cuda, "Triton kernel requires CUDA tensors"
    assert logits.dim() == 3, f"logits must be 3-D, got {logits.shape}"
    assert logits.stride(-1) == 1, "logits must have contiguous V dim"
    B, K_in, V = logits.shape
    assert finished_mask.shape == (B, K_in), (
        f"finished_mask shape {finished_mask.shape} != ({B}, {K_in})"
    )
    assert cum_log_probs.shape == (B, K_in), (
        f"cum_log_probs shape {cum_log_probs.shape} != ({B}, {K_in})"
    )

    # Triton expects an integer dtype for boolean inputs; promote bool to int8.
    if finished_mask.dtype == torch.bool:
        finished_int = finished_mask.to(torch.int8)
    else:
        finished_int = finished_mask
    # Strides assumed contiguous on (B, K_in) grid; enforce.
    if not finished_int.is_contiguous():
        finished_int = finished_int.contiguous()
    if not cum_log_probs.is_contiguous():
        cum_log_probs = cum_log_probs.contiguous()

    out = torch.empty_like(logits)
    grid = (B, K_in)
    _beam_logprobs_prep_kernel[grid](
        logits,
        finished_int,
        cum_log_probs,
        out,
        V,
        logits.stride(0),
        logits.stride(1),
        out.stride(0),
        out.stride(1),
        finished_int.stride(0),
        cum_log_probs.stride(0),
    )
    return out
