# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the native BGMV MoE LoRA ops.

These exercise ``torch.ops.trtllm.bgmv_moe_shrink`` / ``bgmv_moe_expand`` (the
routed-expert MoE LoRA building blocks ported from FlashInfer's bgmv_moe into
TRT-LLM's native kernels, see cpp/tensorrt_llm/kernels/bgmvMoe/) against a plain
PyTorch reference.

Requires a CUDA GPU and the built TensorRT-LLM C++ extension. The BGMV kernels
target sm_80+ (tuned for sm_90). `feat` dims must be in the compiled list in
moeBgmvKernels.cuh (e.g. 768, 2048); rank in {8, 16, 32, 64}.
"""

import pytest
import torch

_TRTLLM_HAS_BGMV = (hasattr(torch.ops, "trtllm")
                    and hasattr(torch.ops.trtllm, "bgmv_moe_shrink")
                    and hasattr(torch.ops.trtllm, "bgmv_moe_expand"))

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_HAS_BGMV,
    reason="Requires CUDA and the built TensorRT-LLM bgmv_moe ops.",
)


def _fill_w_ptr(w_ptr, weights, num_experts, slice_id):
    """Mirror of fill_w_ptr: populate w_ptr[slice_id, :] with per-expert base
    pointers into a [max_loras, num_experts, *, *] weight bank; return the
    element stride between adapters (stride(0))."""
    base = weights.data_ptr()
    expert_stride_bytes = weights.stride(1) * weights.element_size()
    arange = torch.arange(num_experts, dtype=torch.int64, device=weights.device)
    w_ptr[slice_id, :num_experts] = arange * expert_stride_bytes + base
    return weights.stride(0)


@requires_cuda_and_op
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [8, 16])
def test_bgmv_moe_shrink_matches_reference(dtype, rank):
    """y[slice, pair, rank] += x[token] @ A[expert, lora]^T (per-token input)."""
    torch.manual_seed(0)
    device = "cuda"
    num_experts, top_k, num_tokens = 4, 2, 6
    feat_in = 768
    num_slices = 2
    max_loras = 3

    x = torch.randn(num_tokens, feat_in, dtype=dtype, device=device) * 0.1
    # A bank: [max_loras, num_experts, rank, feat_in]
    a = torch.randn(num_slices, max_loras, num_experts, rank, feat_in, dtype=dtype, device=device) * 0.05

    # Per-token routing: expert ids per (token, slot) and per-token adapter id.
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    lora_ids = torch.randint(-1, max_loras, (num_tokens, ), device=device)
    num_pairs = num_tokens * top_k
    token_per_pair = torch.arange(num_tokens, device=device).repeat_interleave(top_k).to(torch.int64)
    expert_per_pair = topk_ids.reshape(-1).to(torch.int64)

    w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride = 0
    for s in range(num_slices):
        lora_stride = _fill_w_ptr(w_ptr, a[s], num_experts, s)

    y = torch.zeros(num_slices, num_pairs, rank, dtype=dtype, device=device)
    torch.ops.trtllm.bgmv_moe_shrink(y, x, w_ptr, token_per_pair, expert_per_pair,
                                     lora_ids.to(torch.int64), lora_stride, False)

    ref = torch.zeros(num_slices, num_pairs, rank, dtype=torch.float32, device=device)
    for s in range(num_slices):
        for p in range(num_pairs):
            t = int(token_per_pair[p])
            l = int(lora_ids[t])
            if l < 0:
                continue
            e = int(expert_per_pair[p])
            ref[s, p] = (a[s, l, e].float() @ x[t].float())
    torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=2e-2)


@requires_cuda_and_op
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bgmv_moe_expand_finalize_matches_reference(dtype):
    """finalize=True: y[token, feat] += w[pair] * (shrink[pair] @ B[expert, lora]^T)."""
    torch.manual_seed(1)
    device = "cuda"
    num_experts, top_k, num_tokens = 4, 2, 6
    rank = 8
    feat_out = 768
    num_slices = 1
    max_loras = 3

    shrink = torch.randn(num_slices, num_tokens * top_k, rank, dtype=dtype, device=device) * 0.1
    b = torch.randn(num_slices, max_loras, num_experts, feat_out, rank, dtype=dtype, device=device) * 0.05

    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    topk_w = torch.rand(num_tokens, top_k, device=device)
    lora_ids = torch.randint(-1, max_loras, (num_tokens, ), device=device)
    num_pairs = num_tokens * top_k
    token_per_pair = torch.arange(num_tokens, device=device).repeat_interleave(top_k).to(torch.int64)
    expert_per_pair = topk_ids.reshape(-1).to(torch.int64)

    w_ptr = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
    lora_stride = 0
    for s in range(num_slices):
        lora_stride = _fill_w_ptr(w_ptr, b[s], num_experts, s)
    slice_start_loc = torch.zeros(num_slices, dtype=torch.int64, device=device)

    y = torch.zeros(num_tokens, feat_out, dtype=torch.float32, device=device)
    torch.ops.trtllm.bgmv_moe_expand(y, shrink, w_ptr, token_per_pair, expert_per_pair,
                                     topk_w.reshape(-1).float().contiguous(), lora_ids.to(torch.int64),
                                     slice_start_loc, feat_out, lora_stride, True)

    ref = torch.zeros(num_tokens, feat_out, dtype=torch.float32, device=device)
    for p in range(num_pairs):
        t = int(token_per_pair[p])
        l = int(lora_ids[t])
        if l < 0:
            continue
        e = int(expert_per_pair[p])
        contrib = b[0, l, e].float() @ shrink[0, p].float()
        ref[t] += float(topk_w.reshape(-1)[p]) * contrib
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)
