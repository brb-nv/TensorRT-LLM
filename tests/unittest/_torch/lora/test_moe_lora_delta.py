# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the routed-MoE LoRA delta builders (native BGMV, no FlashInfer).

Compares ``bgmv_moe_gemm1_lora_delta`` / ``bgmv_moe_gemm2_lora_delta`` (which drive
``torch.ops.trtllm.bgmv_moe_{shrink,expand}``) against a plain PyTorch reference.

Per-adapter pointer model: ``w_ptr[slice, lora_id]`` points to that adapter's
contiguous ``[num_experts, ...]`` bank; the kernel adds the per-expert offset.

Requires a CUDA GPU and the built TensorRT-LLM C++ extension. ``feat`` dims must be
in the compiled BGMV list (moeBgmvKernels.cuh); rank in {8,16,32,64}.
"""

import pytest
import torch

# CPU-importable: ops resolve at call time.
from tensorrt_llm._torch.modules.fused_moe.moe_lora_delta import (
    bgmv_moe_gemm1_lora_delta, bgmv_moe_gemm2_lora_delta, fill_w_ptr)

_TRTLLM_HAS_BGMV = (hasattr(torch.ops, "trtllm")
                    and hasattr(torch.ops.trtllm, "bgmv_moe_shrink"))

requires_cuda_and_op = pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRTLLM_HAS_BGMV,
    reason="Requires CUDA and the built TensorRT-LLM bgmv_moe ops.",
)


@requires_cuda_and_op
def test_gemm1_lora_delta_matches_reference():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    T, k, num_experts, max_loras = 6, 2, 4, 3
    hidden, inter, rank = 2048, 768, 16
    scale = 0.5

    x = torch.randn(T, hidden, dtype=dtype, device=device) * 0.1
    # A: [max_loras, num_experts, rank, hidden] for gate and up.
    a_gate = torch.randn(max_loras, num_experts, rank, hidden, dtype=dtype, device=device) * 0.02
    a_up = torch.randn(max_loras, num_experts, rank, hidden, dtype=dtype, device=device) * 0.02
    # B: [max_loras, num_experts, inter, rank] for gate and up.
    b_gate = torch.randn(max_loras, num_experts, inter, rank, dtype=dtype, device=device) * 0.02
    b_up = torch.randn(max_loras, num_experts, inter, rank, dtype=dtype, device=device) * 0.02

    topk_ids = torch.randint(0, num_experts, (T, k), device=device)
    lora_ids = torch.randint(-1, max_loras, (T, ), device=device)

    # w_ptr[slice, lora_id] per-adapter base pointers (slice 0 = gate, 1 = up).
    wpa = torch.zeros(2, max_loras, dtype=torch.int64, device=device)
    fill_w_ptr(wpa, a_gate, 0)
    fill_w_ptr(wpa, a_up, 1)
    wpb = torch.zeros(2, max_loras, dtype=torch.int64, device=device)
    fill_w_ptr(wpb, b_gate, 0)
    fill_w_ptr(wpb, b_up, 1)

    delta = bgmv_moe_gemm1_lora_delta(x, wpa, wpb, topk_ids, lora_ids, rank, inter,
                                      scale=scale)
    assert delta.shape == (T, k, 2 * inter)

    ref = torch.zeros(T, k, 2 * inter, dtype=torch.float32, device=device)
    for t in range(T):
        l = int(lora_ids[t])
        if l < 0:
            continue
        for j in range(k):
            e = int(topk_ids[t, j])
            g = b_gate[l, e].float() @ (a_gate[l, e].float() @ x[t].float())
            u = b_up[l, e].float() @ (a_up[l, e].float() @ x[t].float())
            ref[t, j, :inter] = scale * g
            ref[t, j, inter:] = scale * u
    torch.testing.assert_close(delta.float(), ref, atol=5e-2, rtol=5e-2)


@requires_cuda_and_op
def test_gemm2_lora_delta_matches_reference():
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    T, k, num_experts, max_loras = 6, 2, 4, 3
    hidden, inter, rank = 2048, 768, 16
    scale = 0.5
    P = T * k

    # Permuted post-SwiGLU activation: use identity permutation p = token*k+slot.
    act = torch.randn(P, inter, dtype=dtype, device=device) * 0.1
    exp2perm = torch.arange(P, dtype=torch.int64, device=device)

    a_down = torch.randn(max_loras, num_experts, rank, inter, dtype=dtype, device=device) * 0.02
    b_down = torch.randn(max_loras, num_experts, hidden, rank, dtype=dtype, device=device) * 0.02

    topk_ids = torch.randint(0, num_experts, (T, k), device=device)
    topk_w = torch.rand(T, k, device=device)
    lora_ids = torch.randint(-1, max_loras, (T, ), device=device)

    wpa = torch.zeros(1, max_loras, dtype=torch.int64, device=device)
    fill_w_ptr(wpa, a_down, 0)
    wpb = torch.zeros(1, max_loras, dtype=torch.int64, device=device)
    fill_w_ptr(wpb, b_down, 0)

    delta = bgmv_moe_gemm2_lora_delta(act, exp2perm, wpa, wpb, topk_ids, topk_w,
                                      lora_ids, rank, hidden, scale=scale)
    assert delta.shape == (T, hidden)

    ref = torch.zeros(T, hidden, dtype=torch.float32, device=device)
    for t in range(T):
        l = int(lora_ids[t])
        if l < 0:
            continue
        for j in range(k):
            p = t * k + j
            e = int(topk_ids[t, j])
            contrib = b_down[l, e].float() @ (a_down[l, e].float() @ act[p].float())
            ref[t] += float(topk_w[t, j]) * scale * contrib
    torch.testing.assert_close(delta.float(), ref, atol=5e-2, rtol=5e-2)
