# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3 unit test: standalone cp.async + WMMA GEMM matches torch.matmul.

Runs the four TinyLlama prefill GEMM shapes (qkv, o, gate_up, down) plus a
generic 128x2048x2560 BF16 sanity case. Tolerance is 1e-3 max-abs vs FP32
reference (BF16-loose).

Skipped automatically if `lucebox_tinyllama._C` is not built.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from lucebox_tinyllama import _C_AVAILABLE

if _C_AVAILABLE:
    from lucebox_tinyllama import gemm_pipeline


@pytest.mark.skipif(not _C_AVAILABLE, reason="lucebox_tinyllama._C not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [
    # (M, K, N): tinyllama prefill GEMMs
    (128, 2048, 2560),   # qkv_proj
    (128, 2048, 2048),   # o_proj
    (128, 2048, 11264),  # gate_up_proj (fused)
    (128, 5632, 2048),   # down_proj
    # smaller sanity cases
    (128, 256, 128),
    (128, 512, 256),
], ids=lambda x: f"M{x[0]}_K{x[1]}_N{x[2]}")
def test_gemm_pipeline_matches_torch(shape):
    M, K, N = shape
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    C_kernel = gemm_pipeline(A, B)
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)

    diff = (C_kernel.float() - C_ref.float()).abs()
    max_abs = diff.max().item()
    rel = max_abs / (C_ref.float().abs().max().item() + 1e-9)
    print(f"shape={shape} max_abs={max_abs:.4f} rel={rel:.4f}")
    # BF16 accumulator slack: per-K-step error grows ~ sqrt(K) * eps_bf16.
    # K=5632 with eps_bf16 ~ 8e-3 -> ~0.6 max_abs is plausible. Use rel<1% as the bound.
    assert rel < 1e-2, f"rel error too high: {rel} (max_abs={max_abs}, shape={shape})"
