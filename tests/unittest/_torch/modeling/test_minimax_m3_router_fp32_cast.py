# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 router input cast: the Triton path is bitwise ``.to(torch.float32)``, and the router arch gates."""

import os

import pytest
import torch

from tensorrt_llm._torch.models import modeling_minimaxm3 as m3

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA op")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


@cuda
@pytest.mark.parametrize("tokens", [2048, 2049, 8192, 16384])
def test_router_cast_bitwise(tokens):
    g = torch.Generator(device="cuda").manual_seed(tokens)
    x = torch.randn(tokens, 6144, device="cuda", generator=g).to(torch.bfloat16)
    x[0, :6] = torch.tensor(
        [float("inf"), float("-inf"), float("nan"), -0.0, 1e-40, -3e38], dtype=torch.bfloat16
    )
    out = m3.minimax_m3_router_input_fp32(x)
    assert out.dtype == torch.float32 and out.shape == x.shape
    assert torch.equal(_bits(out), _bits(x.to(torch.float32)))


@cuda
def test_router_cast_fallbacks():
    x = torch.randn(4096, 2 * 6144, device="cuda").to(torch.bfloat16)
    for t in (x[:, :6144], x[:1024, :6144].contiguous(), x[:, :6144].half()):
        out = m3.minimax_m3_router_input_fp32(t)
        assert torch.equal(_bits(out), _bits(t.to(torch.float32)))


@pytest.mark.parametrize(
    "sm_version,archs,enabled,expected",
    [
        (107, (107,), True, True),
        (103, (100, 103, 107), True, True),
        (100, (100, 103, 107), True, True),
        (103, (107,), True, False),
        (90, (100, 103, 107), True, False),
        (107, (107,), False, False),
    ],
)
def test_router_cast_gate(monkeypatch, sm_version, archs, enabled, expected):
    monkeypatch.setattr(m3, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(m3, "_SM107_ROUTER_FP32_CAST", enabled)
    monkeypatch.setattr(m3, "_ROUTER_FP32_CAST_ARCHS", frozenset(archs))
    gate = m3.MiniMaxM3Gate(hidden_size=64, num_experts=8, top_k=2, routed_scaling_factor=1.0)
    assert gate._sm107_fp32_cast is expected


@pytest.mark.parametrize(
    "sm_version,tf32_archs,tc_archs,tf32,tc",
    [
        (107, (107,), (107,), True, True),
        (103, (107,), (107,), False, False),
        (100, (100, 107), (100, 107), True, True),
        (103, (100, 107), (100, 107), False, False),
    ],
)
def test_router_tf32_and_tc_gen_gates(monkeypatch, sm_version, tf32_archs, tc_archs, tf32, tc):
    # The fixed-tile TF32 CTX router and the generation-step TF32 tensor-core router: SM107 always, SM100 / SM103
    # only where their ARCHS env lists them.
    monkeypatch.setattr(m3, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(m3, "_CTX_ROUTER_TF32_FIXED", True)
    monkeypatch.setattr(m3, "_ROUTER_TC_GEN", True)
    monkeypatch.setattr(m3, "_CTX_ROUTER_TF32_FIXED_ARCHS", frozenset(tf32_archs))
    monkeypatch.setattr(m3, "_ROUTER_TC_GEN_ARCHS", frozenset(tc_archs))
    gate = m3.MiniMaxM3Gate(hidden_size=1024, num_experts=128, top_k=4, routed_scaling_factor=1.0)
    assert gate._ctx_router_tf32 is tf32
    assert gate._router_tc_gen is tc


def test_router_blackwell_default_arch_sets():
    # Unset, the TF32 routers stay off on SM100 / SM103 (opt-in) and the exact fp32 cast is on.
    from tensorrt_llm._torch import blackwell_rubin_opts

    if "TRTLLM_CTX_ROUTER_TF32_FIXED_ARCHS" not in os.environ:
        assert m3._CTX_ROUTER_TF32_FIXED_ARCHS == {107}
    if "TRTLLM_M3_ROUTER_TC_GEN_ARCHS" not in os.environ:
        assert m3._ROUTER_TC_GEN_ARCHS == {107}
    if "TRTLLM_M3_ROUTER_FP32_CAST_ARCHS" not in os.environ and blackwell_rubin_opts.UMBRELLA_ON:
        assert m3._ROUTER_FP32_CAST_ARCHS == {100, 103, 107}
