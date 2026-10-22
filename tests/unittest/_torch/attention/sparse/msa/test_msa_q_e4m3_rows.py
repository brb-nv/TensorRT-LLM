# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8-KV prefill's E4M3 row cast (msa_prefill._q_to_e4m3, TRTLLM_CTX_Q_E4M3_ROWS): its gate, and the bits of
the aten cast wherever that is finite."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
from tensorrt_llm._utils import get_sm_version


def _strided_q(rows, heads=64, dim=128, extra=2 * 4 * 128, device="cuda", scale=1.0):
    g = torch.Generator(device=device).manual_seed(rows)
    packed = torch.randn(rows, heads * dim + extra, device=device, generator=g) * scale
    return packed.to(torch.bfloat16)[:, : heads * dim].view(rows, heads, dim)


@pytest.mark.parametrize(
    "sm, archs, taken",
    [
        (107, (107,), True),
        (103, (100, 103, 107), True),
        (100, (100, 103, 107), True),
        (103, (107,), False),
    ],
)
def test_q_e4m3_rows_gate(monkeypatch, sm, archs, taken):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    monkeypatch.setattr(msa_prefill, "_SM107_Q_E4M3_ROWS", True)
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_ARCHS", frozenset(archs))
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_BLACKWELL_MIN_ELEMS", 0)
    launched = []
    monkeypatch.setattr(msa_prefill, "_rows_to_e4m3_kernel", _Recorder(launched))
    q = _strided_q(4)
    msa_prefill._q_to_e4m3(q)
    assert bool(launched) is taken


@pytest.mark.parametrize(
    "sm, rows, heads, taken",
    [
        (103, 2048, 64, False),  # 2^24 elements: below the Blackwell floor
        (103, 4096, 64, True),  # 2^25
        (100, 8192, 32, True),
        (100, 8192, 16, False),
        (107, 4, 64, True),  # SM107 has no floor
    ],
)
def test_q_e4m3_rows_blackwell_floor(monkeypatch, sm, rows, heads, taken):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    monkeypatch.setattr(msa_prefill, "_SM107_Q_E4M3_ROWS", True)
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_ARCHS", frozenset((100, 103, 107)))
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_BLACKWELL_MIN_ELEMS", 1 << 25)
    launched = []
    monkeypatch.setattr(msa_prefill, "_rows_to_e4m3_kernel", _Recorder(launched))
    msa_prefill._q_to_e4m3(_strided_q(rows, heads=heads))
    assert bool(launched) is taken


class _Recorder:
    def __init__(self, calls):
        self.calls = calls

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append(grid)

        return run


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="SM100/SM103/SM107",
)
@pytest.mark.parametrize("rows", [1, 777, 8192])
def test_q_e4m3_rows_matches_aten(monkeypatch, rows):
    monkeypatch.setattr(msa_prefill, "_SM107_Q_E4M3_ROWS", True)
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_ARCHS", frozenset((100, 103, 107)))
    monkeypatch.setattr(msa_prefill, "_Q_E4M3_ROWS_BLACKWELL_MIN_ELEMS", 0)
    q = _strided_q(
        rows, scale=100.0
    )  # reaches past 448: aten gives NaN there, the row cast saturates
    got = msa_prefill._q_to_e4m3(q)
    want = q.to(torch.float8_e4m3fn)
    assert got.is_contiguous() and got.shape == q.shape
    finite = torch.isfinite(want.float())
    assert torch.equal(got.view(torch.uint8)[finite], want.view(torch.uint8)[finite])
    sat = got.float()[~finite].abs()
    assert bool((sat == 448).all())
