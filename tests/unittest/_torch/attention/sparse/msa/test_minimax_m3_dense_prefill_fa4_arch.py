# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 FP8 dense prefill: FA4 engages on SM107 and the SMs of TRTLLM_MSA_DENSE_FA4_ARCHS (default SM103)
(CPU-level; FA4 itself is stubbed)."""

import os

import pytest

from tensorrt_llm._torch.attention.backends.fmha import msa_prefill

HEAD_DIM, PAGE = 128, 128


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "sm, archs, expected",
    [
        (107, (103, 107), True),
        (103, (103, 107), True),
        (100, (103, 107), False),
        (100, (100, 103, 107), True),
        (103, (107,), False),
        (107, (107,), True),
        (110, (100, 103, 107), False),
        (90, (100, 103, 107), False),
    ],
)
def test_dense_prefill_fa4_engages_on_listed_archs(
    monkeypatch: pytest.MonkeyPatch, sm: int, archs: tuple, expected: bool
) -> None:
    # FA4 is deterministic but not bitwise to fmha_sm100, so architectures outside the arch set keep the stock
    # dense prefill even with a usable FA4 installed. SM107 is the positive control.
    monkeypatch.setattr(msa_prefill, "_dense_fa4_forward", lambda: object())
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: sm)
    monkeypatch.setattr(msa_prefill, "_DENSE_FA4_ARCHS", frozenset(archs))
    assert msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE) is expected


@pytest.mark.cpu_only
def test_dense_prefill_fa4_default_archs() -> None:
    from tensorrt_llm._torch import blackwell_rubin_opts

    if blackwell_rubin_opts.UMBRELLA_ON and "TRTLLM_MSA_DENSE_FA4_ARCHS" not in os.environ:
        assert msa_prefill._DENSE_FA4_ARCHS == {103, 107}
    assert 107 in msa_prefill._DENSE_FA4_ARCHS


@pytest.mark.cpu_only
def test_dense_prefill_fa4_needs_a_usable_fa4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(msa_prefill, "_dense_fa4_forward", lambda: None)
    monkeypatch.setattr(msa_prefill, "get_sm_version", lambda: 107)
    assert not msa_prefill.dense_prefill_fa4_enabled(True, HEAD_DIM, PAGE)
