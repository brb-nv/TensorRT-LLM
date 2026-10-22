# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arch sets of the MiniMax-M3 gates widened from SM107 to SM100/SM103 (blackwell_rubin_opts)."""

import importlib

import pytest

from tensorrt_llm._torch import blackwell_rubin_opts


def _reload(monkeypatch, **env):
    for name in ("TRTLLM_M3_BLACKWELL_RUBIN_OPTS", "TRTLLM_TEST_GATE_ARCHS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(blackwell_rubin_opts)


@pytest.fixture(autouse=True)
def _restore():
    yield
    importlib.reload(blackwell_rubin_opts)


def test_default_adds_blackwell(monkeypatch):
    mod = _reload(monkeypatch)
    assert mod.gate_archs("TRTLLM_TEST_GATE_ARCHS") == {100, 103, 107}
    assert mod.gate_archs("TRTLLM_TEST_GATE_ARCHS", default_blackwell=(103,)) == {103, 107}
    assert mod.gate_archs("TRTLLM_TEST_GATE_ARCHS", default_blackwell=()) == {107}


def test_umbrella_off_keeps_rubin_only(monkeypatch):
    mod = _reload(monkeypatch, TRTLLM_M3_BLACKWELL_RUBIN_OPTS="0", TRTLLM_TEST_GATE_ARCHS="100,103")
    assert mod.gate_archs("TRTLLM_TEST_GATE_ARCHS") == {107}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("107", {107}),
        ("103", {103, 107}),
        ("sm_100a, 103f,107", {100, 103, 107}),
        ("", {107}),
        ("90,120", {107}),
    ],
)
def test_env_list(monkeypatch, raw, expected):
    mod = _reload(monkeypatch, TRTLLM_TEST_GATE_ARCHS=raw)
    assert mod.gate_archs("TRTLLM_TEST_GATE_ARCHS") == expected


def test_bad_entry_raises(monkeypatch):
    mod = _reload(monkeypatch, TRTLLM_TEST_GATE_ARCHS="10x3")
    with pytest.raises(ValueError):
        mod.gate_archs("TRTLLM_TEST_GATE_ARCHS")
