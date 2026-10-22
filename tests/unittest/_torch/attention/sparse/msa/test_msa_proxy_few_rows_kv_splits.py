# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate of the few-row KV split count of the MiniMax-M3 indexer proxy plan (SM107; SM100/SM103 with the
segmented proxy)."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend


@pytest.fixture(autouse=True)
def _sm_count(monkeypatch):
    monkeypatch.setattr(msa_backend, "_sm_count", lambda device_index: 212)


@pytest.mark.parametrize(
    "sm_version,enabled,num_rows,expected",
    [
        (107, True, 1, 212),
        (107, True, 2, 106),
        (107, True, 4, 53),
        (107, True, 5, -1),
        (107, True, 128, -1),
        (103, True, 1, -1),
        (107, False, 1, -1),
    ],
)
def test_proxy_kv_splits(monkeypatch, sm_version, enabled, num_rows, expected):
    # Stock arch sets: SM107 only for the segmented proxy, so SM103 keeps the planner's choice.
    monkeypatch.setattr(msa_backend, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(msa_backend, "_SM107_PROXY_FEW_ROWS_KV_SPLITS", enabled)
    monkeypatch.setattr(msa_backend, "_PROXY_AUTO_KV_SPLITS_ARCHS", frozenset((107,)))
    monkeypatch.setattr(msa_backend, "_PROXY_FEW_ROWS_KV_SPLITS_ARCHS", frozenset((100, 103, 107)))
    assert msa_backend.proxy_kv_splits(num_rows, torch.device("cuda", 0)) == expected
    # SM107 ignores the Blackwell cap and the key-length bound.
    if sm_version == 107:
        got = msa_backend.proxy_kv_splits(num_rows, torch.device("cuda", 0), max_kv_len=300)
        assert got == expected


@pytest.mark.parametrize(
    "sm_version,segmented_archs,few_rows_archs,num_rows,max_kv_len,otherwise,expected",
    [
        # Blackwell with the segmented proxy on: capped at 64 and at one split per 256 keys.
        (103, (103, 107), (100, 103, 107), 1, 400_000, -1, 64),
        (103, (103, 107), (100, 103, 107), 4, 400_000, -1, 53),
        (100, (100, 107), (100, 103, 107), 1, 3_000, -1, 12),
        (103, (103, 107), (100, 103, 107), 1, None, -1, 64),
        (103, (103, 107), (100, 103, 107), 5, 400_000, 1, 1),
        (103, (103, 107), (100, 103, 107), 5, 400_000, -1, -1),
        # Without the segmented proxy, or outside the few-row set, the fallback.
        (103, (107,), (100, 103, 107), 1, 400_000, 1, 1),
        (103, (103, 107), (107,), 1, 400_000, 1, 1),
        (90, (100, 103, 107), (100, 103, 107), 1, 400_000, 1, 1),
    ],
)
def test_proxy_kv_splits_blackwell(
    monkeypatch,
    sm_version,
    segmented_archs,
    few_rows_archs,
    num_rows,
    max_kv_len,
    otherwise,
    expected,
):
    monkeypatch.setattr(msa_backend, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(msa_backend, "_SM107_PROXY_FEW_ROWS_KV_SPLITS", True)
    monkeypatch.setattr(msa_backend, "_PROXY_AUTO_KV_SPLITS_ARCHS", frozenset(segmented_archs))
    monkeypatch.setattr(msa_backend, "_PROXY_FEW_ROWS_KV_SPLITS_ARCHS", frozenset(few_rows_archs))
    got = msa_backend.proxy_kv_splits(
        num_rows, torch.device("cuda", 0), max_kv_len=max_kv_len, otherwise=otherwise
    )
    assert got == expected
