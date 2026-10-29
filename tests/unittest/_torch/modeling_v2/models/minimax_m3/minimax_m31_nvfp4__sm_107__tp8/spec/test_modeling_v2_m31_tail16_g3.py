# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL16_G3 on the tp8 target: block draft probabilities stored with minimax_m31_scatter_rows for up to
TAIL16_G3_MAX_REQUESTS requests. CPU: the default, the kill switch and the stock fallbacks. GPU (SM107, op built):
bitwise the stock index_put store (K 3 and 7, padded last dim, NaN / -inf rows) and under graph replay."""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    dspark_s4 as w8,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase


def test_g3_is_on_by_default():
    assert w8.TAIL16_G3


def test_kill_switch():
    code = f"from {w8.__name__} import TAIL16_G3\nprint(TAIL16_G3)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_TAIL16_G3": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-1] == "False"


def _stock_spy(monkeypatch):
    seen = []
    monkeypatch.setattr(
        SpecWorkerBase, "_store_block_draft_probs", staticmethod(lambda *a: seen.append(a))
    )
    return seen


def test_missing_op_takes_the_stock_store(monkeypatch):
    monkeypatch.setattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None, raising=False)
    seen = _stock_spy(monkeypatch)
    w8.S4WorkerMixin._store_block_draft_probs(None, None, None)
    assert seen == [(None, None, None)]


def test_cpu_and_unsupported_dtype_take_the_stock_store(monkeypatch):
    monkeypatch.setattr(
        torch.ops.trtllm, "minimax_m31_scatter_rows", lambda *a: None, raising=False
    )
    seen = _stock_spy(monkeypatch)
    w8.S4WorkerMixin._store_block_draft_probs(
        torch.zeros(4, 3, 16), torch.tensor([3, 1]), torch.randn(2, 3, 16)
    )
    w8.S4WorkerMixin._store_block_draft_probs(torch.zeros(8, 3, 16, dtype=torch.float64), torch.tensor([1, 2]),
                                              torch.zeros(2, 3, 16, dtype=torch.float64))  # fmt: skip
    assert len(seen) == 2


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or getattr(torch.ops.trtllm, "minimax_m31_scatter_rows", None) is None,
    reason="the TAIL16_G3 op runs on SM107 with the op built",
)


@_GPU
@pytest.mark.parametrize("n", [1, 8])
@pytest.mark.parametrize("k", [3, 7])
@pytest.mark.parametrize("pad", [0, 8])
def test_bitwise_the_stock_store(n, k, pad):
    v = 200064
    dst0 = torch.randn(16, 7, v + pad, device="cuda")
    probs = torch.rand(n, k, v, device="cuda")
    probs[0, 0, 5] = float("nan")
    probs[-1, -1, :100] = float("-inf")
    slots = torch.randperm(16, device="cuda")[:n]
    ref, got = dst0.clone(), dst0.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, probs)
    w8.S4WorkerMixin._store_block_draft_probs(got, slots, probs)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


@_GPU
def test_above_the_request_cap_takes_the_stock_store(monkeypatch):
    n = w8.TAIL16_G3_MAX_REQUESTS + 1
    dst = torch.zeros(n + 2, 7, 64, device="cuda")
    probs = torch.rand(n, 7, 64, device="cuda")
    slots = torch.arange(n, device="cuda")
    seen = _stock_spy(monkeypatch)
    w8.S4WorkerMixin._store_block_draft_probs(dst, slots, probs)
    assert len(seen) == 1


@_GPU
def test_graph_replay():
    v = 200064
    dst = torch.zeros(9, 7, v, device="cuda")
    probs = torch.rand(1, 7, v, device="cuda")
    slots = torch.tensor([2], dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        w8.S4WorkerMixin._store_block_draft_probs(dst, slots, probs)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (17, 83):
        torch.manual_seed(seed)
        probs.uniform_()
        slots.fill_(seed % 9)
        dst.zero_()
        ref = dst.clone()
        SpecWorkerBase._store_block_draft_probs(ref, slots, probs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int32), dst.view(torch.int32))
