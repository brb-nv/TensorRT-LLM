# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The fmha_sm100 MiniMax-M3 performance paths engage on SM107 only (3rdparty/patches/msa_strided_paged_kv.patch).

SM107 runs the sm_100 builds, so every gate is a runtime check of the device capability, resolved once on the host:
the plan side stream and the deferred split-decision read (api.py), the indexer R2 store
(cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py + the AOT fingerprint in cute/src/common/aot_cache.py), the proxy's skipped
correction chain / S double buffer (jit.py defines) and the k2q CSR build's scratch arena (cute/src/sm100/build_k2q_csr,
host C++). SM100 / SM103 keep the stock fmha_sm100. The jit.py / aot_cache.py cells load the installed files with a
stand-in torch, and the k2q cell reads the installed source, so they run without a GPU.
"""

import importlib.util
import inspect
import os
import sys
import types

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
)


@pytest.fixture
def api():
    # Installs the CuTe aliases the packaged MSA sources still reference, then reports importability.
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) is not installed")
    return pytest.importorskip("fmha_sm100.api")


def _fmha_sm100_dir() -> str:
    spec = importlib.util.find_spec("fmha_sm100")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("fmha_sm100 (MSA) is not installed")
    return list(spec.submodule_search_locations)[0]


class _Bound:
    def __init__(self, **arguments):
        self.arguments = arguments


def _side_stream_eligible_args():
    lens = torch.tensor([16, 32], dtype=torch.int32)  # host tensors: the side stream may read them
    return _Bound(
        device=0, qo_segment_lens=lens, kv_segment_lens=lens, qo_offset=None, kv_block_num=-1
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cc, ccs, expected",
    [
        ((10, 0), {(10, 7)}, False),
        ((10, 3), {(10, 7)}, False),
        ((10, 7), {(10, 7)}, True),
        ((12, 0), {(10, 7)}, False),
        ((10, 3), {(10, 3), (10, 7)}, True),
        ((10, 0), {(10, 0), (10, 3), (10, 7)}, True),
        ((12, 0), {(10, 0), (10, 3), (10, 7)}, False),
    ],
)
def test_plan_side_stream_engages_on_listed_archs(api, monkeypatch, cc, ccs, expected):
    # SM107, plus the SM100 / SM103 of TLLM_FMHA_PLAN_SS_ARCHS (by default those of the segmented indexer proxy).
    monkeypatch.setattr(api, "_SS_CC", {0: cc})
    monkeypatch.setattr(api, "_SS_CCS", frozenset(ccs))
    # The eligibility check also asks whether a CUDA graph is being captured; no GPU is needed for that here.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert api._ss_eligible(_side_stream_eligible_args()) is expected


@pytest.mark.cpu_only
def test_plan_side_stream_resolves_the_capability_once(api, monkeypatch):
    calls = []

    def fake_capability(idx):
        calls.append(idx)
        return (10, 7)

    monkeypatch.setattr(api, "_SS_CC", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_capability)
    for _ in range(5):
        assert api._ss_is_sm107(0)
    assert calls == [0]


@pytest.mark.cpu_only
def test_plan_side_stream_kill_env_name_unchanged(api):
    assert '"TLLM_FMHA_PLAN_SIDE_STREAM", "1"' in inspect.getsource(api)


def _load_with_capability(monkeypatch, root: str, cap: tuple, calls: list):
    """jit.py and aot_cache.py from the installed package, with a stand-in torch reporting ``cap``."""

    def capability(*_):
        calls.append(cap)
        return cap

    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=capability)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "jinja2", types.ModuleType("jinja2"))
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = types.ModuleType("cutlass.cute")
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cutlass.cute)
    modules = []
    for rel, name in (("jit.py", "_gate_jit"), ("cute/src/common/aot_cache.py", "_gate_aot_cache")):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


_BLACKWELL_ARCH_ENVS = (
    "TRTLLM_M3_BLACKWELL_RUBIN_OPTS",
    "TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS",
    "TLLM_FMHA_PROXY_MODE_ARCHS",
    "TRTLLM_MSA_R2STORE_ARCHS",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cap", [(10, 7), (10, 0), (10, 3)], ids=["sm107", "sm100", "sm103"])
@pytest.mark.parametrize("chain, dbuf, r2kill", [("0", "1", "0"), ("1", "1", "1"), ("0", "0", "0")])
@pytest.mark.parametrize(
    "arch_env, proxy_mode_sms",
    [
        ({}, (100, 103)),
        ({"TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "107"}, ()),
        ({"TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "100,103"}, (100, 103)),
        ({"TLLM_FMHA_PROXY_MODE_ARCHS": "103"}, (103,)),
        (
            {
                "TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "100,103",
                "TRTLLM_M3_BLACKWELL_RUBIN_OPTS": "0",
            },
            (),
        ),
    ],
    ids=["default", "proxy_rubin_only", "with_segmented_proxy", "explicit_sm103", "umbrella_off"],
)
def test_proxy_modes_and_r2_store_engage_on_listed_archs(
    monkeypatch, cap, chain, dbuf, r2kill, arch_env, proxy_mode_sms
):
    root = _fmha_sm100_dir()
    monkeypatch.setenv("TLLM_FMHA_PROXY_CORRECTION_CHAIN", chain)
    monkeypatch.setenv("TLLM_FMHA_PROXY_S_DOUBLE_BUFFER", dbuf)
    monkeypatch.setenv("TRTLLM_MSA_DISABLE_R2STORE", r2kill)
    for name in _BLACKWELL_ARCH_ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in arch_env.items():
        monkeypatch.setenv(name, value)
    calls = []
    jit, aot = _load_with_capability(monkeypatch, root, cap, calls)
    # Importing either module queries no device (no CUDA initialization before a worker fork).
    assert calls == []
    sm107 = cap == (10, 7)
    # The proxy modes: SM107, and the SM100 / SM103 that follow the segmented proxy or are listed themselves.
    modes_on = sm107 or cap[0] * 10 + cap[1] in proxy_mode_sms
    want_chain = chain == "1" or not modes_on
    want_single = dbuf == "0" or not modes_on
    jit._get_tvm_ffi_include = lambda: "/stub/tvm_ffi/include"  # flags only; nothing is compiled
    flags = jit._get_nvcc_flags(jit.CACHE_BASE)
    assert tuple(jit._proxy_modes()) == (want_chain, want_single)
    assert ("-DFMHA_ONLYSCORE_PER_TILE_CORRECTION" in flags) is want_chain
    assert ("-DFMHA_ONLYSCORE_SINGLE_S" in flags) is want_single
    # The R2 store follows its own arch set (default 100,103), which the umbrella switch also clears.
    r2_on = sm107 or arch_env.get("TRTLLM_M3_BLACKWELL_RUBIN_OPTS", "1") != "0"
    assert aot.r2_store_enabled() is (r2_on and r2kill != "1")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cap", [(10, 7), (10, 0), (10, 3), (12, 0)], ids=["sm107", "sm100", "sm103", "sm120"]
)
@pytest.mark.parametrize(
    "arch_env, r2_sms",
    [
        ({}, (100, 103)),
        ({"TRTLLM_MSA_R2STORE_ARCHS": "107"}, ()),
        ({"TRTLLM_MSA_R2STORE_ARCHS": "100,103"}, (100, 103)),
        ({"TRTLLM_MSA_R2STORE_ARCHS": "103,120"}, (103,)),
        ({"TRTLLM_MSA_R2STORE_ARCHS": "100,103", "TRTLLM_M3_BLACKWELL_RUBIN_OPTS": "0"}, ()),
    ],
    ids=["default", "rubin_only", "blackwell", "sm103_only", "umbrella_off"],
)
def test_r2_store_on_listed_archs(monkeypatch, cap, arch_env, r2_sms):
    # The R2 store warpgroup: SM107, plus the SM100 / SM103 of TRTLLM_MSA_R2STORE_ARCHS (100,103 by default).
    root = _fmha_sm100_dir()
    monkeypatch.delenv("TRTLLM_MSA_DISABLE_R2STORE", raising=False)
    for name in ("TRTLLM_MSA_R2STORE_ARCHS", "TRTLLM_M3_BLACKWELL_RUBIN_OPTS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in arch_env.items():
        monkeypatch.setenv(name, value)
    _, aot = _load_with_capability(monkeypatch, root, cap, [])
    assert aot.r2_store_enabled() is (cap == (10, 7) or cap[0] * 10 + cap[1] in r2_sms)


def _k2q_csr_source() -> str:
    path = os.path.join(_fmha_sm100_dir(), "cute/src/sm100/build_k2q_csr/build_k2q_csr.cu")
    if not os.path.isfile(path):
        pytest.skip("build_k2q_csr.cu is not installed")
    with open(path) as f:
        return f.read()


@pytest.mark.cpu_only
def test_k2q_scratch_arena_engages_on_listed_archs():
    # The k2q CSR build is host C++ compiled at first use, so its gate is checked in the installed source: the arena
    # (and the other k2q changes) take the runtime arch check (SM107, plus the SM100 / SM103 of FMHA_K2Q_OPT_ARCHS,
    # read once with std::getenv), and the other devices keep the stock allocations.
    src = _k2q_csr_source()
    gate = src.index("k2q_opts_state = k2q_opt_arch(cc_major * 10 + cc_minor) ? 1 : 2;")
    arena = src.index("if (k2q_opts && k2q_arena_enabled()) {")
    assert gate < arena
    assert 'std::getenv("FMHA_K2Q_OPT_ARCHS")' in src
    assert 'std::getenv("TRTLLM_M3_BLACKWELL_RUBIN_OPTS")' in src
    assert 'char const* list = env != nullptr ? env : "103,107";' in src
    assert (
        "return sm == 107 || (sm == 100 && (blackwell_mask & 1)) || (sm == 103 && (blackwell_mask & 2));"
        in src
    )
    assert "is_sm107" not in src
    assert "row_counts = torch::zeros({H, total_rows}, opts);" in src
    assert 'std::getenv("FMHA_K2Q_WS_ARENA")' in src


@pytest.mark.cpu_only
def test_deferred_split_read_engages_on_listed_archs(api):
    # The adaptive plan defers its split-decision read only on the capabilities of _PLAN_DEFER_READ_CCS (SM107, plus
    # the SM100 / SM103 of TLLM_FMHA_PLAN_DEFER_READ_ARCHS); other devices read plan_cost immediately.
    src = inspect.getsource(api)
    assert "defer = (_PLAN_DEFER_READ and _ss_cc(device) in _PLAN_DEFER_READ_CCS" in src
    assert '"TLLM_FMHA_PLAN_DEFER_READ", "1"' in src
    assert (10, 7) in api._PLAN_DEFER_READ_CCS


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, {(10, 0), (10, 3), (10, 7)}),
        ({"TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "107"}, {(10, 7)}),
        ({"TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "100,103"}, {(10, 0), (10, 3), (10, 7)}),
        ({"TLLM_FMHA_PLAN_SS_ARCHS": "103"}, {(10, 3), (10, 7)}),
        (
            {"TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS": "103", "TLLM_FMHA_PLAN_SS_ARCHS": "107"},
            {(10, 7)},
        ),
        (
            {"TLLM_FMHA_PLAN_SS_ARCHS": "100,103,120", "TRTLLM_M3_BLACKWELL_RUBIN_OPTS": "0"},
            {(10, 7)},
        ),
    ],
)
def test_plan_side_stream_arch_set_from_env(api, monkeypatch, env, expected):
    for name in (
        "TRTLLM_MSA_PROXY_AUTO_KV_SPLITS_ARCHS",
        "TLLM_FMHA_PLAN_SS_ARCHS",
        "TRTLLM_M3_BLACKWELL_RUBIN_OPTS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    got = api._m3_gate_ccs("TLLM_FMHA_PLAN_SS_ARCHS", api._m3_proxy_auto_kv_splits_sms())
    assert got == expected
