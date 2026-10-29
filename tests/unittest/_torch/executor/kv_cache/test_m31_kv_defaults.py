# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 KV defaults (m31_kv_defaults.arm_minimax_m31): armed on Blackwell-family GPUs only, from the sm_107 and
sm_103 routing and from the built-in MiniMaxM3ForCausalLM with DSpark, and without touching the CuTe AOT keys. CPU
only."""

import ast
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
import tensorrt_llm._torch.models as models
from tensorrt_llm._torch import cute_aot
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_layerwise_onboard, m31_kv_defaults


@pytest.fixture
def unarmed(monkeypatch):
    monkeypatch.setattr(kv_layerwise_onboard, "_M31_DEFAULT_OFF", False)
    monkeypatch.delenv("TRTLLM_M31_DISABLE_KVLO_OFF", raising=False)


@pytest.mark.parametrize(
    "capability, armed",
    [
        ((10, 7), True),  # VR
        ((10, 3), True),  # GB300
        ((10, 0), True),  # B200
        ((9, 0), False),  # not Blackwell: nothing armed
        ((12, 0), False),
    ],
)
def test_armed_on_blackwell_only(unarmed, capability, armed):
    m31_kv_defaults.arm_minimax_m31(capability)
    assert kv_layerwise_onboard._M31_DEFAULT_OFF is armed


def test_arming_leaves_the_cute_aot_keys_unchanged(unarmed):
    """The shipped CuTe DSL AOT objects are keyed on the process's TRTLLM_M31_DISABLE_* / _ENABLE_* settings: an arm
    that set one would make every shipped object miss (each server start compiling them in-process)."""
    before = cute_aot.env_settings()
    m31_kv_defaults.arm_minimax_m31((10, 7))
    assert cute_aot.env_settings() == before


def _calls(fn: ast.AST) -> list:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def _called(call: ast.Call) -> str:
    return getattr(call.func, "attr", None) or getattr(call.func, "id", None)


def test_built_in_minimax_m3_arms_with_dspark():
    """The built-in MiniMaxM3ForCausalLM (the path GB300 serves when no modeling_v2 target routes) arms the defaults in
    its constructor, after super().__init__ built its spec worker and before the KV cache managers are built."""
    tree = ast.parse((Path(models.__file__).parent / "modeling_minimaxm3.py").read_text())
    (cls,) = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3ForCausalLM"
    ]
    (init,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    lines = {_called(c): c.lineno for c in _calls(init)}
    assert "arm_minimax_m31" in lines
    assert lines["__init__"] < lines["arm_minimax_m31"]


@pytest.mark.parametrize(
    "arch, arm", [("Sm103", "arm_minimax_m31"), ("Sm107", "arm_sm107_shared_opt_ins")]
)
def test_every_target_is_armed_before_it_is_returned(arch, arm):
    """route() arms the defaults (directly for sm_103, through arm_sm107_shared_opt_ins for sm_107) before it returns
    any M3.1 target class, so before the model and its KV cache managers are built."""
    path = Path(_family.__file__).resolve().parent / "routing.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "route"]
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    returns = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and arch in n.value.id
    ]
    assert returns
    for ret in returns:
        node, armed = ret, False
        while node in parents and not armed:
            node = parents[node]
            armed = any(_called(c) == arm and c.lineno < ret.lineno for c in _calls(node))
        assert armed, f"{ret.value.id} is returned before {arm}()"
