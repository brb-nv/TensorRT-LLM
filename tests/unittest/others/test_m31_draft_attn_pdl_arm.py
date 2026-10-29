# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MiniMax-M3.1 sm_107 TP8 DRAFT_ATTN_PDL arm.

The worker's prebuild arms the drafter attention PDL flag by default, TRTLLM_M31_DISABLE_DRAFT_ATTN_PDL turns it off,
and the flag reaches both trtllm-gen launches. Sources are parsed and the switch / setter bodies executed without
importing torch; kernel behavior is covered by the GPU microbenchmark.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
TARGET = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    / "minimax_m31_nvfp4__sm_107__tp8"
)
SETTER_SOURCE = ROOT / "tensorrt_llm/_torch/speculative/dflash.py"
DSPARK_SOURCE = ROOT / "tensorrt_llm/_torch/speculative/dspark.py"
MODEL_SOURCE = ROOT / "tensorrt_llm/_torch/models/modeling_dflash.py"


def parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def find_def(tree, cls_name, fn_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == fn_name:
                    return item
    raise AssertionError(f"{cls_name}.{fn_name} not found")


def arm_calls(prebuild):
    return [
        node
        for node in ast.walk(prebuild)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "enable_attention_pdl"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]


def load_switches(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_DRAFT_ATTN_PDL", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_M31_DISABLE_DRAFT_ATTN_PDL", value)
    spec = importlib.util.spec_from_file_location(
        "m31_switches_under_test", TARGET / "m31_switches.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prebuild():
    return find_def(parse(TARGET / "dspark.py"), "M3DSparkWorker", "prebuild")


@pytest.fixture(scope="module")
def setter():
    fn = find_def(parse(SETTER_SOURCE), "DFlashWorker", "enable_attention_pdl")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SETTER_SOURCE), "exec"), namespace)
    return namespace["enable_attention_pdl"]


def test_prebuild_arms_drafter_attention_pdl_once(prebuild):
    calls = arm_calls(prebuild)
    assert len(calls) == 1
    (call,) = calls
    assert not call.keywords and len(call.args) == 2
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "draft_model"
    flag = call.args[1]
    assert isinstance(flag, ast.UnaryOp) and isinstance(flag.op, ast.Not)
    switch = flag.operand
    assert isinstance(switch, ast.Call) and isinstance(switch.func, ast.Name)
    assert switch.func.id == "m31_disabled"
    assert [a.value for a in switch.args if isinstance(a, ast.Constant)] == ["DRAFT_ATTN_PDL"]


def test_prebuild_logs_the_armed_state(prebuild):
    logged = [
        node
        for node in ast.walk(prebuild)
        if isinstance(node, ast.JoinedStr)
        and any(
            isinstance(v, ast.Constant) and "drafter attention PDL" in str(v.value)
            for v in node.values
        )
    ]
    assert len(logged) == 1
    names = {n.id for n in ast.walk(logged[0]) if isinstance(n, ast.Name)}
    assert {"attn_pdl", "rank"} <= names


@pytest.mark.parametrize(
    "value, armed",
    [
        (None, True),
        ("", True),
        ("0", True),
        ("off", True),
        ("1", False),
        ("true", False),
        ("ON", False),
    ],
)
def test_default_on_and_existing_switch_kills(prebuild, monkeypatch, value, armed):
    switches = load_switches(monkeypatch, value)
    flag = arm_calls(prebuild)[0].args[1]
    expr = compile(ast.Expression(body=flag), "<prebuild flag>", "eval")
    assert eval(expr, {"m31_disabled": switches.m31_disabled}) is armed


def test_bad_switch_value_fails_loudly(monkeypatch):
    with pytest.raises(ValueError, match="TRTLLM_M31_DISABLE_DRAFT_ATTN_PDL"):
        load_switches(monkeypatch, "maybe")


@pytest.mark.parametrize("enabled", [True, False])
def test_setter_sets_and_returns_the_model_flag(setter, enabled):
    draft_model = SimpleNamespace(_dflash_attn_pdl=not enabled)
    assert setter(None, draft_model, enabled) is enabled
    assert draft_model._dflash_attn_pdl is enabled


def test_setter_declines_a_model_without_the_flag(setter):
    draft_model = SimpleNamespace()
    assert setter(None, draft_model, True) is False
    assert not hasattr(draft_model, "_dflash_attn_pdl")


def test_target_worker_inherits_the_setter():
    target = parse(TARGET / "dspark.py")
    worker = next(
        n for n in target.body if isinstance(n, ast.ClassDef) and n.name == "M3DSparkWorker"
    )
    assert [b.id for b in worker.bases if isinstance(b, ast.Name)] == [
        "S4WorkerMixin",
        "DSparkWorker",
    ]
    overriders = [
        n.name
        for n in ast.walk(target)
        if isinstance(n, ast.FunctionDef) and n.name == "enable_attention_pdl"
    ]
    assert not overriders
    assert "enable_attention_pdl" not in (TARGET / "dspark_s4.py").read_text()
    dspark = next(
        n
        for n in parse(DSPARK_SOURCE).body
        if isinstance(n, ast.ClassDef) and n.name == "DSparkWorker"
    )
    assert [b.id for b in dspark.bases if isinstance(b, ast.Name)] == ["DFlashWorker"]


def test_flag_reaches_both_trtllm_gen_launches():
    tree = parse(MODEL_SOURCE)
    uses = [
        kw
        for kw in (k for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords)
        if kw.arg == "enable_pdl"
        and isinstance(kw.value, ast.Attribute)
        and kw.value.attr == "_dflash_attn_pdl"
    ]
    assert len(uses) == 2
    inits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_dflash_attn_pdl" for t in n.targets)
    ]
    assert (
        len(inits) == 1
        and isinstance(inits[0].value, ast.Constant)
        and inits[0].value.value is False
    )
