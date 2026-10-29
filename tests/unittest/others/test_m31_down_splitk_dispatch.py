# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission/control tests; native arithmetic is covered by the GPU oracle."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

TARGET = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    / "minimax_m31_nvfp4__sm_107__tp8"
)


def load_source(name):
    spec = importlib.util.spec_from_file_location(name, TARGET / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorMetadata:
    """No CUDA storage: exercises metadata predicates, never kernel correctness."""

    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self.is_cuda = device.startswith("cuda:")
        self.requires_grad = False
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def t(self):
        return self


class MethodMetadata:
    supports_nccl_symmetric_memory_window_output = True

    def apply(self, *args):
        raise AssertionError("a CPU test must not run a stock GPU operator")


class AllReduceMetadata:
    _disable_mpi = False

    def uses_nccl_symmetric_memory_window(self):
        return True

    def forward(self, *args):
        raise AssertionError("the scoped producer must not run a collective")


class LinearMetadata:
    def __init__(self):
        self.weight = TensorMetadata((6144, 1536))
        self.bias = None
        self.tp_mode = "row"
        self.reduce_output = True
        self.partition_plan = SimpleNamespace(enabled=False)
        self.use_custom_cublas_mm = self.use_cute_dsl_bf16_gemm = False
        self.mapping = SimpleNamespace(
            tp_size=8, pp_size=1, cp_size=1, enable_attention_dp=False, tp_group=list(range(8))
        )
        self.quant_method, self.all_reduce = MethodMetadata(), AllReduceMetadata()
        self.stock_calls, self.stock_result = [], object()

    def forward(self, input, **kwargs):
        self.stock_calls.append((input, kwargs))
        return self.stock_result


@pytest.fixture
def dispatch():
    return load_source("mxdraft_down_splitk")


@pytest.fixture
def armed(dispatch):
    linear = LinearMetadata()
    launched, taken = [], []
    torch = SimpleNamespace(
        Tensor=TensorMetadata,
        is_grad_enabled=lambda: False,
        empty=lambda shape, **kwargs: TensorMetadata(shape, **kwargs),
    )
    provider = SimpleNamespace(run_splitk_dense=lambda *args: launched.append(args))
    types = ("bf16", "row", MethodMetadata, AllReduceMetadata, "none")
    forward = dispatch._wrap_forward(
        linear,
        provider,
        (64, 8, 2, 12),
        torch,
        SimpleNamespace(LOW_M_GEMM_ACTIVE=False),
        types,
        3,
        lambda *args: taken.append(args),
    )
    return SimpleNamespace(
        linear=linear,
        forward=forward,
        launched=launched,
        taken=taken,
        torch=torch,
        provider=provider,
    )


@pytest.mark.parametrize("rows", [1, 7, 8])
def test_supported_rows_write_direct_output(armed, rows):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    x = TensorMetadata((rows, 1536))
    out = armed.forward(x, all_reduce_params=params)
    assert out.shape == (rows, 6144)
    assert armed.launched == [(x, armed.linear.weight, None, out, True, (64, 8, 2, 12))]
    assert not armed.linear.stock_calls
    armed.forward(x, all_reduce_params=params)
    assert armed.taken == [(3, rows)]


@pytest.mark.parametrize(
    "input",
    [
        TensorMetadata((0, 1536)),
        TensorMetadata((9, 1536)),
        TensorMetadata((7, 1024)),
        TensorMetadata((1, 7, 1536)),
        TensorMetadata((7, 1536), dtype="fp16"),
        TensorMetadata((7, 1536), device="cpu"),
        TensorMetadata((7, 1536), contiguous=False),
    ],
)
def test_unsupported_inputs_preserve_complete_stock_call(armed, input):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    out = armed.forward(input, all_reduce_params=params, layer_idx=17)
    assert out is armed.linear.stock_result and not armed.launched
    assert armed.linear.stock_calls == [
        (input, dict(all_reduce_params=params, lora_params=None, layer_idx=17))
    ]


@pytest.mark.parametrize("adapter", [{}, {"active": True}])
def test_adapter_arguments_always_use_stock(armed, adapter):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    out = armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params, lora_params=adapter)
    assert out is armed.linear.stock_result and not armed.launched
    assert armed.linear.stock_calls[0][1]["lora_params"] is adapter


@pytest.mark.parametrize(
    "field,value",
    [("tp_size", 4), ("pp_size", 2), ("cp_size", 2), ("enable_attention_dp", True)],
)
def test_parallelism_outside_scope_falls_back(armed, field, value):
    setattr(armed.linear.mapping, field, value)
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


def test_weight_replacement_and_grad_mode_fall_back(armed):
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    original = armed.linear.weight
    armed.linear.weight = TensorMetadata((6144, 1536))
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    armed.linear.weight = original
    armed.torch.is_grad_enabled = lambda: True
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


@pytest.mark.parametrize(
    "params",
    [
        None,
        SimpleNamespace(enable_allreduce=True, fusion_op="none"),
        SimpleNamespace(enable_allreduce=False, fusion_op="other"),
    ],
)
def test_non_deferred_call_uses_stock(armed, params):
    assert (
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
        is armed.linear.stock_result
    )
    assert not armed.launched


def test_launch_error_never_retries_stock(armed):
    def fail(*args):
        raise RuntimeError("kernel launch failed")

    armed.provider.run_splitk_dense = fail
    params = SimpleNamespace(enable_allreduce=False, fusion_op="none")
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        armed.forward(TensorMetadata((7, 1536)), all_reduce_params=params)
    assert not armed.linear.stock_calls


def test_kill_latched_by_existing_switch_skips_all_imports(dispatch, monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MXDRAFT_DOWN_SPLITK", "1")
    switches = load_source("m31_switches")
    assert switches.m31_disabled("MXDRAFT_DOWN_SPLITK")
    # The local test environment has no Torch. An attempted import would fail.
    assert (
        dispatch.route_down_splitk(None, enabled=not switches.m31_disabled("MXDRAFT_DOWN_SPLITK"))
        == 0
    )


def test_unqualified_provider_is_declined(dispatch, monkeypatch):
    monkeypatch.setattr(dispatch.importlib.metadata, "version", lambda name: "0.6.19")
    assert dispatch._load_provider() is None
