# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned BF16 split-K for the TP8 K7 drafter's deferred down projections.

The returned ordinary tensor goes to the existing S9b consumer. Unsupported
calls use the original complete Linear.forward, including its LoRA dispatch.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
from pathlib import Path

FI_MODULE = "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk"
FI_SOURCE_SHA256 = "a9c050cc7fa56f508da2bd969653f323a68421a49b38f8a8ac620ff371f55090"
TACTIC = (64, 8, 2, 12)
ROWS = range(1, 9)


def _module_admitted(linear, weight, bf16, row_mode, method_type, allreduce_type) -> bool:
    """Metadata-only admission; no device reads or synchronization."""
    mapping = linear.mapping
    method = linear.quant_method
    ar = linear.all_reduce
    return (
        linear.weight is weight
        and weight.is_cuda
        and weight.dtype == bf16
        and tuple(weight.shape) == (6144, 1536)
        and weight.is_contiguous()
        and not weight.requires_grad
        and linear.bias is None
        and linear.tp_mode == row_mode
        and linear.reduce_output
        and not linear.partition_plan.enabled
        and not linear.use_custom_cublas_mm
        and not linear.use_cute_dsl_bf16_gemm
        and mapping.tp_size == 8
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and not mapping.enable_attention_dp
        and list(mapping.tp_group) == list(range(8))
        and type(method) is method_type
        and getattr(method.apply, "__func__", None) is method_type.apply
        and method.supports_nccl_symmetric_memory_window_output
        and type(ar) is allreduce_type
        and getattr(ar.forward, "__func__", None) is allreduce_type.forward
        and not ar._disable_mpi
        and ar.uses_nccl_symmetric_memory_window()
    )


def _wrap_forward(linear, provider, tactic, torch, low_m, types, layer, log_taken):
    original = linear.forward
    weight = linear.weight
    bf16, row_mode, method_type, allreduce_type, fusion_none = types
    logged = False

    def forward(input, *, all_reduce_params=None, lora_params=None, layer_idx=None):
        nonlocal logged
        if (
            not low_m.LOW_M_GEMM_ACTIVE
            and not torch.is_grad_enabled()
            and isinstance(input, torch.Tensor)
            and input.dtype == bf16
            and input.device == weight.device
            and not input.requires_grad
            and input.ndim == 2
            and input.shape[0] in ROWS
            and input.shape[1] == 1536
            and input.is_contiguous()
            and lora_params is None
            and all_reduce_params is not None
            and all_reduce_params.enable_allreduce is False
            and all_reduce_params.fusion_op == fusion_none
            and _module_admitted(linear, weight, *types[:4])
        ):
            output = torch.empty((input.shape[0], 6144), dtype=bf16, device=input.device)
            # The original AllReduce.forward is an identity for this explicit
            # no-allreduce call. The existing deferred consumer still runs.
            provider.run_splitk_dense(input, weight.t(), None, output, True, tactic)
            if not logged:
                log_taken(layer, input.shape[0])
                logged = True
            return output
        return original(
            input,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            layer_idx=layer_idx,
        )

    return forward


def _load_provider():
    """Decline unqualified providers before importing or launching a kernel."""
    try:
        if importlib.metadata.version("flashinfer-python") != "0.6.18":
            return None
        module = importlib.import_module(FI_MODULE)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != FI_SOURCE_SHA256:
            return None
    except (ImportError, OSError):
        return None
    return module


def _prepare_aot(provider) -> None:
    """Use the existing object export/load path for the pinned FI compiler."""
    from tensorrt_llm._torch import cute_aot

    if getattr(provider, "_m31_down_splitk_aot", False):
        return
    sources = (str(Path(provider.__file__).resolve()), str(Path(__file__).resolve()))
    provider._get_compiled_splitk_kernel = cute_aot.cached_compile(
        "m31_mxdraft_down_splitk.FI_splitk", lambda: sources
    )(provider._get_compiled_splitk_kernel)
    provider._m31_down_splitk_aot = True


def route_down_splitk(draft_model, *, enabled: bool = True) -> int:
    """Prewarm then install the five scoped forwards, outside CUDA capture.

    The caller latches the target kill switch at import/prebuild. A killed
    process never imports or prewarms FI through this route. Kernel/compile
    errors propagate; they are never retried through stock after a launch.
    """
    if not enabled:
        return 0
    import torch

    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modules import low_m_gemm
    from tensorrt_llm._torch.modules.linear import (
        Linear,
        TensorParallelMode,
        UnquantizedLinearMethod,
    )
    from tensorrt_llm.functional import AllReduceFusionOp
    from tensorrt_llm.logger import logger

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MXDRAFT_DOWN_SPLITK must be prepared outside capture")
    layers = list(draft_model.model.layers)
    if len(layers) != 5 or low_m_gemm.LOW_M_GEMM_ACTIVE:
        return 0
    linears = [layer.mlp.down_proj for layer in layers]
    if all(getattr(linear, "_m31_down_splitk", None) is not None for linear in linears):
        return len(linears)
    types = (
        torch.bfloat16,
        TensorParallelMode.ROW,
        UnquantizedLinearMethod,
        AllReduce,
        AllReduceFusionOp.NONE,
    )
    if any(
        type(linear) is not Linear
        or getattr(linear.forward, "__func__", None) is not Linear.forward
        or not _module_admitted(linear, linear.weight, *types[:4])
        or torch.cuda.get_device_capability(linear.weight.device) != (10, 7)
        for linear in linears
    ):
        return 0
    provider = _load_provider()
    if provider is None:
        logger.info("M3.1 MXDRAFT_DOWN_SPLITK: provider not qualified; stock forwards kept")
        return 0
    _prepare_aot(provider)
    tactic = provider.SplitKTactic(*TACTIC)
    # Prewarm every admitted layout before installing any forward. These
    # ordinary temporary outputs never enter an all-reduce or serving state.
    with torch.inference_mode():
        for linear in linears:
            for rows in ROWS:
                x = torch.zeros((rows, 1536), dtype=torch.bfloat16, device=linear.weight.device)
                out = torch.empty((rows, 6144), dtype=torch.bfloat16, device=x.device)
                provider.run_splitk_dense(x, linear.weight.t(), None, out, True, tactic)
        torch.cuda.synchronize(linears[0].weight.device)

    def log_taken(layer, rows):
        logger.info(
            f"M3.1 MXDRAFT_DOWN_SPLITK taken layer={layer} rows={rows} "
            "provider=FI_a9c050cc tactic=64,8,2,12 output=ordinary"
        )

    for layer, linear in enumerate(linears):
        original = linear.forward
        linear.forward = _wrap_forward(
            linear, provider, tactic, torch, low_m_gemm, types, layer, log_taken
        )
        linear._m31_down_splitk = (original, linear.weight)
    return len(linears)
