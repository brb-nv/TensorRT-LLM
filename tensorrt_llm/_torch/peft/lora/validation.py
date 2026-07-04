# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers for routed-expert (MoE) LoRA.

MoE LoRA is supported only on the Cutlass backend with unquantized fp16/bf16 or
per-tensor FP8 (qdq) base weights. This module provides a single helper,
`check_moe_lora_supported`, that callers (typically the MoE factory in
`create_moe.py`) can invoke at construction time so that unsupported
combinations fail loudly instead of silently dropping the LoRA contribution at
runtime.

Runtime-only rejections (min-latency mode, alltoall, CUDA-graph without slot
pointers) are enforced in the C++ thop / runtime call paths and are NOT
re-checked here.
"""

from typing import Iterable, Optional, Set

from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.quantization.mode import QuantMode

# Canonical routed-expert MoE LoRA module names (single source of truth).
from .layer import MOE_LORA_MODULE_NAMES

# Base-weight quantization bits that MoE LoRA does not support. Only per-tensor
# FP8 (qdq) composes with MoE LoRA; any of the bits below makes the combination
# unsupported.
_UNSUPPORTED_QUANT = (
    QuantMode.INT4_WEIGHTS
    | QuantMode.INT8_WEIGHTS
    | QuantMode.ACTIVATIONS
    | QuantMode.FP8_ROWWISE
    | QuantMode.FP8_1x128_128x128
    | QuantMode.W4A8_QSERVE
    | QuantMode.NVFP4
    | QuantMode.W4A8_NVFP4_FP8
    | QuantMode.W4A8_MXFP4_FP8
    | QuantMode.W4A16_MXFP4
    | QuantMode.W4A8_MXFP4_MXFP8
    | QuantMode.MXFP8
)

_MOE_LORA_MODULE_NAME_SET: Set[str] = set(MOE_LORA_MODULE_NAMES)


def _normalize_targets(lora_target_modules: Iterable[str]) -> Set[str]:
    return {name.lower() for name in lora_target_modules or []}


def has_moe_lora_targets(lora_config: Optional[LoraConfig]) -> bool:
    """Return True iff `lora_config` requests LoRA on any routed-expert MoE module."""
    if lora_config is None:
        return False
    return bool(
        _MOE_LORA_MODULE_NAME_SET
        & _normalize_targets(getattr(lora_config, "lora_target_modules", []) or [])
    )


def _is_supported_quant(quant_mode) -> bool:
    """Return True iff the only base-weight quantization is per-tensor FP8 (qdq).

    The CUTLASS MoE LoRA kernel runs the LoRA GEMM on the bf16/fp16 activations,
    dequantizing the per-tensor FP8 (qdq) activations to the backbone type before
    the LoRA GEMM. FP8 block-scale, NVFP4, and the integer / MXFP4 / W4A8 formats
    in `_UNSUPPORTED_QUANT` have no such path and stay rejected.
    """
    if quant_mode is None:
        return False
    # quant_mode may be a QuantMode, or a QuantModeWrapper that holds a per-layer
    # list of QuantModes and forwards has_* queries. Normalize to the underlying
    # QuantMode(s) so the bitwise check works in either case.
    objs = getattr(quant_mode, "objs", None)
    modes = objs if objs is not None else [quant_mode]
    has_supported = False
    for mode in modes:
        if mode.has_fp8_qdq():
            has_supported = True
        if bool(mode & _UNSUPPORTED_QUANT):
            return False
    return has_supported


def _is_quantized(quant_mode) -> bool:
    """Return True iff the layer has any active (non-kv-cache) quantization."""
    if quant_mode is None or not hasattr(quant_mode, "has_any_quant"):
        return False
    try:
        return bool(quant_mode.has_any_quant(exclude_kv_cache=True))
    except TypeError:
        # Older signatures may not accept the kwarg; fall back.
        return bool(quant_mode.has_any_quant())


def _is_fp8_block_scale(quant_mode) -> bool:
    """Return True iff every underlying QuantMode uses FP8 block scales.

    This is the only base-weight quantization the native TRTLLM-gen MoE LoRA
    path supports today (the FP8 block-scale runner + Mn-bias GEMM1 + BGMV
    delta). BF16 and FP4 are deferred to follow-ups.
    """
    if quant_mode is None:
        return False
    objs = getattr(quant_mode, "objs", None)
    modes = objs if objs is not None else [quant_mode]
    saw_fp8_block = False
    for mode in modes:
        if not mode.has_fp8_block_scales():
            return False
        saw_fp8_block = True
    return saw_fp8_block


def check_moe_lora_supported(
    *,
    moe_backend_name: str,
    lora_config: Optional[LoraConfig],
    quant_config,
    layer_idx: Optional[int] = None,
) -> None:
    """Raise `ValueError` if a routed-expert MoE LoRA cannot run on the chosen
    backend / quant combination.

    Args:
        moe_backend_name: The resolved `moe_backend` string (e.g. "CUTLASS",
            "WIDEEP", "TRTLLM"). Comparison is case-insensitive.
        lora_config: The model's `LoraConfig`, or None.
        quant_config: The model's `QuantConfig`, or None. We only reject when
            the layer is actually quantized (`quant_mode.has_any_quant`).
        layer_idx: Optional layer index for diagnostic messages.

    Constraints:
        - MoE backend MUST be CUTLASS or TRTLLM.
        - CUTLASS: base weight quantization MUST be off (fp16/bf16) or per-tensor
          FP8 (qdq). FP8 block-scale / FP4 / INT8 / INT4 / W4A8 ... are rejected.
        - TRTLLM (trtllm-gen): base weight quantization MUST be FP8 block-scale.
          BF16 (no native runner) and FP4 (no Mn-bias GEMM1 kernel) are deferred.

    Other constraints (alltoall, min-latency, CUDA-graph) are enforced at
    runtime; we do not pre-check them here because they depend on per-call
    state that isn't available at factory time.
    """
    if not has_moe_lora_targets(lora_config):
        return

    prefix = f"[layer_idx={layer_idx}] " if layer_idx is not None else ""
    backend = (moe_backend_name or "").upper()

    if backend not in ("CUTLASS", "TRTLLM"):
        raise ValueError(
            f"{prefix}Routed-expert MoE LoRA requires moe_backend in "
            f"{{'CUTLASS', 'TRTLLM'}}; got moe_backend={moe_backend_name!r}. "
            f"Disable LoRA on MoE modules (remove {sorted(MOE_LORA_MODULE_NAMES)} "
            "from lora_config.lora_target_modules) or switch backend."
        )

    quant_mode = getattr(quant_config, "quant_mode", None) if quant_config is not None else None
    is_quantized = _is_quantized(quant_mode)

    if backend == "CUTLASS":
        if is_quantized and not _is_supported_quant(quant_mode):
            raise ValueError(
                f"{prefix}Routed-expert MoE LoRA on the Cutlass backend only "
                f"supports unquantized fp16/bf16 or per-tensor FP8 (qdq) base "
                f"weights; got quant_mode={quant_mode}. FP8 block-scale / FP4 / "
                "INT4 / INT8 / W4A8 base weights are not supported."
            )
    else:  # TRTLLM (trtllm-gen)
        if not (is_quantized and _is_fp8_block_scale(quant_mode)):
            raise ValueError(
                f"{prefix}Routed-expert MoE LoRA on the TRTLLM (trtllm-gen) "
                f"backend requires FP8 block-scale base weights; got "
                f"quant_mode={quant_mode}. BF16 (no native trtllm-gen MoE runner) "
                "and FP4 (no Mn-bias GEMM1 kernel) MoE LoRA are deferred; use the "
                "Cutlass backend for unquantized / per-tensor-FP8 MoE LoRA."
            )
