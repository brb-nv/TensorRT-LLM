# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation helpers for routed-expert (MoE) LoRA.

The MVP supports MoE LoRA only on the Cutlass backend with unquantized fp16/bf16
base weights. This module provides a single helper, `check_moe_lora_supported`,
that callers (typically the MoE factory in `create_moe.py`) can invoke at
construction time so that unsupported combinations fail loudly instead of
silently dropping the LoRA contribution at runtime.

Runtime-only rejections (min-latency mode, alltoall, FP4 base, CUDA-graph
without slot pointers) are enforced in the C++ thop / runtime call paths and
are NOT re-checked here.
"""

from collections.abc import Iterable

from tensorrt_llm.lora_helper import LoraConfig

from .moe_layout import MOE_LORA_MODULE_NAMES


def _normalize_targets(lora_target_modules: Iterable[str]) -> set[str]:
    return {name.lower() for name in lora_target_modules or []}


def has_moe_lora_targets(lora_config: LoraConfig | None) -> bool:
    """Return True iff `lora_config` requests LoRA on any routed-expert MoE module."""
    if lora_config is None:
        return False
    targets = _normalize_targets(lora_config.lora_target_modules)
    return bool(MOE_LORA_MODULE_NAMES & targets)


def check_moe_lora_supported(
    *,
    moe_backend_name: str,
    lora_config: LoraConfig | None,
    quant_config,
    layer_idx: int | None = None,
) -> None:
    """Raise `ValueError` if a routed-expert MoE LoRA cannot run on the chosen
    backend / quant combination.

    Args:
        moe_backend_name: The resolved `moe_backend` string (e.g. "CUTLASS",
            "WIDEEP", "TRTLLM"). Comparison is case-insensitive.
        lora_config: The model's `LoraConfig`, or None.
        quant_config: The model's `QuantConfig`, or None. We only reject when
            the layer is actually quantized. The per-layer `layer_quant_mode`
            is preferred over the global `quant_mode` so this matches the view
            used for backend selection under mixed/per-layer quantization.
        layer_idx: Optional layer index for diagnostic messages.

    Constraints (MVP):
        - MoE backend MUST be CUTLASS.
        - Base weight quantization MUST be off (no FP8 / FP4 / INT8 / INT4 / W4A8 ...).

    Other constraints (alltoall, min-latency, FP4, CUDA-graph) are enforced at
    runtime; we do not pre-check them here because they depend on per-call
    state that isn't available at factory time.
    """
    if not has_moe_lora_targets(lora_config):
        return

    prefix = f"[layer_idx={layer_idx}] " if layer_idx is not None else ""

    if (moe_backend_name or "").upper() != "CUTLASS":
        raise ValueError(
            f"{prefix}Routed-expert MoE LoRA requires moe_backend='CUTLASS'; got "
            f"moe_backend={moe_backend_name!r}. Disable LoRA on MoE modules "
            f"(remove {sorted(MOE_LORA_MODULE_NAMES)} from "
            "lora_config.lora_target_modules) or switch to the Cutlass MoE backend."
        )

    if quant_config is not None:
        # Prefer the per-layer quant mode so this check agrees with the
        # backend-selection logic in create_moe.py, which also uses
        # `layer_quant_mode`. Fall back to the global `quant_mode`.
        quant_mode = getattr(quant_config, "layer_quant_mode", None)
        if quant_mode is None:
            quant_mode = getattr(quant_config, "quant_mode", None)
        is_quantized = False
        if quant_mode is not None and hasattr(quant_mode, "has_any_quant"):
            try:
                is_quantized = bool(quant_mode.has_any_quant(exclude_kv_cache=True))
            except TypeError:
                # Older signatures may not accept the kwarg; fall back.
                is_quantized = bool(quant_mode.has_any_quant())
        if is_quantized:
            raise ValueError(
                f"{prefix}Routed-expert MoE LoRA MVP only supports unquantized "
                f"fp16/bf16 base weights; got quant_mode={quant_mode}. "
                "FP8/FP4/INT4/INT8 base weights combined with MoE LoRA are not "
                "supported yet."
            )
