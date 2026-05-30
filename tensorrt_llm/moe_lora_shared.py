# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE LoRA shared-outer detection and fused-MoE kernel flags.

A routed-expert MoE LoRA module is "shared-outer" when one side (`A` or `B`) is
shared across all experts (the "outer", residual-stream side) and the other
side stays per-expert. The fused-MoE op honors this via `LoraParams::*_shared_a/b`
flags, zero-offsetting the per-expert pointer arithmetic in the kernel's
`setupLoraWorkspace`.

The shared side is detected from the weights: it shows up as expert slices that
are all identical, which is how a shared-outer adapter looks once its shared
matrix is replicated across experts (for example a PEFT export that broadcasts
the shared side before saving). Detection feeds the six-bool kernel flag dict
the fused-MoE op consumes.
"""

from collections.abc import Callable, Iterable

import torch

# Mapping from the TRT-LLM LoRA module name to the kernel-side module prefix
# the fused-MoE op uses. Matches the slot mapping in
# `fused_moe_cutlass._extract_moe_lora_tensors`:
#   moe_gate    -> fc1   (up projection)
#   moe_4h_to_h -> fc2   (down projection)
#   moe_h_to_4h -> gated (gate-side up projection in SwiGLU)
MODULE_TO_KERNEL_PREFIX: dict[str, str] = {
    "moe_gate": "fc1",
    "moe_4h_to_h": "fc2",
    "moe_h_to_4h": "gated",
}

# Per-side boolean flags accepted by `torch.ops.trtllm.fused_moe`.
KERNEL_FLAG_KEYS = (
    "fc1_shared_a",
    "fc1_shared_b",
    "fc2_shared_a",
    "fc2_shared_b",
    "gated_shared_a",
    "gated_shared_b",
)


def all_false_flags() -> dict[str, bool]:
    """Return a fresh dict of the six kernel flag keys, all set to False.

    Used as the default for adapters with no shared side.
    """
    return {key: False for key in KERNEL_FLAG_KEYS}


def _all_experts_equal(stacked: torch.Tensor) -> bool:
    """Return True iff `stacked` has at least two experts that are all equal."""
    if stacked.ndim < 1 or stacked.shape[0] < 2:
        return False
    return bool((stacked == stacked[0]).all().item())


def detect_shared_sides_from_stacked(
    a_stacked: torch.Tensor,
    b_stacked: torch.Tensor,
) -> tuple[bool, bool]:
    """Detect whether the A or B side of a stacked MoE LoRA module is shared.

    A side is shared-outer when every expert holds the same matrix. That is how
    a shared-outer adapter looks once its shared side is replicated across
    experts (for example a PEFT export that broadcasts the shared matrix before
    saving per-expert weights).

    Args:
        a_stacked: LoRA A weights, shape (num_experts, rank, in_dim).
        b_stacked: LoRA B weights, shape (num_experts, out_dim, rank).

    Returns:
        (shared_a, shared_b). A side is reported shared only when there are at
        least two experts and all expert slices are bit-identical.
    """
    return _all_experts_equal(a_stacked), _all_experts_equal(b_stacked)


def shared_sides_to_kernel_flags(
    detected: dict[str, tuple[bool, bool]],
) -> dict[str, bool]:
    """Build the six-bool kernel flag dict from per-module shared-side detection.

    Args:
        detected: Mapping from MoE LoRA module name to its detected
            (shared_a, shared_b) pair. Unknown module names are ignored.

    Returns:
        Dict with the six keys defined by `KERNEL_FLAG_KEYS`, each True iff that
        side of that module is shared across experts.
    """
    flags = all_false_flags()
    for module_name, (shared_a, shared_b) in detected.items():
        prefix = MODULE_TO_KERNEL_PREFIX.get(module_name)
        if prefix is None:
            continue
        if shared_a:
            flags[f"{prefix}_shared_a"] = True
        if shared_b:
            flags[f"{prefix}_shared_b"] = True
    return flags


def merge_moe_shared_flags_for_batch(
    active_uids: Iterable[str],
    get_flags: Callable[[str], dict[str, bool]],
) -> dict[str, bool] | None:
    """Merge per-adapter MoE shared-outer flags for one fused-MoE call.

    Args:
        active_uids: LoRA task ids present in the current batch.
        get_flags: Callable returning the six-bool flag dict for a uid (e.g.
            `LoraManager.get_moe_shared_flags`).

    Returns:
        The flag dict to set as `lora_params['moe_shared_flags']`, or None
        when there are no active uids or every flag is False.

    Raises:
        ValueError: when more than one uid is active and their flag dicts
            differ. The fused-MoE op applies one global flag set per call.
    """
    uids = list(active_uids)
    if not uids:
        return None
    merged: dict[str, bool] | None = None
    for uid in uids:
        flags = get_flags(uid)
        if merged is None:
            merged = flags
        elif merged != flags:
            raise ValueError(
                "MoE LoRA shared-outer flags must match across all adapters "
                f"in a batch; got mismatched flags for active uids {uids}. "
                "The fused-MoE op applies one global flag set per call."
            )
    assert merged is not None
    return merged if any(merged.values()) else None
