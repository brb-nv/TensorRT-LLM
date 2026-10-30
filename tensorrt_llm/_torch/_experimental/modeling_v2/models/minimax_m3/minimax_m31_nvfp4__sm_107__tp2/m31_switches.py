# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of this MiniMax-M3.1 target's default-on items (each target keeps its own copy: targets share
only the catalog).

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced.
The environment is read once, at import, so every rank and every step of a process sees the same setting (a step must
not change its graph mid-run). Each item lists its id and what the switch restores where it reads it.

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off.
"""

import os
from typing import FrozenSet

_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))


def _is_set(name: str, value: str) -> bool:
    """Whether the switch variable ``name`` is set to ``value``; raises on a value of neither kind."""
    word = value.strip().lower()
    if word in _SET:
        return True
    if word in _UNSET:
        return False
    raise ValueError(
        f"{name}={value!r}: expected 1 / true / on / yes (set) or 0 / false / off / no / empty (unset)"
    )


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = frozenset(
    name[len(_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_PREFIX) and _is_set(name, value)
)

# M31 H84 (opt-in, TRTLLM_M31_SM103_TP2_TARGET=1): this target also serves GB300 (sm_103) TP2. There the items below
# are off whatever their switch says:
# * not bitwise to the path they replace: TP2WIDE_LC2 (+ LC2_MIXED), S9B and S9B_MC
#   (drafter AR + norm order), S9BVR (the drafter's one-shot all-reduce instead of AUTO), MXDRAFT (drafter MXFP8
#   GEMV), VR_GEMV48 / LINEAR_CUBLASLT (33..64-row GEMMs), VR_DRAFT_FMHA_SM100F (another drafter FMHA kernel),
#   TP2AGG_QKV_A1 (opt-in; also sm_107a only);
# * Rubin-only: AGG_TSEL (its op is gated to sm_107), SOFTMAX_PATHB (its op requires SM107); R1A (routing does not
#   arm it on this route).
# The environment is read at import, as every switch here.
M31_SM103_TP2_ENV = "TRTLLM_M31_SM103_TP2_TARGET"
M31_SM103_TP2 = _is_set(M31_SM103_TP2_ENV, os.environ.get(M31_SM103_TP2_ENV, ""))
M31_SM103_TP2_FORCED_OFF: FrozenSet[str] = frozenset(
    (
        "TP2WIDE_LC2",
        "LC2_MIXED",
        "S9B",
        "S9B_MC",
        "S9BVR",
        "MXDRAFT",
        "VR_GEMV48",
        "LINEAR_CUBLASLT",
        "VR_DRAFT_FMHA_SM100F",
        "TP2AGG_QKV_A1",
        "AGG_TSEL",
        "SOFTMAX_PATHB",
        "R1A",
    )
)


def m31_sm103_tp2() -> bool:
    """Whether this process serves GB300 (sm_103) TP2 through this target (M31 H84): the env gate on an sm_103
    device. Decided once."""
    if "v" not in _M31_SM103_STATE:
        on = False
        if M31_SM103_TP2:
            import torch

            on = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)
        _M31_SM103_STATE["v"] = on
    return _M31_SM103_STATE["v"]


_M31_SM103_STATE: dict = {}
if m31_sm103_tp2():
    _DISABLED = _DISABLED | M31_SM103_TP2_FORCED_OFF


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process."""
    return item_id in _DISABLED


# Items that are OFF by default (rule 1b without a passing gate c, or a failed gate c): ``TRTLLM_M31_ENABLE_<ID>=1``
# opts in; the item's ``TRTLLM_M31_DISABLE_<ID>`` still wins.
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

_ENABLED: FrozenSet[str] = frozenset(
    name[len(_ENABLE_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_ENABLE_PREFIX) and _is_set(name, value)
)


def m31_enabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_ENABLE_<item_id>`` opted a default-off item in for this process."""
    return item_id in _ENABLED
