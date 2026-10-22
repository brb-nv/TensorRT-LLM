# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arch sets for MiniMax-M3 optimizations first enabled on Rubin (SM107) and widened to Blackwell (SM100/SM103).

Each widened gate reads its arch set once, at import, through ``gate_archs``:

- SM107 is always in the set, so Rubin behaves exactly as before. Rubin keeps the gate's own feature kill switch.
- SM100 and SM103 are added from the gate's ``*_ARCHS`` env var (a comma-separated SM list such as ``100,103,107``,
  or ``default_blackwell`` when the var is unset). Only 100 and 103 are taken from it; other archs are ignored.
- ``TRTLLM_M3_BLACKWELL_RUBIN_OPTS=0`` (umbrella kill switch) drops SM100/SM103 from every widened gate, which
  restores the stock Blackwell behavior.

The gate itself then checks ``get_sm_version() in archs`` (``get_sm_version`` is cached).
"""

import os
from typing import Iterable

RUBIN_SM = 107
BLACKWELL_SMS = frozenset((100, 103))
UMBRELLA_ENV = "TRTLLM_M3_BLACKWELL_RUBIN_OPTS"
UMBRELLA_ON = os.environ.get(UMBRELLA_ENV, "1") != "0"


def parse_arch_list(raw: str) -> frozenset:
    """``"100, 103,107"`` -> {100, 103, 107}; ``sm_103``/``103a`` style entries are accepted too."""
    archs = set()
    for item in raw.replace(";", ",").split(","):
        item = item.strip().lower()
        if item.startswith("sm_"):
            item = item[3:]
        item = item.rstrip("af")
        if not item:
            continue
        try:
            archs.add(int(item))
        except ValueError as e:
            raise ValueError(f"bad SM arch {item!r} in arch list {raw!r}") from e
    return frozenset(archs)


def gate_archs(env_name: str, default_blackwell: Iterable[int] = (100, 103)) -> frozenset:
    """The SM versions a widened gate runs on: SM107 plus the Blackwell SMs that ``env_name`` (else
    ``default_blackwell``) lists, unless the umbrella switch is off. Call once at import."""
    raw = os.environ.get(env_name)
    listed = parse_arch_list(raw) if raw is not None else frozenset(default_blackwell)
    blackwell = (listed & BLACKWELL_SMS) if UMBRELLA_ON else frozenset()
    return frozenset((RUBIN_SM,)) | blackwell


def describe(archs: frozenset) -> str:
    return ",".join(str(a) for a in sorted(archs))
