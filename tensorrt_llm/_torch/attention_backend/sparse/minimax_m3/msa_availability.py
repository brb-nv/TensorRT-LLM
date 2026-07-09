# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Availability checks for the MiniMax-M3 MSA sparse attention kernels.

The MSA path depends on the external fmha_sm100 package and runs only on
SM100 GPUs. These helpers gate backend selection so a request for the MSA
path fails early with a clear message on unsupported systems.
"""

from __future__ import annotations

import importlib.util

import torch

# fmha_sm100 targets the SM100 architecture (compute capability 10.x).
MSA_MIN_COMPUTE_CAPABILITY = (10, 0)
MSA_PACKAGE = "fmha_sm100"


def _has_msa_package() -> bool:
    return importlib.util.find_spec(MSA_PACKAGE) is not None


def _current_compute_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability()
    except RuntimeError:
        return None


def is_msa_available() -> bool:
    """Return True when the fmha_sm100 package and an SM100 GPU are present."""
    if not _has_msa_package():
        return False
    capability = _current_compute_capability()
    return capability is not None and capability >= MSA_MIN_COMPUTE_CAPABILITY


def ensure_msa_available() -> None:
    """Raise RuntimeError if the MSA sparse attention path cannot run here."""
    if not _has_msa_package():
        raise RuntimeError(
            f"MiniMax-M3 MSA sparse attention requires the {MSA_PACKAGE} package, "
            "which is not installed."
        )
    capability = _current_compute_capability()
    if capability is None:
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention requires a CUDA device and could not "
            "query the compute capability."
        )
    if capability < MSA_MIN_COMPUTE_CAPABILITY:
        major, minor = capability
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention requires compute capability "
            f"{MSA_MIN_COMPUTE_CAPABILITY[0]}.{MSA_MIN_COMPUTE_CAPABILITY[1]} or "
            f"higher, but the current device is {major}.{minor}."
        )


__all__ = [
    "MSA_MIN_COMPUTE_CAPABILITY",
    "MSA_PACKAGE",
    "ensure_msa_available",
    "is_msa_available",
]
