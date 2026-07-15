# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Availability checks for the MiniMax-M3 MSA sparse attention kernels.

The MSA path depends on the external fmha_sm100 package and runs only on the
SM100 architecture family (SM100 and SM103). These helpers gate backend
selection so a request for the MSA path fails early with a clear message on
unsupported systems.
"""

from __future__ import annotations

import importlib.util

import torch

from tensorrt_llm._utils import is_sm_100f

# fmha_sm100 runs on the SM100 architecture family (SM100 and SM103). Other
# architectures, including SM120, are not supported.
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


def _is_supported_device(capability: tuple[int, int] | None) -> bool:
    if capability is None:
        return False
    major, minor = capability
    return is_sm_100f(major * 10 + minor)


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
    if not _is_supported_device(capability):
        major, minor = capability
        raise RuntimeError(
            "MiniMax-M3 MSA sparse attention requires an SM100 or SM103 device, "
            f"but the current device is compute capability {major}.{minor}."
        )


__all__ = [
    "MSA_PACKAGE",
    "ensure_msa_available",
]
