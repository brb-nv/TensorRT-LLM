# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-safe decode wrapper for MiniMax-M3 sparse attention.

Drives the external MSA (`fmha_sm100`) SM100 kernels with launch
arguments assembled from device tensors, so decode steps can be captured
into CUDA graphs and replayed correctly as sequence state advances.

Public surface:

* `M3DecodeGeometry`: compile/alloc-time geometry key for one layer family.
* `M3DecodeKernelDriver`: persistent-buffer kernel driver (proxy MQA + top-k
  block selection + block-sparse GQA) owned by the attention metadata.
* `resolve_decode_driver`: return the driver owned by the attention
  metadata (`m3_meta.decode_driver`), building an eager fallback only when
  no CUDA-graph metadata pre-built one.
"""

from .dispatch import M3DecodeGeometry, M3DecodeKernelDriver, resolve_decode_driver

__all__ = [
    "M3DecodeGeometry",
    "M3DecodeKernelDriver",
    "resolve_decode_driver",
]
