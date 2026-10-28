# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARIA fast spec sampler (opt-in, ``TRTLLM_ARIA_FAST_SAMPLER=1``): the one-model rejection sampler's top-p renorm on
the image's compiled MiniMax-M3.1 TOPP_DET v3 op, for every target on GB300 (sm_103), not only the modeling_v2 TP4 /
TP8 targets that install it today.

Stock, the sampler runs per decode step (DSpark K7, B gen requests):
  - draft block: ``softmax`` + ``top_p_renorm_probs`` over B x 7 rows, then ``sampling_from_probs``;
  - verify:      ``softmax`` + ``top_p_renorm_probs`` over B x 8 rows, then ``chain_speculative_sampling``.
FlashInfer's deterministic top-p renorm is AIR top-p: an init kernel, three radix passes
(``AirTopPRenormRadixKernel``, 11 / 11 / 10 bits of the fp32 key, a few CTAs per row with 64-bit shared-memory
histogram atomics) and an apply kernel on one CTA per row; so 2 calls x 3 passes = the 6 radix launches per step.

The override is ``trtllm::minimax_m31_topp_renorm_det_v3`` (kernels/minimaxM31/toppRenormDetV3Sm103.cu): FlashInfer's
det renorm in one persistent launch, bit for bit on rows with top_p < 1. Rows with top_p >= 1 keep every token
(FlashInfer drops low tail buckets there in some rows; keep-all is what top_p = 1 means, and what the M3.1 TP4 / TP8
targets already ship). It returns None for inputs outside its contract, and the stock call runs. Softmax, sampling
and the rejection kernel (its Philox stream included) stay the stock ones.

The workspace is allocated once per device by ``prebuild`` (from ``warmup_sampling_module``, before warmup and any CUDA
graph capture); a call that finds none inside a capture declines (stock call) rather than allocate from the graph pool.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from tensorrt_llm.logger import logger

ENV = "TRTLLM_ARIA_FAST_SAMPLER"
ENABLED = os.environ.get(ENV, "").strip().lower() in ("1", "true", "on", "yes")

# Rows the top-p renorm override takes (max_batch_size 64 x 8 verify rows); its workspace holds this many rows.
TOPP_MAX_ROWS = 512
TOPP_MAX_CTAS_PER_ROW = 16

_STATE: dict = {}
_WS: dict = {}


def _op(name: str):
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    return getattr(torch.ops.trtllm, name, None)


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = ("sms", idx)
    if key not in _STATE:
        _STATE[key] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _STATE[key]


def _ctas(rows: int, sms: int, cap: int) -> int:
    """Largest power of two <= cap with ctas x rows <= sms; 1 past that (one CTA per row needs no co-residency)."""
    g = cap
    while g > 1 and g * rows > sms:
        g //= 2
    return g


def _device(device: Optional[torch.device]) -> torch.device:
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _workspace(kind: str, device: torch.device, numel: int, dtype: torch.dtype, zero: bool):
    key = (kind, device.index)
    ws = _WS.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        ws = (torch.zeros if zero else torch.empty)(numel, dtype=dtype, device=device)
        _WS[key] = ws
    return ws


def _topp_ws(device: torch.device):
    return _workspace("topp", device, TOPP_MAX_ROWS * _STATE["topp_ws_bytes"], torch.uint8, zero=True)


def top_p_renorm(probs: torch.Tensor, top_p: torch.Tensor) -> Optional[torch.Tensor]:
    """FlashInfer ``top_p_renorm_probs(probs, top_p, is_deterministic=True)`` via TOPP_DET v3, or None."""
    if (
        not isinstance(top_p, torch.Tensor)
        or not isinstance(probs, torch.Tensor)
        or not probs.is_cuda
        or probs.dtype != torch.float32
        or probs.dim() != 2
        or not probs.is_contiguous()
        or not 1 <= probs.shape[0] <= TOPP_MAX_ROWS
        or probs.shape[1] % 4 != 0
        or probs.data_ptr() % 16 != 0
        or top_p.dtype != torch.float32
        or top_p.dim() != 1
        or top_p.shape[0] != probs.shape[0]
        or not top_p.is_contiguous()
        or top_p.device != probs.device
    ):
        return None
    ws = _topp_ws(probs.device)
    if ws is None:
        return None
    rows = probs.shape[0]
    out = torch.empty_like(probs)
    _STATE["topp_op"](
        probs, top_p, ws, out, _ctas(rows, _sm_count(probs.device), TOPP_MAX_CTAS_PER_ROW), True, True
    )
    return out


def install() -> bool:
    """Once per process: when ``TRTLLM_ARIA_FAST_SAMPLER`` is set, on sm_103 with the op built, install the top-p
    renorm override and log the engagement (or why not) once. Returns whether it is installed."""
    if "installed" in _STATE:
        return _STATE["installed"]
    _STATE["installed"] = False
    if not ENABLED:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        logger.info(f"[ARIA fast sampler] {ENV}=1 but not taken: needs sm_103 (GB300); stock sampler")
        return False
    from . import flashinfer as fi_ops

    parts = []
    topp = _op("minimax_m31_topp_renorm_det_v3")
    if topp is not None:
        _STATE["topp_op"] = topp
        _STATE["topp_ws_bytes"] = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes())
        fi_ops.set_top_p_renorm_override(top_p_renorm)
        _STATE["installed"] = True
        parts.append(f"top-p renorm=TOPP_DET v3 (rows<={TOPP_MAX_ROWS})")
    _STATE["parts"] = parts
    logger.info(
        f"[ARIA fast sampler] {ENV}=1: "
        + ("engaged: " + "; ".join(parts) if parts else "not taken: TOPP_DET v3 is not built; stock sampler")
    )
    return _STATE["installed"]


def prebuild(device: Optional[torch.device] = None) -> None:
    """Install (once) and allocate the device's workspace; must run outside a CUDA-graph capture."""
    if not install():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    dev = _device(device)
    _topp_ws(dev)
