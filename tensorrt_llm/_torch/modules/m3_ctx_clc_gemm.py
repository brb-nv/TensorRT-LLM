# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTX prefill MXFP8 GEMM on the CuTe DSL persistent CLC mainloop of the fused QKV kernel (MiniMax-M3).

For the (N, K) shapes in ``_CTX_CLC_GEMM_SHAPES`` at >= ``_CTX_CLC_GEMM_MIN_ROWS``
rows on SM107 (and opted-in SM100 / SM103), ``MXFP8LinearMethod.apply`` runs the GEMM on the m3_qkv_fusion
kernel's GEMM-only handle instead of the CUTLASS ``mxfp8_mxfp8_gemm``. It
consumes the same E4M3 activation / swizzled UE8M0 scales and the module's
weight / swizzled weight scales / alpha. The result is deterministic but not
bitwise the CUTLASS result on SM107 (different K-reduction order inside the
MMA pipeline). ``TRTLLM_CTX_CLC_GEMM=0`` is the kill switch.

SM100 / SM103 run the same handle on the K32 SM100 MMA atom (see
``m3_qkv_fusion.kernel``) where ``TRTLLM_CTX_CLC_GEMM_ARCHS`` lists them
(opt-in; see ``blackwell_rubin_opts``), with the shape table
``_CTX_CLC_GEMM_SHAPES_SM10X`` (aggregated TP2 / TP4 MiniMax-M3 shapes, at most
2 heads per tile for 512 TMEM columns). On GB200 it was bitwise the CUTLASS
result for every shape in that table but mostly slower than the autotuned
CUTLASS tactic, hence opt-in. It launches with PDL there only when
``TRTLLM_ENABLE_PDL=1``.

Engagement is decided in eager execution only: under torch.compile M is
symbolic, and the compiled graph keeps the CUTLASS op unchanged. CTX chunks
above the piecewise CUDA-graph buckets run eagerly.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ..._utils import get_sm_version
from ..blackwell_rubin_opts import RUBIN_SM, gate_archs
from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE, IS_CUTLASS_DSL_RUBIN_AVAILABLE

_CTX_CLC_GEMM = os.environ.get("TRTLLM_CTX_CLC_GEMM", "1") != "0"
_CTX_CLC_GEMM_MIN_ROWS = 1024
# At and above this many rows the wider (4 heads = 512 columns) tile wins.
_CTX_CLC_GEMM_WIDE_ROWS = 12288
# (N, K) -> (heads per tile below _CTX_CLC_GEMM_WIDE_ROWS rows, at or above it); all 2-CTA M256.
_CTX_CLC_GEMM_SHAPES: Dict[Tuple[int, int], Tuple[int, int]] = {
    (6144, 4096): (2, 2),  # o_proj
    (3072, 6144): (2, 4),  # shared expert gate_up
    (12288, 6144): (2, 4),  # dense MLP gate_up
    (6144, 6144): (2, 4),  # dense MLP down
}
# SM107 always; SM100 / SM103 only where TRTLLM_CTX_CLC_GEMM_ARCHS lists them.
_CTX_CLC_GEMM_ARCHS = gate_archs("TRTLLM_CTX_CLC_GEMM_ARCHS", default_blackwell=())
# SM100 / SM103: the aggregated TP2 / TP4 (no attention DP) CTX shapes; 1-head tiles below _CTX_CLC_GEMM_WIDE_ROWS,
# 2-head tiles at or above it (the closest to CUTLASS on GB200).
_CTX_CLC_GEMM_SHAPES_SM10X: Dict[Tuple[int, int], Tuple[int, int]] = {
    (6144, 4096): (1, 2),  # TP2 o_proj
    (3072, 6144): (1, 2),  # TP2 shared expert gate_up
    (6144, 1536): (1, 2),  # TP2 shared expert down
    (12288, 6144): (1, 2),  # TP2 dense MLP gate_up
    (6144, 6144): (1, 2),  # TP2 dense MLP down, TP4 dense MLP gate_up
    (6144, 2048): (1, 2),  # TP4 o_proj
    (1536, 6144): (1, 2),  # TP4 shared expert gate_up
    (6144, 768): (1, 2),  # TP4 shared expert down
    (6144, 3072): (1, 2),  # TP4 dense MLP down
}
_SM10X_ENABLE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "0") == "1"

_platform_ok: Optional[bool] = None
_failed: Dict[Tuple[int, int], str] = {}
# Each (N, heads per tile) handle compiles on first use, and a narrow-tile variant is first needed only by a chunk of
# fewer than _CTX_CLC_GEMM_WIDE_ROWS rows, usually a long prompt's last chunk, inside a served request. The first take
# (the warmup) therefore builds every variant. Kill switch: TRTLLM_CTX_CLC_GEMM_PRECOMPILE=0 compiles each on first use.
_PRECOMPILE = os.environ.get("TRTLLM_CTX_CLC_GEMM_PRECOMPILE", "1") != "0"
_precompiled = [False]


def ctx_clc_gemm_enabled() -> bool:
    """Kill switch and platform: SM107 with the Rubin CuTe DSL helpers importable, or a listed SM100 / SM103 with the
    CuTe DSL importable."""
    global _platform_ok
    if not _CTX_CLC_GEMM:
        return False
    if _platform_ok is None:
        sm = get_sm_version()
        if sm == RUBIN_SM:
            _platform_ok = bool(IS_CUTLASS_DSL_RUBIN_AVAILABLE)
        else:
            _platform_ok = bool(IS_CUTLASS_DSL_AVAILABLE) and sm in _CTX_CLC_GEMM_ARCHS
    return _platform_ok


def _shapes() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """The (N, K) -> heads-per-tile table of this device."""
    return _CTX_CLC_GEMM_SHAPES if get_sm_version() == RUBIN_SM else _CTX_CLC_GEMM_SHAPES_SM10X


def ctx_clc_heads_per_tile(m: int, n: int, k: int) -> Optional[int]:
    """The tile width for an eligible (M, N, K), or None when the shape stays on CUTLASS."""
    hpts = _shapes().get((n, k))
    if hpts is None or m < _CTX_CLC_GEMM_MIN_ROWS:
        return None
    return hpts[1] if m >= _CTX_CLC_GEMM_WIDE_ROWS else hpts[0]


def _get_gemm(n: int, heads_per_tile: int):
    from ..kernels.rubin.m3_qkv_fusion.plain_gemm import get_gemm

    if get_sm_version() == RUBIN_SM:
        return get_gemm(n, heads_per_tile)
    return get_gemm(n, heads_per_tile, enable_pdl=_SM10X_ENABLE_PDL)


def _precompile_all() -> None:
    """Build every (N, heads per tile) handle of _CTX_CLC_GEMM_SHAPES; a failure is recorded as on first use."""
    _precompiled[0] = True
    shapes = _shapes()
    for (n, k), hpts in shapes.items():
        for hpt in sorted(set(hpts)):
            if (n, hpt) in _failed:
                continue
            try:
                _get_gemm(n, hpt)
            except Exception as exc:  # noqa: BLE001  (a DSL compile failure must leave serving on CUTLASS)
                _failed[(n, hpt)] = f"{type(exc).__name__}: {exc}"
                _skip(n, k, hpt, f"compile failed ({_failed[(n, hpt)]})")
    logger.info_once(
        f"MiniMax-M3 CTX CLC MXFP8 GEMM: {len(shapes)} shapes precompiled for every tile width",
        key="m3_ctx_clc_gemm_precompiled",
    )


def _skip(n: int, k: int, hpt: int, reason: str) -> None:
    logger.warning_once(
        f"MiniMax-M3 CTX CLC MXFP8 GEMM not taken (N={n}, K={k}, hpt={hpt}): {reason}; using CUTLASS",
        key=f"m3_ctx_clc_gemm_skip_{n}_{k}_{hpt}",
    )


def _clc_gemm_impl(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    heads_per_tile: int,
) -> torch.Tensor:
    gemm = _get_gemm(weight.shape[0], heads_per_tile)
    out = torch.empty((act.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=act.device)
    gemm.run_args(gemm.make_args(act, act_sf, weight, weight_sf, alpha, out))
    return out


@torch.library.custom_op("trtllm::m3_ctx_clc_mxfp8_gemm", mutates_args=())
def m3_ctx_clc_mxfp8_gemm(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    heads_per_tile: int,
) -> torch.Tensor:
    """bf16 ``alpha * act @ weight.T`` for MXFP8 operands with swizzled UE8M0 K32 scales (SM107 CLC mainloop).

    The caller validates the operands (``try_ctx_clc_mxfp8_gemm``); an invalid layout raises AssertionError.
    """
    gemm = _get_gemm(weight.shape[0], heads_per_tile)
    reason = gemm.check(act, act_sf, weight, weight_sf, alpha)
    assert not reason, reason
    return _clc_gemm_impl(act, act_sf, weight, weight_sf, alpha, heads_per_tile)


@m3_ctx_clc_mxfp8_gemm.register_fake
def _(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    heads_per_tile: int,
) -> torch.Tensor:
    return act.new_empty((act.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def try_ctx_clc_mxfp8_gemm(
    act: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The CLC GEMM output for an eligible eager CTX call, or None for the caller's CUTLASS path.

    Call only from eager code (M must be a concrete row count) after ``ctx_clc_gemm_enabled()``.
    """
    m = act.shape[0] if act.dim() == 2 else -1
    n, k = weight.shape[0], weight.shape[-1]
    hpt = ctx_clc_heads_per_tile(m, n, k)
    if hpt is None:
        return None
    if _PRECOMPILE and not _precompiled[0]:
        _precompile_all()
    key = (n, hpt)
    if key in _failed:
        return None
    if out_dtype != torch.bfloat16:
        _skip(n, k, hpt, f"output dtype {out_dtype} is not bfloat16")
        return None
    try:
        gemm = _get_gemm(n, hpt)
    except Exception as exc:  # noqa: BLE001  (a DSL compile failure must leave serving on CUTLASS)
        _failed[key] = f"{type(exc).__name__}: {exc}"
        _skip(n, k, hpt, f"compile failed ({_failed[key]})")
        return None
    reason = gemm.check(act, act_sf, weight, weight_sf, alpha)
    if reason:
        _skip(n, k, hpt, reason)
        return None
    try:
        # Direct call: in eager the custom-op dispatch would only add host time per GEMM.
        out = _clc_gemm_impl(act, act_sf, weight, weight_sf, alpha, hpt)
    except AssertionError as exc:
        _skip(n, k, hpt, f"assertion {exc}")
        return None
    logger.info_once(
        f"MiniMax-M3 CTX CLC MXFP8 GEMM taken (N={n}, K={k}, hpt={hpt})",
        key=f"m3_ctx_clc_gemm_taken_{n}_{k}_{hpt}",
    )
    return out
