# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFLASH_FUSED_MERGE: one launch for the windowed DSpark block decode's later-key merge
(``modeling_dflash.dflash_merge_later_block_keys_``), routed per shape only where it is bitwise the compiled merge.

The compiled merge is three inductor Triton kernels: a q.k reduction, the per-head softmax / sigmoid weight, and the
per-element accumulation. Its bits depend on the reduction's autotuned launch config: the summation order follows the
config's chunk (R0_BLOCK) and rows per warp (XBLOCK / num_warps), and inductor picks that config by timing at its
first compile of each shape. ``_merge_kernel`` evaluates the same fp32 expression graph in the same order, with the
reduction laid out as one of those configs (``LAYOUTS``: heads per program = XBLOCK, num_warps, chunk = R0_BLOCK).

Which layout matches is only known from the live compiled merge, so each shape is verified before use: on its first
call outside CUDA-graph capture, and again after any recompile of the merge (its dynamo cache entries change), the
compiled merge and the layouts in order run on the call's own inputs plus seeded random probes; the shape routes to
the first layout whose outputs are bitwise equal on every probe, else keeps the compiled merge. A shape first seen
during capture keeps the compiled merge.

Armed process-wide by the MiniMax-M3.1 sm_107 targets' routing (``routing.arm_sm107_shared_opt_ins``) so that the
target directories (the AOT objects' source digests) are not edited. Kill switch TRTLLM_M31_DISABLE_DFLASH_FUSED_MERGE=1
(the compiled merge everywhere).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.triton_helpers import math as tl_math

from tensorrt_llm.logger import logger

KILL = "TRTLLM_M31_DISABLE_DFLASH_FUSED_MERGE"

#: (heads per program, num_warps, reduction chunk): one per bitwise class of the reduction's launch configs at
#: head_dim 128, the classes autotune has picked on sm_107 first.
LAYOUTS: Tuple[Tuple[int, int, int], ...] = (
    (16, 4, 64),
    (8, 8, 128),
    (8, 4, 128),
    (8, 2, 8),
    (8, 4, 16),
    (8, 8, 32),
    (4, 8, 64),
    (4, 8, 128),
    (2, 8, 128),
    (8, 8, 64),
    (8, 4, 32),
    (8, 2, 16),
    (8, 2, 32),
    (16, 2, 8),
    (16, 2, 16),
    (16, 2, 32),
    (16, 1, 8),
    (16, 1, 16),
)
#: Output elements the probes of one verification cover together (at least 2 probes, at most ``MAX_PROBES``).
PROBE_ELEMENTS = 1 << 21
MAX_PROBES = 64

_ARMED = False
_ROUTES: dict = {}
#: Verification count, wall seconds, layouts launched and shapes kept on the compiled merge (reported by tests / MBs).
STATS = {"verifications": 0, "verify_s": 0.0, "layout_trials": 0, "kept_compiled": 0}


def _killed(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "on", "yes")


def arm_sm107_fused_merge() -> None:
    """Take DFLASH_FUSED_MERGE in this process (the sm_107 targets), unless its kill switch is set."""
    global _ARMED
    _ARMED = not _killed(KILL)


def armed() -> bool:
    return _ARMED


@triton.jit
def _later_logit(
    q_ptr,
    q_rows,
    k_ptr,
    key_offset,
    row_valid,
    j,
    k: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    R0: tl.constexpr,
    HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    # The masked selects keep each product rounded on its own (one chunk) or fused only into its own lane's running
    # sum (chunked), as the compiled reduction does.
    if R0 == HEAD_DIM:
        cols = tl.arange(0, HEAD_DIM)[None, :]
        q = tl.load(q_ptr + (cols + HEAD_DIM * q_rows), row_valid, other=0.0).to(tl.float32)
        key = tl.load(k_ptr + (key_offset + cols) + 0 * q_rows, row_valid, other=0.0).to(tl.float32)
        logit = tl.sum(tl.where(row_valid, q * key, 0), 1)
    else:
        acc = tl.full([HEADS, R0], 0, tl.float32)
        for r0 in tl.range(0, HEAD_DIM, R0):
            cols = r0 + tl.arange(0, R0)[None, :]
            col_valid = cols < HEAD_DIM
            q = tl.load(q_ptr + (cols + HEAD_DIM * q_rows), col_valid & row_valid, other=0.0).to(
                tl.float32
            )
            key = tl.load(
                k_ptr + (key_offset + cols) + 0 * q_rows, col_valid & row_valid, other=0.0
            ).to(tl.float32)
            acc = tl.where(col_valid & row_valid, acc + q * key, acc)
        logit = tl.sum(acc, 1)
    distance = k - j
    later = distance > 0
    if WINDOW_RIGHT >= 0:
        later = later & (distance <= WINDOW_RIGHT)
    scaled = logit * tl.full([1], SCALE, tl.float32)
    return tl.where(later == 0, tl.full([1], float("-inf"), tl.float32), scaled)


@triton.jit
def _merge_kernel(
    out_ptr,
    lse_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    num_rows,
    v_stride_b,
    v_stride_t,
    v_stride_h,
    BLK: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    R0: tl.constexpr,
    SCALE: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    NUM_HEADS: tl.constexpr = NUM_KV_HEADS * GROUP
    HEAD_TILES: tl.constexpr = NUM_HEADS // HEADS
    pid = tl.program_id(0)
    token = pid // HEAD_TILES
    h0 = (pid % HEAD_TILES) * HEADS
    b = token // BLK
    j = token % BLK
    kvh = h0 // GROUP

    heads = token * NUM_HEADS + h0 + tl.arange(0, HEADS)
    q_rows = heads[:, None]
    row_valid = q_rows < num_rows
    key_base = (b * BLK * NUM_KV_HEADS + kvh) * HEAD_DIM
    key_stride: tl.constexpr = NUM_KV_HEADS * HEAD_DIM

    max_logit = _later_logit(
        q_ptr, q_rows, k_ptr, key_base, row_valid, j, 0, HEAD_DIM, R0, HEADS, SCALE, WINDOW_RIGHT
    )
    for k in tl.static_range(1, BLK):
        logit = _later_logit(
            q_ptr,
            q_rows,
            k_ptr,
            key_base + k * key_stride,
            row_valid,
            j,
            k,
            HEAD_DIM,
            R0,
            HEADS,
            SCALE,
            WINDOW_RIGHT,
        )
        max_logit = tl.maximum(max_logit, logit, tl.PropagateNan.ALL)
    finite = (max_logit == max_logit) & (
        tl_math.abs(max_logit) != tl.full([1], float("inf"), tl.float32)
    )
    zero = tl.full([1], 0.0, tl.float32)
    one = tl.full([1], 1.0, tl.float32)
    max_logit = tl.where(finite, max_logit, zero)

    first = _later_logit(
        q_ptr, q_rows, k_ptr, key_base, row_valid, j, 0, HEAD_DIM, R0, HEADS, SCALE, WINDOW_RIGHT
    )
    sum_exp = libdevice.exp(first - max_logit)
    for k in tl.static_range(1, BLK):
        logit = _later_logit(
            q_ptr,
            q_rows,
            k_ptr,
            key_base + k * key_stride,
            row_valid,
            j,
            k,
            HEAD_DIM,
            R0,
            HEADS,
            SCALE,
            WINDOW_RIGHT,
        )
        sum_exp = sum_exp + libdevice.exp(logit - max_logit)
    has_later = sum_exp > zero
    sum_safe = tl.where(has_later, sum_exp, one)
    lse = tl.load(lse_ptr + heads, heads < num_rows)
    sig_arg = (max_logit + tl_math.log(sum_safe)) - lse * tl.full(
        [1], 0.6931471805599453, tl.float32
    )
    weight = tl.where(has_later, tl.sigmoid(sig_arg), zero)
    prob_scale = weight / sum_safe

    cols = tl.arange(0, HEAD_DIM)[None, :]
    out_ptrs = out_ptr + (cols + HEAD_DIM * q_rows)
    value_base = v_ptr + (cols + b * v_stride_b + kvh * v_stride_h)
    merged = tl.load(out_ptrs).to(tl.float32) * (one - weight)[:, None]
    for k in tl.static_range(BLK):
        logit = _later_logit(
            q_ptr,
            q_rows,
            k_ptr,
            key_base + k * key_stride,
            row_valid,
            j,
            k,
            HEAD_DIM,
            R0,
            HEADS,
            SCALE,
            WINDOW_RIGHT,
        )
        coef = prob_scale * libdevice.exp(logit - max_logit)
        merged = merged + coef[:, None] * tl.load(value_base + k * v_stride_t).to(tl.float32)
    tl.store(out_ptrs, merged)


def supported(out, lse_log2, query, block_key, block_value) -> bool:
    """Whether the inputs are in the layout the kernel reads (else the compiled merge runs)."""
    if query.dim() != 4 or block_key.dim() != 4 or block_value.dim() != 4:
        return False
    B, blk, num_heads, head_dim = query.shape
    num_kv_heads = block_key.shape[2]
    return (
        num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and (num_heads // num_kv_heads) % 16 == 0
        and head_dim == 128
        and query.dtype == torch.bfloat16
        and block_key.dtype == torch.bfloat16
        and block_value.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and lse_log2.dtype == torch.float32
        and tuple(block_key.shape) == (B, blk, num_kv_heads, head_dim)
        and tuple(block_value.shape) == (B, blk, num_kv_heads, head_dim)
        and tuple(out.shape) == (B * blk, num_heads, head_dim)
        and lse_log2.numel() == B * blk * num_heads
        and query.is_contiguous()
        and block_key.is_contiguous()
        and out.is_contiguous()
        and lse_log2.is_contiguous()
        and block_value.stride(3) == 1
        and B * blk > 0
    )


def launch(
    layout, out, lse_log2, query, block_key, block_value, scale: float, window_right: int
) -> None:
    """The merge, in place in ``out``, with reduction ``layout`` (one of ``LAYOUTS``); inputs must be ``supported``."""
    heads, num_warps, r0 = layout
    B, blk, num_heads, head_dim = query.shape
    num_kv_heads = block_key.shape[2]
    _merge_kernel[(B * blk * (num_heads // heads),)](
        out,
        lse_log2,
        query,
        block_key,
        block_value,
        B * blk * num_heads,
        block_value.stride(0),
        block_value.stride(1),
        block_value.stride(2),
        BLK=blk,
        NUM_KV_HEADS=num_kv_heads,
        GROUP=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        HEADS=heads,
        R0=r0,
        SCALE=float(scale),
        WINDOW_RIGHT=int(window_right),
        num_warps=num_warps,
        num_stages=1,
    )


def _compiled_code(tip: Callable):
    for cell in getattr(tip, "__closure__", None) or ():
        orig = getattr(cell.cell_contents, "_torchdynamo_orig_callable", None)
        if orig is not None:
            return getattr(orig, "__code__", None)
    return None


def _epoch(tip: Callable):
    """Changes whenever ``tip`` may dispatch differently: its dynamo cache entries (else any dynamo compile in the
    process) and whether piecewise CUDA graphs run it eagerly."""
    from tensorrt_llm._torch.utils import is_piecewise_running

    entries = None
    code = _compiled_code(tip)
    if code is not None:
        try:
            from torch._dynamo.eval_frame import _debug_get_cache_entry_list

            entries = len(_debug_get_cache_entry_list(code))
        except (ImportError, AttributeError, RuntimeError, TypeError):
            entries = None
    if entries is None:
        entries = ("graphs", int(torch._dynamo.utils.counters["stats"]["unique_graphs"]))
    return (entries, bool(is_piecewise_running()))


def _probe_like(
    t: torch.Tensor, gen: torch.Generator, scale: float, shift: float = 0.0
) -> torch.Tensor:
    noise = (
        torch.randn(t.shape, device=t.device, generator=gen, dtype=torch.float32) * scale + shift
    )
    return torch.empty_like(t).copy_(noise)


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def verify(
    tip: Callable,
    out,
    lse_log2,
    query,
    block_key,
    block_value,
    scale: float,
    window_right: int,
    layouts=LAYOUTS,
) -> Optional[Tuple[int, int, int]]:
    """The first of ``layouts`` bitwise ``tip`` on the call's own inputs and seeded random probes of the same
    shapes and strides (``out`` is not written), or None."""
    t0 = time.perf_counter()
    n = min(MAX_PROBES, max(2, -(-PROBE_ELEMENTS // out.numel())))
    gen = torch.Generator(device=query.device)
    gen.manual_seed(0x5EED)
    probes = [(out, lse_log2, query, block_key)]
    for i in range(1, n):
        s = 4.0 if i % 2 else 1.0
        probes.append(
            (
                _probe_like(out, gen, 0.05),
                _probe_like(lse_log2, gen, 3.0, 2.0),
                _probe_like(query, gen, s),
                _probe_like(block_key, gen, s),
            )
        )
    refs = []
    for o, lse, q, k in probes:
        ref = o.clone()
        tip(ref, lse, q, k, block_value, scale, window_right)
        refs.append(ref)
    chosen = None
    for layout in layouts:
        STATS["layout_trials"] += 1
        equal = True
        for (o, lse, q, k), ref in zip(probes, refs):
            got = o.clone()
            launch(layout, got, lse, q, k, block_value, scale, window_right)
            if not _bitwise(got, ref):
                equal = False
                break
        if equal:
            chosen = layout
            break
    STATS["verifications"] += 1
    STATS["verify_s"] += time.perf_counter() - t0
    if chosen is None:
        STATS["kept_compiled"] += 1
    return chosen


def merge_later_block_keys_(
    tip: Callable, out, lse_log2, query, block_key, block_value, scale: float, window_right: int
) -> bool:
    """``tip(out, ...)`` in one launch where a verified layout is bitwise ``tip`` for these shapes; returns False
    (nothing written) where the caller has to run ``tip`` itself."""
    if not _ARMED or not supported(out, lse_log2, query, block_key, block_value):
        return False
    key = (
        tuple(query.shape),
        tuple(block_key.shape),
        tuple(block_value.stride()),
        query.device.index,
        float(scale),
        int(window_right),
    )
    epoch = _epoch(tip)
    route = _ROUTES.get(key)
    if route is None or route[0] != epoch:
        if torch.cuda.is_current_stream_capturing():
            return False
        layout = verify(tip, out, lse_log2, query, block_key, block_value, scale, window_right)
        route = _ROUTES[key] = (_epoch(tip), layout)
        logger.info(
            f"[DFLASH_FUSED_MERGE] {tuple(query.shape)} window {window_right}: "
            + (f"layout {layout}" if layout is not None else "compiled merge kept")
        )
    if route[1] is None:
        return False
    launch(route[1], out, lse_log2, query, block_key, block_value, scale, window_right)
    return True
