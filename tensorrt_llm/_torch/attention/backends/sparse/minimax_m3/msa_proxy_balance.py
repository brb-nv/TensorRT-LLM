# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""o12a: work-balanced KV-split plans for the MiniMax-M3 MSA indexer proxy pass.

The proxy pass (fmha_sm100, output_maxscore only) scores every cached index-K block for every new context query, so
its work is O(new tokens x visible prefix). Stock plans it in one of two ways (msa_indexer.segmented_proxy_plan):

* unsplit: one CTA per (query tile, head), 256-query tiles when any row has more than 128 queries;
* split: 128-query segments with one uniform KV split count, chosen as num_sms // (segments x heads), and only when
  that is at least 1, some row has more than 64 queries and some prefix is at least 4096 tokens.

The geometries AgentX produces leave most SMs idle under that rule:

* 5k-16k-token chunks have 40-127 segments x 2 heads, so stock runs them unsplit as 20-64 tiles x 2 heads on ~150
  SMs, with causal tile lengths that differ by the chunk length;
* a step that mixes a long-prefix row with another row is set by the long row's few tiles (a tail);
* rows of at most 64 queries are never split, so one or two CTAs scan the whole prefix.

This module plans the proxy on the host instead. Rows are cut into the same 128-query segments as stock's split
plan (same page-table gather), every (segment, head) tile has a KV range of ki 256-key iterations, and the
concatenated work of all tiles is cut into num_ctas equal ranges (McNaughton's wrap-around rule): each CTA gets
about W / num_ctas iterations, a tile is split only where a CTA boundary falls inside it, and a boundary that would
leave a piece shorter than min_piece iterations is moved to the tile edge. No GPU planner kernel runs.

Exactness: every 128-key block max of every query is still produced by exactly one piece, from the same Q.K
products the unsplit kernel computes, and a max is order-free, so the scores are bitwise those of the stock plans
(the same argument stock's split plan relies on; checked by test/o12a_bench.py exact). The plan uses the
library's own work-list format (fmha_sm100 HostPrecomputedTileScheduler: per-CTA [begin, end) ranges, packed
(tile, head, segment) items, per-item KV tile ranges and split index), which its adaptive planner also emits with
arbitrary per-tile piece boundaries.

Switch: M3_O12A_PROXY_SPLIT=0 restores stock routing exactly (msa_indexer then runs the stock functions only).
stack-a: the master switch M3AX_STACK_A=0 (also false/off/no) does the same.
"""

from __future__ import annotations

import math
import os
from typing import List, NamedTuple, Optional, Sequence, Tuple

# fmha_sm100 dispatch (jit.py _FMHA_SM100_DISPATCH): a 128-query tile pairs with 256-key KV tiles, a 256-query tile
# with 128-key ones; split-KV exists only for the 128-query tile. Both tiles cover 32768 (query, key) pairs per
# iteration, which is the cost unit used below ("iterations").
SEG_LEN = 128
QO_TILE = 128
KV_TILE = 256
QO_TILE_WIDE = 256
KV_TILE_WIDE = 128
# fmha_sm100 plan.cuh cost model: 43 per KV iteration, 110 per work item -> ~2.6 iterations per extra piece.
_LIB_PIECE_OVH = 110.0 / 43.0
# fmha_sm100 api._compute_pack_factor candidates.
_PACK_FACTORS = (1, 2, 4, 6, 8, 16)


class Config(NamedTuple):
    enabled: bool  # M3_O12A_PROXY_SPLIT (default 1) and stack-a's M3AX_STACK_A; 0 = stock routing, bit for bit
    force: bool  # M3_O12A_FORCE: take the balanced plan whenever it applies (diagnostics / tests)
    margin_rel: float  # M3_O12A_MARGIN_REL: required predicted saving, fraction of stock's lower bound
    margin_iters: float  # M3_O12A_MARGIN_ITERS: required predicted saving, iterations per launch
    iter_ratio: float  # M3_O12A_ITER_RATIO: cost of a 128x256 iteration / a 256x128 one (unsplit wide tiles)
    piece_ovh: float  # M3_O12A_PIECE_OVH: cost of one work item beyond its iterations, in iterations
    min_piece: int  # M3_O12A_MIN_PIECE: shortest piece a CTA boundary may cut, in iterations
    max_pieces: int  # M3_O12A_MAX_PIECES: most pieces of one tile (split-KV workspace rows)
    pack: bool  # M3_O12A_PACK (default 1): pack index heads into 128-row tiles as fmha_sm100 does for <= 32 queries


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _stack_master_on() -> bool:
    """stack-a master switch M3AX_STACK_A (default on): 0/false/off/no turns o12a off like M3_O12A_PROXY_SPLIT=0."""
    return os.environ.get("M3AX_STACK_A", "1").strip().lower() not in ("0", "false", "off", "no")


_CONFIG: Optional[Config] = None


def config() -> Config:
    """The o12a knobs, read from the environment once per process (reload_config() re-reads them)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(
            enabled=_stack_master_on() and os.environ.get("M3_O12A_PROXY_SPLIT", "1") != "0",
            force=os.environ.get("M3_O12A_FORCE", "0") == "1",
            margin_rel=_env_float("M3_O12A_MARGIN_REL", 0.03),
            margin_iters=_env_float("M3_O12A_MARGIN_ITERS", 8.0),
            iter_ratio=_env_float("M3_O12A_ITER_RATIO", 1.0),
            piece_ovh=_env_float("M3_O12A_PIECE_OVH", _LIB_PIECE_OVH),
            min_piece=max(1, int(_env_float("M3_O12A_MIN_PIECE", 4))),
            max_pieces=max(2, int(_env_float("M3_O12A_MAX_PIECES", 64))),
            pack=os.environ.get("M3_O12A_PACK", "1") != "0",
        )
    return _CONFIG


def reload_config() -> Config:
    global _CONFIG
    _CONFIG = None
    return config()


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pack_factor(max_qo_len: int, num_qo_heads: int, num_kv_heads: int = 1) -> int:
    """fmha_sm100 api._compute_pack_factor (MQA proxy: num_kv_heads 1)."""
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1


def _kv_iters(kv_len: int, offset: int, tile: int, qo_tile: int, kv_tile: int, pf: int) -> int:
    """fmha_sm100 plan.cuh compute_kv_iters (causal) for query tile `tile` of a row."""
    packed_q_end = (tile + 1) * qo_tile
    q_end = (packed_q_end - 1) // pf + 1 if pf > 1 else packed_q_end
    eff = min(q_end + offset, kv_len)
    return _cdiv(eff, kv_tile) if eff > 0 else 0


class Segments(NamedTuple):
    qo: List[int]  # queries per segment (<= SEG_LEN)
    kv: List[int]  # causal KV length per segment (its last query's)
    off: List[int]  # causal offset per segment (cached prefix + first query's index in the row)
    pages: List[int]  # pages of each segment's page-table copy
    shifts: List[int]  # row's first page in kv_indices minus the segment's first gathered slot
    num_gathered: int


def segment_rows(qo_lens: Sequence[int], qo_offsets: Sequence[int], page_size: int) -> Segments:
    """Rows as segments of at most SEG_LEN queries, exactly as msa_indexer._segmented_proxy_plan cuts them."""
    seg_qo, seg_kv, seg_off, seg_pages, seg_shifts = [], [], [], [], []
    row_first_page = num_gathered = 0
    for qo_len, prefix in zip(qo_lens, qo_offsets):
        for first in range(0, qo_len, SEG_LEN):
            n = min(SEG_LEN, qo_len - first)
            kv_len = prefix + first + n
            pages = (kv_len + page_size - 1) // page_size
            seg_qo.append(n)
            seg_kv.append(kv_len)
            seg_off.append(prefix + first)
            seg_pages.append(pages)
            seg_shifts.append(row_first_page - num_gathered)
            num_gathered += pages
        row_first_page += (prefix + qo_len + page_size - 1) // page_size
    return Segments(seg_qo, seg_kv, seg_off, seg_pages, seg_shifts, num_gathered)


class Layout(NamedTuple):
    """A work list in fmha_sm100's split-KV format, before packing."""

    cta_begin: List[int]  # per CTA: first item index (items are CTA-major)
    item_tile: List[int]  # per item: tile index (tile = segment * heads + head)
    item_kb: List[int]  # per item: first KV iteration
    item_ke: List[int]  # per item: one past the last KV iteration
    item_sub: List[int]  # per item: split index within its tile
    tile_pieces: List[int]  # per tile: number of pieces
    num_ctas_used: int
    makespan: float  # max over CTAs of iterations + piece_ovh per item


def wrap_layout(ki: Sequence[int], num_ctas: int, *, piece_ovh: float, min_piece: int, max_pieces: int) -> Layout:
    """Cut the concatenated tiles into num_ctas ranges of equal work (McNaughton's wrap-around).

    Tile order is kept, so the two heads of a segment (same index-K pages) stay adjacent and every CTA's first
    piece starts at the top of its tile's KV range (fmha_sm100 walks a piece from its last KV tile down).
    """
    n = len(ki)
    total = sum(ki)
    ki_max = max(ki) if n else 0
    # Work is measured as iterations plus piece_ovh per work item: each tile costs ki + piece_ovh, and every CTA
    # boundary that falls inside a tile adds one more item.
    cost = total + piece_ovh * n
    # Range per CTA: the even share, but long enough that no tile is cut into more than max_pieces pieces and no
    # CTA is handed less than two minimum pieces of work.
    share = (cost + piece_ovh * num_ctas) / num_ctas if num_ctas > 0 else float(cost)
    share = max(share, ki_max / max(1, max_pieces - 1) + piece_ovh, 2.0 * min_piece + piece_ovh)
    ctas = max(1, min(num_ctas, int(math.ceil(cost / (share - piece_ovh))))) if total > 0 else 1
    step = cost / ctas
    # CTA boundaries, placed in cost space (a tile's overhead sits before its first iteration), mapped to
    # iteration positions in the concatenated tiles and snapped to a tile edge when within min_piece of it.
    bounds = [0]
    t = 0  # tile containing the current boundary
    it_start = 0  # iteration offset of tile t
    cost_start = 0.0  # cost offset of tile t
    for c in range(1, ctas):
        y = c * step
        while t < n and cost_start + ki[t] + piece_ovh <= y:
            it_start += ki[t]
            cost_start += ki[t] + piece_ovh
            t += 1
        if t >= n:
            x = total
        else:
            p = int(round(y - cost_start - piece_ovh))
            if p < min_piece:
                p = 0
            elif ki[t] - p < min_piece:
                p = ki[t]
            x = it_start + p
        x = max(x, bounds[-1])
        bounds.append(x)
    bounds.append(total)
    # Walk the tiles and the CTA ranges together; a piece ends at whichever boundary comes first.
    item_tile, item_kb, item_ke, item_sub = [], [], [], []
    add_tile, add_kb, add_ke, add_sub = item_tile.append, item_kb.append, item_ke.append, item_sub.append
    tile_pieces = [0] * n
    cta_begin = [0]
    makespan = 0.0
    c = 0
    nxt = bounds[1]  # end of CTA c's range
    load_it = 0  # iterations on CTA c
    first_item = 0  # CTA c's first item
    start = 0
    for u in range(n):
        end = start + ki[u]
        p = start
        sub = 0
        while p < end:
            while nxt <= p and c + 1 < ctas:
                # Close CTA c, open the next one (empty ranges included).
                makespan = max(makespan, load_it + piece_ovh * (len(item_tile) - first_item))
                c += 1
                cta_begin.append(len(item_tile))
                first_item = len(item_tile)
                load_it = 0
                nxt = bounds[c + 1]
            q = end if end < nxt else nxt
            add_tile(u)
            add_kb(p - start)
            add_ke(q - start)
            add_sub(sub)
            sub += 1
            load_it += q - p
            p = q
        tile_pieces[u] = sub
        start = end
    makespan = max(makespan, load_it + piece_ovh * (len(item_tile) - first_item))
    n_items = len(item_tile)
    cta_begin.extend([n_items] * (num_ctas - len(cta_begin)))
    return Layout(cta_begin, item_tile, item_kb, item_ke, item_sub, tile_pieces, ctas, makespan)


def stock_lower_bound(
    qo_lens: Sequence[int],
    qo_offsets: Sequence[int],
    segs: Segments,
    seg_ki: Sequence[int],
    *,
    num_heads: int,
    num_ctas: int,
    stock_splits: int,
    piece_ovh: float,
    iter_ratio: float,
) -> float:
    """A lower bound on the makespan (in 128x256-iteration units) of the plan stock builds for these rows.

    max(longest work item, total work / CTAs): fmha_sm100's direct greedy planner can only do worse (it assigns
    items in launch order, not longest first), so a balanced plan predicted to beat this bound beats stock.
    """
    if stock_splits <= 0:
        # Unsplit plan over the rows themselves (msa_backend plan_for, num_kv_splits=1).
        max_q = max(qo_lens)
        pf = pack_factor(max_q, num_heads)
        heads = num_heads // pf
        wide = max_q * pf > QO_TILE
        qo_tile, kv_tile = (QO_TILE_WIDE, KV_TILE_WIDE) if wide else (QO_TILE, KV_TILE)
        longest = 0
        work = 0.0
        for q, off in zip(qo_lens, qo_offsets):
            kv_len = off + q
            for tile in range(_cdiv(q * pf, qo_tile)):
                k = _kv_iters(kv_len, off, tile, qo_tile, kv_tile, pf)
                if k <= 0:
                    continue
                longest = max(longest, k)
                work += (k + piece_ovh) * heads
        bound = max(longest + piece_ovh, work / num_ctas)
        return bound if wide else bound * iter_ratio
    longest = 0
    work = 0.0
    for k in seg_ki:
        splits = min(stock_splits, max(1, k // 4))
        longest = max(longest, _cdiv(k, splits))
        work += (k + splits * piece_ovh) * num_heads
    return max(longest + piece_ovh, work / num_ctas) * iter_ratio


class Decision(NamedTuple):
    use_balanced: bool
    reason: str
    stock_lb: float
    balanced: float
    layout: Optional[Layout]
    segs: Optional[Segments]
    pf: int


def decide(
    qo_lens: Sequence[int],
    qo_offsets: Sequence[int],
    *,
    num_index_heads: int,
    page_size: int,
    num_ctas: int,
    stock_splits: int,
    cfg: Optional[Config] = None,
) -> Decision:
    """Whether to replace stock's proxy plan with a balanced one for these rows (pure host arithmetic)."""
    cfg = cfg or config()
    if not qo_lens or max(qo_lens) <= 0:
        return Decision(False, "empty", 0.0, 0.0, None, None, 1)
    segs = segment_rows(qo_lens, qo_offsets, page_size)
    pf = pack_factor(max(segs.qo), num_index_heads) if cfg.pack else 1
    heads = num_index_heads // pf
    seg_ki = [_kv_iters(kv, off, 0, QO_TILE, KV_TILE, pf) for kv, off in zip(segs.kv, segs.off)]
    ki = [k for k in seg_ki for _ in range(heads)]
    layout = wrap_layout(
        ki, num_ctas, piece_ovh=cfg.piece_ovh, min_piece=cfg.min_piece, max_pieces=cfg.max_pieces
    )
    balanced = layout.makespan * cfg.iter_ratio  # 128-query tiles, in wide-tile iteration units
    stock_lb = stock_lower_bound(
        qo_lens,
        qo_offsets,
        segs,
        seg_ki,
        num_heads=num_index_heads,
        num_ctas=num_ctas,
        stock_splits=stock_splits,
        piece_ovh=cfg.piece_ovh,
        iter_ratio=cfg.iter_ratio,
    )
    if cfg.force:
        return Decision(True, "forced", stock_lb, balanced, layout, segs, pf)
    gain = stock_lb - balanced
    if gain >= max(cfg.margin_iters, cfg.margin_rel * stock_lb):
        return Decision(True, "gain", stock_lb, balanced, layout, segs, pf)
    return Decision(False, "margin", stock_lb, balanced, layout, segs, pf)


# ----------------------------------------------------------------------------------------------------------------
# Device plan (needs torch and fmha_sm100; everything above is plain Python so tools/cpu_selftest.py runs anywhere)
# ----------------------------------------------------------------------------------------------------------------

_API_NAMES = (
    "_make_plan_info",
    "_alloc_workspace_buf",
    "_BuffTag",
    "_get_num_cta",
)


def library_api(fmha_sm100):
    """fmha_sm100.api, or None when it lacks the internals this plan is built from (then stock routing is kept)."""
    api = getattr(fmha_sm100, "api", None)
    if api is None:
        try:
            import importlib

            api = importlib.import_module(fmha_sm100.__name__ + ".api")
        except ImportError:
            return None
    if not all(hasattr(api, name) for name in _API_NAMES):
        return None
    return api


def _stage(parts, dtype, device, align: int, h2d):
    """Ship lists to the device in one non-blocking copy; return a view per list, each starting `align`-aligned."""
    flat = []
    spans = []
    for part in parts:
        start = len(flat)
        flat.extend(part)
        spans.append((start, len(part)))
        pad = (-len(flat)) % align
        if pad:
            flat.extend([0] * pad)
    dev = h2d(flat, dtype, device)
    return [dev[s : s + n] for s, n in spans]


_DUMMY_WORKSPACE_O = {}


def _dummy_workspace_o(device):
    """A 16-element bf16 stand-in for the split-KV partial-O workspace.

    The proxy runs fmha_sm100 in OnlyScore mode (output_o=False): every partial-O write is compiled out
    (sm100_fmha_fwd_mainloop: `kNeedOutput && IsSplitKV`; epilogue O store: `NeedOutput_`), so the workspace is
    only an address the launch passes through. _fmha_sm100 still needs a tensor to select the split variant.
    test/o12a_bench.py exact checks with a guard pattern around this buffer that nothing writes to it.
    """
    import torch

    key = str(device)
    buf = _DUMMY_WORKSPACE_O.get(key)
    if buf is None:
        buf = torch.zeros(16, dtype=torch.bfloat16, device=device)
        _DUMMY_WORKSPACE_O[key] = buf
    return buf


def build_plan(
    fmha_sm100,
    decision: Decision,
    *,
    num_index_heads: int,
    kv_indices,
    num_ctas: int,
    workspace_o=None,
):
    """The fmha_sm100 plan tuple and gathered page table for a balanced layout, or None if the library lacks the
    plan internals (callers then keep stock's plan)."""
    import torch

    from tensorrt_llm._utils import async_tensor_h2d

    api = library_api(fmha_sm100)
    if api is None or decision.layout is None:
        return None
    segs, lay, pf = decision.segs, decision.layout, decision.pf
    heads = num_index_heads // pf
    device = kv_indices.device
    nseg = len(segs.qo)
    seg_q_packed = [q * pf for q in segs.qo]
    total_q_packed = sum(seg_q_packed)
    max_kv = max(segs.kv)
    max_k_tiles = _cdiv(_cdiv(max_kv, 128), 128) * 128
    if heads * max_k_tiles * total_q_packed > (1 << 31):
        return None  # fmha_sm100 would drop max_score (api._fmha_sm100_plan); keep stock
    num_splits = max(lay.tile_pieces) if lay.tile_pieces else 1
    split = num_splits > 1

    # Two staged copies (int32 metadata + work list, int64 packed items + gather counts); each array starts on a
    # 128-byte boundary, as the library's separately allocated plan buffers do.
    qo_cum = [0]
    for q in seg_q_packed:
        qo_cum.append(qo_cum[-1] + q)
    kv_cum = [0]
    for k in segs.kv:
        kv_cum.append(kv_cum[-1] + k)
    pg_cum = [0]
    for p in segs.pages:
        pg_cum.append(pg_cum[-1] + p)
    n_items = len(lay.item_tile)
    begins = lay.cta_begin + [n_items]
    ranges = [(begins[c + 1] << 32) | begins[c] for c in range(num_ctas)]
    items = [((u % heads) << 16) | (u // heads) for u in lay.item_tile]  # qo tile 0 | head | segment
    i32_parts = [qo_cum, kv_cum, segs.off, segs.kv, pg_cum, seg_q_packed]
    i64_parts = [ranges, items, segs.shifts, segs.pages]
    if split:
        seg_splits = [max(lay.tile_pieces[s * heads : (s + 1) * heads]) for s in range(nseg)]
        i32_parts += [lay.item_kb, lay.item_ke, lay.item_sub]
        i64_parts += [seg_splits, seg_q_packed]
    i32_views = _stage(i32_parts, torch.int32, device, 32, async_tensor_h2d)
    i64_views = _stage(i64_parts, torch.int64, device, 16, async_tensor_h2d)
    qo_segment_offsets, kv_segment_offsets, qo_offset, kv_segment_lens, kv_page_indptr, qo_segment_lens = i32_views[:6]
    packed_work_range, packed_work_info, shifts, pages = i64_views[:4]
    gather = torch.arange(segs.num_gathered, device=device)
    gather += torch.repeat_interleave(shifts, pages, output_size=segs.num_gathered)
    plan_kv_indices = kv_indices.index_select(0, gather)

    if split:
        kv_tile_begin, kv_tile_end, kv_split_idx = i32_views[6:9]
        seg_split_dev, seg_q_dev = i64_views[4:6]
        # Per packed query row: its segment's piece count (read only by the split-KV reduction, which an
        # OnlyScore launch never runs; kept so the plan is complete).
        num_kv_splits_per_row = torch.repeat_interleave(seg_split_dev, seg_q_dev, output_size=total_q_packed).to(
            torch.int32
        )
        # The epilogue stores one LSE per (split, packed row, head) even in OnlyScore mode.
        workspace_lse = torch.empty(num_splits * total_q_packed * heads, dtype=torch.float32, device=device)
        ws_o = workspace_o if workspace_o is not None else _dummy_workspace_o(device)
    else:
        kv_tile_begin = kv_tile_end = kv_split_idx = num_kv_splits_per_row = workspace_lse = ws_o = None
        num_splits = 1
    cute_workspace = api._alloc_workspace_buf(
        api._BuffTag.fmha_sm100_cutlass_workspace, 32 * 1024 * 1024, device, torch.uint8
    )
    plan = api._make_plan_info(
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        kv_tile_begin_indices=kv_tile_begin,
        kv_tile_end_indices=kv_tile_end,
        kv_split_indices=kv_split_idx,
        num_kv_splits=num_splits,
        workspace_o=ws_o,
        workspace_lse=workspace_lse,
        max_qo_len=max(seg_q_packed),
        predicted_speedup=1.0,
        num_kv_splits_per_row=num_kv_splits_per_row,
        qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets,
        kv_page_indptr=kv_page_indptr,
        max_k_tiles=max_k_tiles,
        qo_segment_lens=qo_segment_lens,
        kv_segment_lens=kv_segment_lens,
        qo_offset=qo_offset,
        pack_factor=pf,
        orig_num_qo_heads=num_index_heads,
        qo_len_uniform=min(segs.qo) == max(segs.qo),
        cute_workspace_buffer=cute_workspace,
    )
    # Marks the plan as o12a's (tests, debugging); _fmha_sm100 reads only the keys above.
    plan["o12a_layout"] = (lay.num_ctas_used, num_splits, pf)
    batch = nseg
    return (False, 0, batch, plan, None), plan_kv_indices


__all__ = [
    "Config",
    "Decision",
    "Layout",
    "Segments",
    "build_plan",
    "config",
    "decide",
    "library_api",
    "pack_factor",
    "reload_config",
    "segment_rows",
    "stock_lower_bound",
    "wrap_layout",
]
