# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staged parity diagnostics: MSA vs Triton MiniMax-M3 sparse attention.

Goal: localize the accuracy regression on the ``use_msa=True`` path of
``TestMiniMaxM3MXFP8::test_mxfp8`` (the Triton ``use_msa=False`` path
passes). Rather than assume the indexer is at fault, the tests split the
sparse-attention pipeline into independently-checkable stages so the
*first stage that fails* points at the culprit:

    Stage 0  selection-semantics spec ...... pure-torch, runs anywhere
    Stage 1  index-branch scoring .......... per-index-head top-k parity
                                             (isolates cache-layout permute,
                                              kv_indices page mapping, causal
                                              offset, scale, matmul)
    Stage 2  block selection ............... union vs amax reduction
                                             (isolates the index-head ->
                                              kv-head reduction)
    Stage 3  main attention ................ dense-equivalence of the sparse
                                             GQA kernel when *all* blocks are
                                             selected (isolates the FMHA /
                                             GQA math, HND cache permute, and
                                             causal masking from selection)
    Stage 4  full layer .................... end-to-end MSA vs Triton output

Each stage runs for both a ``group == 1`` config (num_index_heads ==
num_kv_heads, the single-GPU / no-reduction case) and a ``group == 4``
config (the TP-sharded case, where the index projection is replicated so
each rank sees more index heads than KV heads). Comparing the two isolates
divergences that only appear under grouping.

Stages 1-4 need an SM100 GPU with the external ``fmha_sm100`` package and
are skipped otherwise. Stage 0 always runs.
"""

from __future__ import annotations

import importlib
import math
from typing import Dict, List, Set, Tuple

import pytest
import torch

sparse_minimax_m3 = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3")
backend = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.backend")
msa_backend = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend")
metadata_mod = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata")

MiniMaxM3SparseConfig = metadata_mod.MiniMaxM3SparseConfig
MiniMaxM3SparseAttentionMetadata = metadata_mod.MiniMaxM3SparseAttentionMetadata

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK = 16
# (num_kv_heads, num_index_heads) -> group = num_index_heads // num_kv_heads
GROUP_CONFIGS = [
    pytest.param(4, 4, id="group1"),
    pytest.param(1, 4, id="group4"),
]

# Selection stages (1/2/2b/4) MUST use sequences with more valid blocks than
# ``topk`` so the top-k actually drops blocks; otherwise both selectors return
# "all valid blocks" and the amax-vs-union reduction is never exercised. With
# page_size=128 and topk=16, that means >2048 tokens. Mixed lengths also keep
# the per-query ``num_valid_pages`` / local-block path under test.
#   block counts: ceil([2600, 4000, 800] / 128) = [21, 32, 7]
SEL_SEQ_LENS = [2600, 4000, 800]
SEL_EXTEND = [300, 900, 400]  # prefill extend (<= seq_len); tail tokens span >16 blocks


def _msa_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major != 10:  # SM100 family
        return False
    try:
        importlib.import_module("fmha_sm100")
        return True
    except ImportError:
        return False


requires_msa = pytest.mark.skipif(
    not _msa_available(), reason="MSA fmha_sm100 + SM100 GPU required")


# ===========================================================================
# Stage 0: pure-torch selection-semantics spec (runs anywhere)
# ===========================================================================


def _union_topk_selection(block_scores: torch.Tensor, *, num_kv_heads: int,
                          topk: int) -> List[List[Set[int]]]:
    """Triton-equivalent: per-index-head top-k, OR-union across the group."""
    num_index_heads, total_q, n_blocks = block_scores.shape
    group = num_index_heads // num_kv_heads
    k = min(topk, n_blocks)
    vals, idx = block_scores.topk(k=k, dim=-1)
    out: List[List[Set[int]]] = []
    for h_kv in range(num_kv_heads):
        per_q: List[Set[int]] = []
        for q in range(total_q):
            sel: Set[int] = set()
            for g in range(group):
                h = h_kv * group + g
                sel |= {
                    int(idx[h, q, j])
                    for j in range(k) if vals[h, q, j] != float("-inf")
                }
            per_q.append(sel)
        out.append(per_q)
    return out


def _amax_topk_selection(block_scores: torch.Tensor, *, num_kv_heads: int,
                         topk: int) -> List[List[Set[int]]]:
    """MSA-equivalent: amax across the group first, then a single top-k."""
    num_index_heads, total_q, n_blocks = block_scores.shape
    group = num_index_heads // num_kv_heads
    reduced = block_scores.view(num_kv_heads, group, total_q,
                                n_blocks).amax(dim=1)
    k = min(topk, n_blocks)
    vals, idx = reduced.topk(k=k, dim=-1)
    out: List[List[Set[int]]] = []
    for h_kv in range(num_kv_heads):
        out.append([{
            int(idx[h_kv, q, j])
            for j in range(k) if vals[h_kv, q, j] != float("-inf")
        } for q in range(total_q)])
    return out


@pytest.mark.parametrize("group", [1, 2, 4])
def test_selection_semantics_spec(group: int):
    """union-topk == amax-topk iff group == 1; they diverge for group > 1."""
    torch.manual_seed(0)
    num_kv_heads = 2
    num_index_heads = num_kv_heads * group
    total_q, n_blocks, topk = 8, 12, 4
    scores = torch.randn(num_index_heads, total_q, n_blocks)

    union = _union_topk_selection(scores, num_kv_heads=num_kv_heads, topk=topk)
    amax = _amax_topk_selection(scores, num_kv_heads=num_kv_heads, topk=topk)

    if group == 1:
        for h in range(num_kv_heads):
            for q in range(total_q):
                assert union[h][q] == amax[h][q]
    else:
        assert any(union[h][q] != amax[h][q] for h in range(num_kv_heads)
                   for q in range(total_q)), (
                       "expected union/amax divergence for group>1; reseed if "
                       "this trips (degenerate scores).")


# ===========================================================================
# Shared synthetic-layer fixtures for live (SM100) stages
# ===========================================================================


def _make_config(*, num_kv_heads: int, num_index_heads: int, topk: int = TOPK,
                 g: int = 4, init_blocks: int = 0,
                 local_blocks: int = 1) -> "MiniMaxM3SparseConfig":
    return MiniMaxM3SparseConfig(
        num_q_heads=num_kv_heads * g,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        num_index_heads=num_index_heads,
        sparse_index_dim=HEAD_DIM,
        block_size=PAGE_SIZE,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


def _build_metadata(*, is_prefill: bool, seq_lens: List[int],
                    extend_seq_lens: List[int] | None, device: torch.device):
    """Contiguous-slot metadata: slot ids are a plain arange, so
    ``req_to_token[b, pos]`` maps onto both the flat-slot cache (Triton)
    and the reshaped paged cache (MSA), with ``page = slot // PAGE_SIZE``.
    """
    batch = len(seq_lens)
    max_kv_len = max(((s + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
                     for s in seq_lens)
    req_to_token = torch.arange(batch * max_kv_len, dtype=torch.int32,
                                device=device).view(batch, max_kv_len)
    slot_ids = torch.arange(batch, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    if is_prefill:
        assert extend_seq_lens is not None
        prefix = torch.tensor(
            [seq_lens[b] - extend_seq_lens[b] for b in range(batch)],
            dtype=torch.int32, device=device)
        cu = [0]
        for x in extend_seq_lens:
            cu.append(cu[-1] + x)
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=True, req_to_token=req_to_token, slot_ids=slot_ids,
            seq_lens=seq_lens_t, seq_lens_cpu=seq_lens_t.cpu(),
            prefix_lens=prefix,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=device),
            extend_seq_lens_cpu=list(extend_seq_lens))
    else:
        meta = MiniMaxM3SparseAttentionMetadata(
            is_prefill=False, req_to_token=req_to_token, slot_ids=slot_ids,
            seq_lens=seq_lens_t, seq_lens_cpu=seq_lens_t.cpu())
    meta.prepare()
    return meta


def _rand_caches(meta, config, device):
    """Random main K/V + index-K, in both flat-slot and paged views."""
    num_slots = int(meta.req_to_token.numel())
    num_pages = num_slots // PAGE_SIZE
    d, di = config.head_dim, config.sparse_index_dim
    hkv = config.num_kv_heads
    k_flat = torch.randn(num_slots, hkv, d, dtype=torch.bfloat16, device=device)
    v_flat = torch.randn_like(k_flat)
    idx_k_flat = torch.randn(num_slots, 1, di, dtype=torch.bfloat16,
                             device=device)
    return {
        "num_slots": num_slots,
        "num_pages": num_pages,
        "k_flat": k_flat,
        "v_flat": v_flat,
        "idx_k_flat": idx_k_flat,
        "k_paged4d": k_flat.view(num_pages, PAGE_SIZE, hkv, d),
        "v_paged4d": v_flat.view(num_pages, PAGE_SIZE, hkv, d),
        "idx_k_paged4d": idx_k_flat.view(num_pages, PAGE_SIZE, 1, di),
    }


def _q_batch_and_positions(meta, device):
    """Return (q_batch_row, q_positions) int64 for prefill/decode."""
    if meta.is_prefill:
        return (meta.q_batch_row.to(torch.int64),
                meta.q_positions.to(torch.int64))
    batch = int(meta.slot_ids.shape[0])
    return torch.arange(batch, device=device, dtype=torch.int64), None


# ---- pure-torch references (ground truth for main attention) --------------


def _torch_index_block_scores(idx_q, idx_k_flat, meta, config):
    """Per-(q, index_head, block) max score, pure torch. ``idx_q`` is
    ``[total_q, num_index_heads, dim]``. Returns
    ``[num_index_heads, total_q, n_blocks]`` with -inf on invalid blocks."""
    device = idx_q.device
    total_q, H, di = idx_q.shape
    q_batch_row, q_positions = _q_batch_and_positions(meta, device)
    scale = di**-0.5
    seq_lens = meta.seq_lens.to(torch.int64)
    max_k = int(meta.max_seqlen_k)
    n_blocks = (max_k + config.block_size - 1) // config.block_size
    out = torch.full((H, total_q, n_blocks), float("-inf"), device=device)
    idx_kf = idx_k_flat.squeeze(1).to(torch.float32)  # [num_slots, di]
    for i in range(total_q):
        b = int(q_batch_row[i])
        s = int(seq_lens[b])
        if meta.is_prefill:
            s = min(s, int(q_positions[i]) + 1)  # causal
        slots = meta.req_to_token[b, :s].to(torch.int64)
        k = idx_kf[slots]  # [s, di]
        qk = (idx_q[i].to(torch.float32) @ k.t()) * scale  # [H, s]
        nblk = (s + config.block_size - 1) // config.block_size
        for blk in range(nblk):
            lo = blk * config.block_size
            hi = min(lo + config.block_size, s)
            out[:, i, blk] = qk[:, lo:hi].amax(dim=-1)
    return out


def _torch_dense_gqa(q, k_flat, v_flat, meta, config):
    """Exact dense GQA reference (causal for prefill, full for decode).

    ``q`` is ``[total_q, num_q_heads, d]``; returns the same shape.
    """
    device = q.device
    total_q, Hq, d = q.shape
    hkv = config.num_kv_heads
    g = Hq // hkv
    scale = d**-0.5
    q_batch_row, q_positions = _q_batch_and_positions(meta, device)
    seq_lens = meta.seq_lens.to(torch.int64)
    out = torch.zeros(total_q, Hq, d, device=device, dtype=torch.float32)
    kf = k_flat.to(torch.float32)
    vf = v_flat.to(torch.float32)
    for i in range(total_q):
        b = int(q_batch_row[i])
        s = int(seq_lens[b])
        if meta.is_prefill:
            s = min(s, int(q_positions[i]) + 1)
        slots = meta.req_to_token[b, :s].to(torch.int64)
        k = kf[slots]  # [s, hkv, d]
        v = vf[slots]
        for h in range(Hq):
            hk = h // g
            qk = (q[i, h].to(torch.float32) @ k[:, hk].t()) * scale  # [s]
            w = torch.softmax(qk, dim=-1)
            out[i, h] = w @ v[:, hk]
    return out.to(q.dtype)


def _msa_indices_to_sets(kv_block_indexes) -> Dict[Tuple[int, int], Set[int]]:
    total_q, num_kv_heads, _ = kv_block_indexes.shape
    return {(q, h): {int(b) for b in kv_block_indexes[q, h].tolist() if b >= 0}
            for q in range(total_q) for h in range(num_kv_heads)}


def _perhead_topk_sets(block_scores, topk) -> Dict[Tuple[int, int], Set[int]]:
    """block_scores [H, total_q, n_blocks] -> per (q, head) top-k block set."""
    H, total_q, n_blocks = block_scores.shape
    k = min(topk, n_blocks)
    vals, idx = block_scores.topk(k=k, dim=-1)
    return {(q, h): {int(idx[h, q, j]) for j in range(k)
                     if vals[h, q, j] != float("-inf")}
            for h in range(H) for q in range(total_q)}


# ===========================================================================
# Stage 1a: kv_indices page-table correctness (pure CPU, no MSA/GPU)
# ===========================================================================


@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
def test_build_kv_indices_global_page_table(is_prefill):
    """``_build_kv_indices_and_lens`` must emit *global* page ids.

    ``kv_indices`` is the flattened per-request page table into the paged
    K/V cache; the page ids must index the global cache, not a per-request
    [0, max_blocks_per_seq) range. With a contiguous slot layout, request
    ``b``'s pages are ``[b*max_page + i for i in range(num_pages_b)]``.

    Runs on CPU with no MSA dependency. Fails against the current
    ``clamp_max(max_page - 1)`` which collapses every request after the
    first onto ``max_page - 1``.
    """
    page_size = PAGE_SIZE
    seq_lens = [2600, 4000, 800]  # blocks = [21, 32, 7]; only 1st fits < max
    extend = seq_lens if is_prefill else None
    device = torch.device("cpu")
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    kv_indices, kv_lens = msa_backend._build_kv_indices_and_lens(meta, page_size)

    max_kv_len = int(meta.req_to_token.shape[1])
    max_page = max_kv_len // page_size
    expected = []
    for b, s in enumerate(seq_lens):
        num_pages = (s + page_size - 1) // page_size
        expected.extend(b * max_page + i for i in range(num_pages))

    got = kv_indices.tolist()
    assert got == expected, (
        "kv_indices page table is corrupted. "
        f"got[:40]={got[:40]} expected[:40]={expected[:40]}. "
        "The clamp_max(max_page-1) in _build_kv_indices_and_lens uses the "
        "per-request page count as an upper bound on GLOBAL page ids, so "
        "every request whose pages exceed max_page-1 (i.e. every request "
        "after the first in a contiguous layout, and most requests in "
        "production where block ids are global) gets its page table "
        "collapsed onto max_page-1.")


# ===========================================================================
# Stage 1: index-branch scoring parity (per-index-head top-k)
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_index_scoring_perhead_topk_parity(is_prefill, num_kv_heads,
                                           num_index_heads):
    """MSA proxy vs torch: same per-INDEX-HEAD top-k blocks (before reduction).

    This isolates the index scoring path (cache HND permute, kv_indices
    page mapping, causal offset, sm_scale, matmul) from the group
    reduction. If this FAILS, the discrepancy is in the proxy FMHA /
    metadata plumbing, not the union-vs-amax reduction.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_lens = SEL_SEQ_LENS
    extend = SEL_EXTEND if is_prefill else None
    config = _make_config(num_kv_heads=num_kv_heads,
                          num_index_heads=num_index_heads)
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    caches = _rand_caches(meta, config, device)
    total_q = (int(meta.q_batch_row.shape[0]) if is_prefill else len(seq_lens))
    idx_q = torch.randn(total_q, num_index_heads, config.sparse_index_dim,
                        dtype=torch.bfloat16, device=device)

    # torch reference per-head block scores + top-k.
    ref_scores = _torch_index_block_scores(idx_q, caches["idx_k_flat"], meta,
                                           config)
    ref_sets = _perhead_topk_sets(ref_scores, TOPK)

    # MSA proxy max_score -> per-head top-k.
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = (
        msa_backend._qo_lens_offsets_from_metadata(meta))
    kv_indices, _ = msa_backend._build_kv_indices_and_lens(meta, PAGE_SIZE)
    idx_k_msa = msa_backend._idx_cache_to_msa_paged(caches["idx_k_paged4d"])
    proxy_cls = msa_backend._select_proxy_fmha_class()
    assert proxy_cls is not None, "no IndexerProxyFmha available"
    max_score = proxy_cls().forward_proxy(
        idx_q, idx_k_msa, qo_lens_cpu=qo_lens_cpu, kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu, kv_indices=kv_indices,
        sm_scale=config.sparse_index_dim**-0.5, causal=is_prefill)
    # max_score is [num_index_heads, max_k_tiles, total_q] -> [H, total_q, blk]
    msa_scores = max_score.permute(0, 2, 1).float().contiguous()
    n_blocks = ref_scores.shape[-1]
    msa_scores = msa_scores[:, :, :n_blocks]
    msa_sets = _perhead_topk_sets(msa_scores, TOPK)

    mism = [(key, sorted(ref_sets[key]), sorted(msa_sets.get(key, set())))
            for key in ref_sets if ref_sets[key] != msa_sets.get(key, set())]
    for key, r, m in mism[:8]:
        print(f"[stage1] (q,idx_head)={key} torch={r} msa={m} "
              f"only_torch={sorted(set(r)-set(m))} only_msa={sorted(set(m)-set(r))}")
    assert not mism, (
        f"Stage 1 FAIL: MSA proxy per-index-head top-k differs from torch "
        f"for {len(mism)}/{len(ref_sets)} (q, idx_head) pairs -> the index "
        f"SCORING path (not the reduction) is the discrepancy. See prints.")


# ===========================================================================
# Stage 1c: sparse_topk_select kernel vs torch.topk (no M3 plumbing at all)
# ===========================================================================


def _rand_max_score(num_heads: int, num_tiles: int, total_q: int, *,
                    valid_tiles: int) -> torch.Tensor:
    """Build a ``[num_heads, num_tiles, total_q]`` fp32 max_score on cuda.

    The first ``valid_tiles`` tiles carry finite, per-column-unique scores
    (jitter added so top-k has no ties); tiles ``[valid_tiles, num_tiles)``
    are ``-inf`` padding, matching how the proxy FMHA fills padding tiles.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    ms = torch.randn(num_heads, num_tiles, total_q, dtype=torch.float32,
                     device=device)
    jitter = (torch.arange(num_tiles, device=device, dtype=torch.float32)
              * 1e-3).view(1, num_tiles, 1)
    ms = ms + jitter
    if valid_tiles < num_tiles:
        ms[:, valid_tiles:, :] = float("-inf")
    return ms.contiguous()


def _compare_topk_kernel_vs_torch(max_score: torch.Tensor, *,
                                  num_valid_pages, topk: int, label: str):
    """Compare ``sparse_topk_select`` to ``torch.topk`` over the tile axis.

    Returns the list of mismatching ``((head, token), torch_set, kernel_set)``.
    Reference sets drop tiles whose score is ``-inf`` (padding) so the
    comparison is well-defined when fewer than ``topk`` tiles are finite.
    """
    import fmha_sm100

    num_heads, num_tiles, total_q = max_score.shape
    k = min(topk, num_tiles)
    _, ref_idx = max_score.topk(k=k, dim=1)  # [H, k, total_q]
    out = fmha_sm100.sparse_topk_select(max_score.contiguous(), topk,
                                        num_valid_pages=num_valid_pages)
    assert tuple(out.shape) == (total_q, num_heads, topk), out.shape

    mism = []
    for h in range(num_heads):
        for q in range(total_q):
            ref_set = {
                int(t) for t in ref_idx[h, :, q].tolist()
                if math.isfinite(float(max_score[h, int(t), q]))
            }
            k_set = {int(x) for x in out[q, h].tolist() if x >= 0}
            if ref_set != k_set:
                mism.append(((h, q), sorted(ref_set), sorted(k_set)))
    for key, r, m in mism[:6]:
        print(f"[{label}] T={num_tiles} Q={total_q} H={num_heads} "
              f"nvp={num_valid_pages} (h,q)={key} torch={r} kernel={m} "
              f"only_torch={sorted(set(r)-set(m))} "
              f"only_kernel={sorted(set(m)-set(r))}")
    return mism


@requires_msa
@pytest.mark.parametrize("num_tiles", [7, 17, 21, 32, 40])
@pytest.mark.parametrize("total_q", [1, 3])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_sparse_topk_select_kernel_vs_torch(num_tiles, total_q, num_heads):
    """Directly compare ``fmha_sm100.sparse_topk_select`` to ``torch.topk``.

    Strips away the entire M3 pipeline (proxy FMHA, metadata, caches,
    force blocks, index-head reduction). Feeds a hand-built ``max_score``
    ``[num_qo_heads, max_k_tiles, total_qo_len]`` and checks the kernel
    selects the same top-16 tile indices per (head, token) as a plain
    torch top-k over the tile axis.

    Stage 2b's group=1 failure showed the kernel returning a head-invariant,
    strided result for the *longest / fully-dense* column while torch.topk
    (validated in Stage 1) did not. This test localizes that to the kernel
    itself vs. its invocation, and finds the triggering shape.
    """
    max_score = _rand_max_score(num_heads, num_tiles, total_q,
                                valid_tiles=num_tiles)
    mism = _compare_topk_kernel_vs_torch(
        max_score, num_valid_pages=num_tiles, topk=TOPK, label="stage1c")
    assert not mism, (
        f"Stage 1c FAIL: sparse_topk_select disagrees with torch.topk for "
        f"{len(mism)}/{num_heads * total_q} (head, token) columns at "
        f"num_tiles={num_tiles}, total_q={total_q}, num_heads={num_heads}. "
        "The kernel (or its invocation contract), not the M3 plumbing, is the "
        "source of the selection discrepancy.")


@requires_msa
@pytest.mark.parametrize("num_tiles,valid_pages", [(32, 21), (40, 17),
                                                   (32, 16), (24, 20)])
@pytest.mark.parametrize("nvp_mode", ["scalar", "none"], ids=["nvp=valid",
                                                              "nvp=None"])
def test_sparse_topk_select_num_valid_pages_contract(num_tiles, valid_pages,
                                                     nvp_mode):
    """Exercise the ``num_valid_pages`` contract of ``sparse_topk_select``.

    ``max_score`` has ``num_tiles`` tiles but only the first ``valid_pages``
    are finite; the rest are ``-inf`` (exactly how the proxy FMHA pads
    padding tiles). Two modes:

      * ``nvp=valid``: pass ``num_valid_pages=valid_pages`` (the OOB-clamp
        path — kernel should map indices in ``[valid_pages, num_tiles)`` to
        ``-1`` and never select them).
      * ``nvp=None``: omit it (kernel uses ``max_k_tiles``); with -inf
        padding the kernel must still avoid the padding tiles.

    Reference is ``torch.topk`` over the tile axis, filtered to finite
    entries. ``valid_pages >= topk`` so the top-16 is unambiguous.
    """
    num_heads, total_q = 4, 3
    max_score = _rand_max_score(num_heads, num_tiles, total_q,
                                valid_tiles=valid_pages)
    nvp = valid_pages if nvp_mode == "scalar" else None
    mism = _compare_topk_kernel_vs_torch(
        max_score, num_valid_pages=nvp, topk=TOPK,
        label=f"stage1c-nvp[{nvp_mode}]")
    assert not mism, (
        f"Stage 1c num_valid_pages FAIL ({nvp_mode}): kernel disagrees with "
        f"torch.topk for {len(mism)}/{num_heads * total_q} columns at "
        f"num_tiles={num_tiles}, valid_pages={valid_pages}. The "
        "num_valid_pages / OOB-clamp contract is the source of the "
        "discrepancy.")


# ===========================================================================
# Stage 2: block-selection parity (union vs amax reduction)
# ===========================================================================


@pytest.mark.xfail(
    strict=False,
    reason="MSA bug #2 (open): _msa_index_proxy_and_topk passes a batch-max "
    "scalar num_valid_pages and force_end_blocks=local_blocks to "
    "sparse_topk_select, so the forced local block is pinned to the global "
    "nvp-1 instead of each query's own last valid block. Fails for any "
    "request shorter than the batch-longest. Flip to a normal assertion once "
    "the per-query force fix lands.")
@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_block_selection_parity(is_prefill, num_kv_heads, num_index_heads):
    """MSA vs Triton final selected blocks (after the index-head reduction).

    If Stage 1 passed but this FAILS for group>1, the discrepancy is the
    reduction: MSA amax-then-topk vs Triton per-head-topk-then-union.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_lens = SEL_SEQ_LENS
    extend = SEL_EXTEND if is_prefill else None
    config = _make_config(num_kv_heads=num_kv_heads,
                          num_index_heads=num_index_heads)
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    caches = _rand_caches(meta, config, device)
    total_q = (int(meta.q_batch_row.shape[0]) if is_prefill else len(seq_lens))
    idx_q = torch.randn(total_q, num_index_heads, config.sparse_index_dim,
                        dtype=torch.bfloat16, device=device)

    # Triton selection (union of per-head top-k).
    idx_k_padded = backend._gather_paged_batched(
        caches["idx_k_flat"], meta.req_to_token, meta.slot_ids,
        int(meta.max_seqlen_k))
    q_batch_row, q_positions = _q_batch_and_positions(meta, device)
    _, block_mask = backend._index_attention_and_select(
        idx_q, idx_k_padded, None, meta.seq_lens, q_batch_row, q_positions,
        config=config, max_k=int(meta.max_seqlen_k), disable_index_value=True,
        idx_sm_scale=config.sparse_index_dim**-0.5, causal=is_prefill)
    triton_sets = {(q, h): set(torch.nonzero(block_mask[h, q]).flatten().tolist())
                   for h in range(num_kv_heads) for q in range(total_q)}

    # MSA selection (amax then top-k).
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = (
        msa_backend._qo_lens_offsets_from_metadata(meta))
    kv_indices, _ = msa_backend._build_kv_indices_and_lens(meta, PAGE_SIZE)
    idx_k_msa = msa_backend._idx_cache_to_msa_paged(caches["idx_k_paged4d"])
    kv_block_indexes = msa_backend._msa_index_proxy_and_topk(
        idx_q, idx_k_msa, qo_lens_cpu=qo_lens_cpu, kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu, kv_indices=kv_indices, config=config,
        idx_sm_scale=config.sparse_index_dim**-0.5, causal=is_prefill,
        init_blocks=config.init_blocks, local_blocks=config.local_blocks)
    msa_sets = _msa_indices_to_sets(kv_block_indexes)

    mism = [(key, sorted(triton_sets[key]), sorted(msa_sets.get(key, set())))
            for key in triton_sets
            if triton_sets[key] != msa_sets.get(key, set())]
    for key, t, m in mism[:8]:
        print(f"[stage2] (q,kv)={key} triton={t} msa={m} "
              f"dropped_by_msa={sorted(set(t)-set(m))}")
    assert not mism, (
        f"Stage 2 FAIL: MSA selected different blocks than Triton for "
        f"{len(mism)}/{len(triton_sets)} (q, kv_head) pairs (group="
        f"{num_index_heads // num_kv_heads}). If Stage 1 passed this is the "
        f"index-head reduction (amax-then-topk vs per-head-topk-union).")


# ===========================================================================
# Stage 2b: selection parity with init/local forcing DISABLED
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_block_selection_parity_no_forcing(is_prefill, num_kv_heads,
                                           num_index_heads):
    """Same as Stage 2 but with init_blocks=local_blocks=0.

    Isolates the ``force_begin_blocks`` / ``force_end_blocks`` path of
    ``sparse_topk_select`` (which uses a *scalar*, batch-wide
    ``num_valid_pages``) from the base top-k. Expectation:

      * group=1 should now PASS -> the pure per-head top-k agrees, and the
        Stage 2 failure was the batch-wide local/init block forcing.
      * If group=1 still FAILS here, the scalar ``num_valid_pages`` OOB
        clamp (also batch-wide) is contributing independently of forcing.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_lens = SEL_SEQ_LENS
    extend = SEL_EXTEND if is_prefill else None
    config = _make_config(num_kv_heads=num_kv_heads,
                          num_index_heads=num_index_heads, init_blocks=0,
                          local_blocks=0)
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    caches = _rand_caches(meta, config, device)
    total_q = (int(meta.q_batch_row.shape[0]) if is_prefill else len(seq_lens))
    idx_q = torch.randn(total_q, num_index_heads, config.sparse_index_dim,
                        dtype=torch.bfloat16, device=device)

    idx_k_padded = backend._gather_paged_batched(
        caches["idx_k_flat"], meta.req_to_token, meta.slot_ids,
        int(meta.max_seqlen_k))
    q_batch_row, q_positions = _q_batch_and_positions(meta, device)
    _, block_mask = backend._index_attention_and_select(
        idx_q, idx_k_padded, None, meta.seq_lens, q_batch_row, q_positions,
        config=config, max_k=int(meta.max_seqlen_k), disable_index_value=True,
        idx_sm_scale=config.sparse_index_dim**-0.5, causal=is_prefill)
    triton_sets = {(q, h): set(torch.nonzero(block_mask[h, q]).flatten().tolist())
                   for h in range(num_kv_heads) for q in range(total_q)}

    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = (
        msa_backend._qo_lens_offsets_from_metadata(meta))
    kv_indices, _ = msa_backend._build_kv_indices_and_lens(meta, PAGE_SIZE)
    idx_k_msa = msa_backend._idx_cache_to_msa_paged(caches["idx_k_paged4d"])
    kv_block_indexes = msa_backend._msa_index_proxy_and_topk(
        idx_q, idx_k_msa, qo_lens_cpu=qo_lens_cpu, kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu, kv_indices=kv_indices, config=config,
        idx_sm_scale=config.sparse_index_dim**-0.5, causal=is_prefill,
        init_blocks=0, local_blocks=0)
    msa_sets = _msa_indices_to_sets(kv_block_indexes)

    mism = [(key, sorted(triton_sets[key]), sorted(msa_sets.get(key, set())))
            for key in triton_sets
            if triton_sets[key] != msa_sets.get(key, set())]
    for key, t, m in mism[:8]:
        print(f"[stage2b] (q,kv)={key} triton={t} msa={m} "
              f"only_triton={sorted(set(t)-set(m))} only_msa={sorted(set(m)-set(t))}")
    group = num_index_heads // num_kv_heads
    if group == 1:
        assert not mism, (
            f"Stage 2b: group=1 with NO forcing still differs for "
            f"{len(mism)}/{len(triton_sets)} pairs. With forcing off and no "
            "reduction, this is either the sparse_topk_select kernel itself "
            "(see Stage 1c) or the scalar num_valid_pages contract; it is NOT "
            "the amax-vs-union reduction.")
    else:
        # group>1: any residual mismatch here is the amax-vs-union reduction
        # (forcing is off), which is a separate, expected divergence.
        print(f"[stage2b] group={group}: {len(mism)}/{len(triton_sets)} "
              "mismatches remain with forcing off (amax-vs-union reduction).")


# ===========================================================================
# Stage 3: main-attention dense-equivalence (selection factored out)
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_main_attention_dense_equivalence(is_prefill, num_kv_heads,
                                          num_index_heads):
    """With ALL blocks selected, the sparse GQA must equal dense attention.

    Feeds an identical (dense) block selection to both the MSA sparse GQA
    kernel and the Triton masked GQA, and compares each against an exact
    torch dense reference. This isolates the main-attention math, the HND
    cache permute, and causal masking from the indexer entirely. If the
    MSA output diverges here, the bug is in the FMHA / cache-layout path,
    NOT the block selection.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Short sequences so every request fits in <= TOPK blocks (=> "all
    # blocks selected" is representable in the fixed kernel budget).
    seq_lens = [130, 256, 200]
    extend = [130, 256, 200] if is_prefill else None
    config = _make_config(num_kv_heads=num_kv_heads,
                          num_index_heads=num_index_heads)
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    caches = _rand_caches(meta, config, device)
    Hq = config.num_q_heads
    total_q = (int(meta.q_batch_row.shape[0]) if is_prefill else len(seq_lens))
    q = torch.randn(total_q, Hq, config.head_dim, dtype=torch.bfloat16,
                    device=device)

    ref = _torch_dense_gqa(q, caches["k_flat"], caches["v_flat"], meta, config)

    max_k = int(meta.max_seqlen_k)
    n_blocks = (max_k + PAGE_SIZE - 1) // PAGE_SIZE

    # Triton sparse GQA with an all-True block mask (== dense).
    k_padded = backend._gather_paged_batched(caches["k_flat"], meta.req_to_token,
                                             meta.slot_ids, max_k)
    v_padded = backend._gather_paged_batched(caches["v_flat"], meta.req_to_token,
                                             meta.slot_ids, max_k)
    q_batch_row, q_positions = _q_batch_and_positions(meta, device)
    block_mask = torch.ones(num_kv_heads, total_q, n_blocks, dtype=torch.bool,
                            device=device)
    o_triton = backend._sparse_gqa_masked(
        q, k_padded, v_padded, block_mask, meta.seq_lens, q_batch_row,
        q_positions, config=config, max_k=max_k, sm_scale=config.head_dim**-0.5,
        causal=is_prefill)
    triton_rel = ((o_triton.float() - ref.float()).abs().max() /
                  (ref.float().abs().max() + 1e-6)).item()

    # MSA sparse GQA with all blocks listed in kv_block_indexes (== dense).
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = (
        msa_backend._qo_lens_offsets_from_metadata(meta))
    kv_indices, _ = msa_backend._build_kv_indices_and_lens(meta, PAGE_SIZE)
    k_paged = msa_backend._cache_view_to_msa_paged(caches["k_paged4d"])
    v_paged = msa_backend._cache_view_to_msa_paged(caches["v_paged4d"])
    # Per-(q, kv) block list = all valid pages of that query's request.
    seq_lens_i = meta.seq_lens.to(torch.int64)
    kv_block_indexes = torch.full((total_q, num_kv_heads, n_blocks), -1,
                                  dtype=torch.int32, device=device)
    for i in range(total_q):
        b = int(q_batch_row[i])
        s = int(seq_lens_i[b])
        if is_prefill:
            s = min(s, int(q_positions[i]) + 1)
        nblk = (s + PAGE_SIZE - 1) // PAGE_SIZE
        rng = torch.arange(nblk, dtype=torch.int32, device=device)
        kv_block_indexes[i, :, :nblk] = rng
    o_msa = msa_backend._msa_sparse_attention(
        q, k_paged, v_paged, kv_block_indexes, qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu, qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices, sm_scale=config.head_dim**-0.5,
        causal=is_prefill)
    msa_rel = ((o_msa.float() - ref.float()).abs().max() /
               (ref.float().abs().max() + 1e-6)).item()

    print(f"[stage3] triton_rel={triton_rel:.4f} msa_rel={msa_rel:.4f}")
    assert triton_rel < 5e-2, (
        f"Stage 3: Triton dense-equiv rel diff {triton_rel:.4f} too high "
        "(reference/harness issue, investigate before trusting MSA result).")
    assert msa_rel < 5e-2, (
        f"Stage 3 FAIL: MSA main attention diverges from dense even with ALL "
        f"blocks selected (rel diff {msa_rel:.4f}). The bug is in the MSA FMHA "
        "/ HND cache permute / causal path, NOT the block selection.")


# ===========================================================================
# Stage 4: full-layer output parity (end-to-end)
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_full_layer_output_parity(is_prefill, num_kv_heads, num_index_heads):
    """End-to-end MSA vs Triton output for one synthetic sparse layer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    seq_lens = SEL_SEQ_LENS
    extend = SEL_EXTEND if is_prefill else None
    config = _make_config(num_kv_heads=num_kv_heads,
                          num_index_heads=num_index_heads)
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    caches = _rand_caches(meta, config, device)
    Hq = config.num_q_heads
    total_q = (int(meta.q_batch_row.shape[0]) if is_prefill else len(seq_lens))
    q = torch.randn(total_q, Hq, config.head_dim, dtype=torch.bfloat16,
                    device=device)
    idx_q = torch.randn(total_q, num_index_heads, config.sparse_index_dim,
                        dtype=torch.bfloat16, device=device)

    if is_prefill:
        _, o_triton = backend.minimax_m3_sparse_prefill(
            q, caches["k_flat"], caches["v_flat"], idx_q, caches["idx_k_flat"],
            None, meta, config, disable_index_value=True)
        o_msa = msa_backend.minimax_m3_msa_sparse_prefill(
            q, caches["k_paged4d"], caches["v_paged4d"], idx_q,
            caches["idx_k_paged4d"], meta, config)
    else:
        _, o_triton = backend.minimax_m3_sparse_decode(
            q, idx_q, caches["k_flat"], caches["v_flat"], caches["idx_k_flat"],
            None, meta, config, disable_index_value=True)
        o_msa = msa_backend.minimax_m3_msa_sparse_decode(
            q, idx_q, caches["k_paged4d"], caches["v_paged4d"],
            caches["idx_k_paged4d"], meta, config)

    diff = (o_triton.float() - o_msa.float()).abs()
    rel = (diff.max() / (o_triton.float().abs().max() + 1e-6)).item()
    group = num_index_heads // num_kv_heads
    print(f"[stage4] group={group} max_abs={diff.max().item():.4f} rel={rel:.4f}")
    if group == 1:
        assert rel < 1e-1, (
            f"Stage 4: group=1 end-to-end output should match closely, "
            f"rel={rel:.4f}. Divergence here with Stages 1-3 passing implies "
            "a wiring bug in the full entry points.")
    else:
        # Measurement only: group>1 divergence is expected if Stage 2 shows
        # the selection differs. The magnitude quantifies the accuracy risk.
        assert math.isfinite(rel)
