# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staged parity diagnostics: MSA vs Triton MiniMax-M3 sparse attention.

Localizes accuracy or index-out-of-bounds regressions on the
``use_msa=True`` path (the Triton ``use_msa=False`` path is the oracle) by
splitting the sparse-attention pipeline into independently checkable stages,
so the first failing stage points at the culprit:

    Stage 0  selection-semantics spec ...... pure torch, runs anywhere
    Stage 1a kv_indices page table ......... pure CPU, no GPU/MSA
    Stage 1  index-branch scoring .......... proxy per-index-head top-k parity
    Stage 2  block selection ............... union (Triton) vs amax (MSA)
    Stage 3  main attention ................ sparse GQA == dense when all
                                             blocks are selected

Stages 1-3 need an SM100 GPU with the external ``fmha_sm100`` package and are
skipped otherwise. Stages 0 and 1a always run.

The MSA path is the current DSA-aligned surface: the page table comes from
``common.build_kv_page_indices``, block selection from ``MsaIndexer``, and the
sparse GQA from ``fmha.msa_sparse_gqa.run_msa_sparse_gqa``.
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Set, Tuple

import pytest
import torch

backend = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.backend")
common = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common")
indexer_mod = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.indexer")
metadata_mod = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata")
msa_gqa = pytest.importorskip(
    "tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa")

MiniMaxM3SparseConfig = metadata_mod.MiniMaxM3SparseConfig
MiniMaxM3SparseAttentionMetadata = metadata_mod.MiniMaxM3SparseAttentionMetadata
MsaIndexer = indexer_mod.MsaIndexer

PAGE_SIZE = 128
HEAD_DIM = 128
TOPK = 16
# (num_kv_heads, num_index_heads); group = num_index_heads // num_kv_heads.
GROUP_CONFIGS = [
    pytest.param(4, 4, id="group1"),
    pytest.param(1, 4, id="group4"),
]

# Selection stages must use sequences with more valid blocks than topk so the
# top-k actually drops blocks. page_size=128, topk=16 -> >2048 tokens.
#   block counts: ceil([2600, 4000, 800] / 128) = [21, 32, 7]
SEL_SEQ_LENS = [2600, 4000, 800]
SEL_EXTEND = [300, 900, 400]  # prefill extend; tail tokens span >16 blocks


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
# Shared synthetic-layer fixtures
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
                    extend_seq_lens: Optional[List[int]],
                    device: torch.device):
    """Contiguous-slot metadata: slot ids are a plain arange, so
    ``req_to_token[b, pos]`` maps onto both the flat-slot cache (Triton) and
    the reshaped paged cache (MSA), with ``page = slot // PAGE_SIZE``.
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


def _lens_from_meta(meta):
    """Return (qo_lens_cpu, kv_lens_cpu, qo_offset_cpu) int32 for the MSA API."""
    kv_lens_cpu = meta.seq_lens_cpu.to(torch.int32)
    if meta.is_prefill:
        qo_lens_cpu = torch.tensor(meta.extend_seq_lens_cpu, dtype=torch.int32)
    else:
        qo_lens_cpu = torch.ones(kv_lens_cpu.shape[0], dtype=torch.int32)
    qo_offset_cpu = kv_lens_cpu - qo_lens_cpu
    return qo_lens_cpu, kv_lens_cpu, qo_offset_cpu


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


# ---- pure-torch references -------------------------------------------------


def _torch_index_block_scores(idx_q, idx_k_flat, meta, config):
    """Per-(index_head, q, block) max score. ``idx_q`` is
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
    """Exact dense GQA reference (causal for prefill, full for decode)."""
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
    """``build_kv_page_indices`` must emit *global* page ids.

    ``kv_indices`` is the flattened per-request page table into the paged K/V
    cache; the page ids must index the global cache, not a per-request
    [0, max_blocks_per_seq) range. With a contiguous slot layout, request
    ``b``'s pages are ``[b*max_page + i for i in range(num_pages_b)]``. A stale
    per-request clamp collapses every request after the first, corrupting the
    page table (wrong K/V read; and out-of-bounds gathers in the extreme).
    Runs on CPU with no MSA dependency.
    """
    device = torch.device("cpu")
    seq_lens = [2600, 4000, 800]  # blocks = [21, 32, 7]
    extend = seq_lens if is_prefill else None
    meta = _build_metadata(is_prefill=is_prefill, seq_lens=seq_lens,
                           extend_seq_lens=extend, device=device)
    _, kv_lens_cpu, _ = _lens_from_meta(meta)
    kv_indices = common.build_kv_page_indices(meta.req_to_token, meta.slot_ids,
                                              kv_lens_cpu, PAGE_SIZE)

    max_page = int(meta.req_to_token.shape[1]) // PAGE_SIZE
    expected = []
    for b, s in enumerate(seq_lens):
        num_pages = (s + PAGE_SIZE - 1) // PAGE_SIZE
        expected.extend(b * max_page + i for i in range(num_pages))

    got = kv_indices.tolist()
    assert got == expected, (
        "kv_indices page table is corrupted. "
        f"got[:40]={got[:40]} expected[:40]={expected[:40]}. Page ids must be "
        "global cache block ids; a per-request clamp collapses requests whose "
        "pages exceed max_page-1.")


# ===========================================================================
# Stage 1: index-branch scoring parity (per-index-head top-k)
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_index_scoring_perhead_topk_parity(is_prefill, num_kv_heads,
                                           num_index_heads):
    """MSA proxy vs torch: same per-INDEX-HEAD top-k blocks (before reduction).

    Isolates the index scoring path (HND cache permute, kv_indices page
    mapping, causal offset, sm_scale, matmul) from the group reduction.
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

    ref_scores = _torch_index_block_scores(idx_q, caches["idx_k_flat"], meta,
                                           config)
    ref_sets = _perhead_topk_sets(ref_scores, TOPK)

    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = _lens_from_meta(meta)
    kv_indices = common.build_kv_page_indices(meta.req_to_token, meta.slot_ids,
                                              kv_lens_cpu, PAGE_SIZE)
    idx_k_msa = common.cache_view_to_msa_paged(caches["idx_k_paged4d"])
    max_score = indexer_mod._proxy_max_score(
        idx_q, idx_k_msa, qo_lens_cpu=qo_lens_cpu, kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu, kv_indices=kv_indices,
        sm_scale=config.sparse_index_dim**-0.5, causal=True)
    # max_score is [num_index_heads, max_k_tiles, total_q] -> [H, total_q, blk]
    msa_scores = max_score.permute(0, 2, 1).float().contiguous()
    n_blocks = ref_scores.shape[-1]
    msa_scores = msa_scores[:, :, :n_blocks]
    msa_sets = _perhead_topk_sets(msa_scores, TOPK)

    mism = [(key, sorted(ref_sets[key]), sorted(msa_sets.get(key, set())))
            for key in ref_sets if ref_sets[key] != msa_sets.get(key, set())]
    for key, r, m in mism[:8]:
        print(f"[stage1] (q,idx_head)={key} torch={r} msa={m} "
              f"only_torch={sorted(set(r) - set(m))} "
              f"only_msa={sorted(set(m) - set(r))}")
    assert not mism, (
        f"Stage 1 FAIL: MSA proxy per-index-head top-k differs from torch for "
        f"{len(mism)}/{len(ref_sets)} (q, idx_head) pairs -> the index SCORING "
        f"path (not the reduction) is the discrepancy. See prints.")


# ===========================================================================
# Stage 2: block-selection parity (union vs amax reduction)
# ===========================================================================


def _msa_select_blocks(meta, config, idx_q, idx_k_paged4d):
    """Run the current MSA indexer selection -> [total_q, num_kv_heads, topk]."""
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = _lens_from_meta(meta)
    kv_indices = common.build_kv_page_indices(meta.req_to_token, meta.slot_ids,
                                              kv_lens_cpu, PAGE_SIZE)
    idx_k_cache = idx_k_paged4d  # [num_pages, page_size, 1, di]; indexer permutes
    return MsaIndexer(config).select_blocks(
        idx_q, idx_k_cache, idx_sm_scale=config.sparse_index_dim**-0.5,
        qo_lens_cpu=qo_lens_cpu, kv_lens_cpu=kv_lens_cpu,
        qo_offset_cpu=qo_offset_cpu, kv_indices=kv_indices)


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_block_selection_parity(is_prefill, num_kv_heads, num_index_heads):
    """MSA vs Triton final selected blocks, with per-query local/init forcing.

      * group==1 (no index-head reduction): MSA must match Triton EXACTLY.
      * group>1: MSA's amax-then-top-k is a strict SUBSET of Triton's
        per-index-head-top-k-then-union (a known TP artifact); assert only the
        subset invariant (MSA never picks a block Triton did not).
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

    kv_block_indexes = _msa_select_blocks(meta, config, idx_q,
                                          caches["idx_k_paged4d"])
    msa_sets = _msa_indices_to_sets(kv_block_indexes)

    mism = [(key, sorted(triton_sets[key]), sorted(msa_sets.get(key, set())))
            for key in triton_sets
            if triton_sets[key] != msa_sets.get(key, set())]
    group = num_index_heads // num_kv_heads
    for key, t, m in mism[:8]:
        print(f"[stage2] group={group} (q,kv)={key} triton={t} msa={m} "
              f"dropped_by_msa={sorted(set(t) - set(m))} "
              f"extra_in_msa={sorted(set(m) - set(t))}")

    if group == 1:
        assert not mism, (
            f"Stage 2 FAIL (group=1): MSA differs from Triton for "
            f"{len(mism)}/{len(triton_sets)} (q, kv_head) pairs. With no "
            "index-head reduction, MSA must match the reference exactly.")
    else:
        extra = [(key, sorted(set(m) - set(t)))
                 for key, t, m in mism if set(m) - set(t)]
        assert not extra, (
            f"Stage 2 (group={group}): MSA selected blocks OUTSIDE Triton's "
            f"union for {len(extra)} pairs, e.g. {extra[:4]}. amax should only "
            "under-select vs the union; extra blocks indicate a real bug.")


# ===========================================================================
# Stage 3: main-attention dense-equivalence (selection factored out)
# ===========================================================================


@requires_msa
@pytest.mark.parametrize("is_prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("num_kv_heads,num_index_heads", GROUP_CONFIGS)
def test_main_attention_dense_equivalence(is_prefill, num_kv_heads,
                                          num_index_heads):
    """With ALL blocks selected, the sparse GQA must equal dense attention.

    Feeds an identical (dense) block selection to the MSA sparse GQA kernel and
    the Triton masked GQA, and compares each against an exact torch dense
    reference. Isolates the main-attention math, the HND cache permute, and
    causal masking from the indexer.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    # Short sequences so every request fits in <= TOPK blocks.
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
    # _sparse_gqa_masked returns a flattened [total_q, num_q_heads * head_dim];
    # reshape to [total_q, num_q_heads, head_dim] to match the references.
    o_triton = backend._sparse_gqa_masked(
        q, k_padded, v_padded, block_mask, meta.seq_lens, q_batch_row,
        q_positions, config=config, max_k=max_k, sm_scale=config.head_dim**-0.5,
        causal=is_prefill).view(total_q, Hq, config.head_dim)
    triton_rel = ((o_triton.float() - ref.float()).abs().max() /
                  (ref.float().abs().max() + 1e-6)).item()

    # MSA sparse GQA with all valid blocks listed in kv_block_indexes (== dense).
    qo_lens_cpu, kv_lens_cpu, qo_offset_cpu = _lens_from_meta(meta)
    kv_indices = common.build_kv_page_indices(meta.req_to_token, meta.slot_ids,
                                              kv_lens_cpu, PAGE_SIZE)
    k_paged = common.cache_view_to_msa_paged(caches["k_paged4d"])
    v_paged = common.cache_view_to_msa_paged(caches["v_paged4d"])
    # The MSA sparse prefill kernel only accepts kv_block_num in {4,8,16,32};
    # production always emits topk=16, but this synthetic "all blocks" list is
    # narrower, so round the width up to a supported value and -1-pad the tail.
    block_width = next(v for v in (4, 8, 16, 32) if v >= n_blocks)
    seq_lens_i = meta.seq_lens.to(torch.int64)
    kv_block_indexes = torch.full((total_q, num_kv_heads, block_width), -1,
                                  dtype=torch.int32, device=device)
    for i in range(total_q):
        b = int(q_batch_row[i])
        s = int(seq_lens_i[b])
        if is_prefill:
            s = min(s, int(q_positions[i]) + 1)
        nblk = (s + PAGE_SIZE - 1) // PAGE_SIZE
        rng = torch.arange(nblk, dtype=torch.int32, device=device)
        kv_block_indexes[i, :, :nblk] = rng
    o_msa = msa_gqa.run_msa_sparse_gqa(
        q, k_paged, v_paged, kv_block_indexes, qo_lens_cpu=qo_lens_cpu,
        kv_lens_cpu=kv_lens_cpu, qo_offset_cpu=qo_offset_cpu,
        kv_indices=kv_indices, sm_scale=config.head_dim**-0.5,
        causal=is_prefill, head_dim=config.head_dim).view(total_q, Hq,
                                                          config.head_dim)
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
