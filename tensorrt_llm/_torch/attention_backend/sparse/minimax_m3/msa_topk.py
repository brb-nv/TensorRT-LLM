# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
#
# The bitonic top-k helpers and the streaming per-query block selection are
# adapted from vLLM's MiniMax-M3 lightning-indexer top-k kernel
# (vllm/models/minimax_m3/common/ops/index_topk.py, Apache-2.0). The proxy
# per-block max score is produced by fmha_sm100 upstream, so this kernel only
# performs the init and local forcing, causal masking, and top-k selection.
"""Triton top-k block selection for the MiniMax-M3 MSA sparse indexer.

The indexer computes a per-block max score with fmha_sm100 and reduces it to
KV-head granularity (see msa_indexer). This module selects, for every query
token and KV head, the top-k blocks by score after forcing the init and local
blocks and masking blocks beyond the query's causal extent.

triton_select_blocks_from_maxscore returns [total_q, num_kv_heads, topk] int32
block ids with -1 tail padding. Block ids are returned in score order. The
fmha_sm100 sparse gather builds its k2q reverse index by scattering query
tokens per block id, so their order within a query row does not affect the
result.

The kernel is CUDA-graph safe: block sizes come from the n_blocks shape
constant, and no autotuning runs on the hot path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .common import _INIT_SCORE, _LOCAL_SCORE

# Fills masked-out score positions. It must lose to every real score and to the
# forced init and local sentinels. Declared as a Triton constexpr so the jitted
# kernel can read it as a module global.
_NEG_SCORE = tl.constexpr(-1e30)


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    """Bitonic compare-and-swap that carries an index payload alongside x.

    The score x drives the ordering while ids, the 1-indexed block id, is
    permuted identically, so selected scores stay paired with their block ids.
    """
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    """One bitonic merge stage.

    An order of 2 builds alternating bitonic runs. An order of True or False
    performs a monotonic descending or ascending merge.
    """
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _msa_topk_index_kernel(
    s_ptr,  # per-block scores, viewed [num_kv_heads, total_q, n_blocks]
    nvb_ptr,  # per-query valid block count, [total_q] int32
    ti_ptr,  # output block ids, [total_q, num_kv_heads, topk] int32
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_q,
    stride_s_b,
    stride_ti_q,
    stride_ti_h,
    stride_ti_t,
    INIT_SCORE: tl.constexpr,
    LOCAL_SCORE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    N_DIMS_K: tl.constexpr,
):
    """Per (query, KV head) streaming top-k over the valid blocks.

    Grid is (total_q, num_kv_heads). Each program streams the query's valid
    blocks in BLOCK_SIZE_K chunks, forces the init and local blocks to their
    sentinel scores, and maintains a running bitonic top-k. The first
    BLOCK_SIZE_T entries after the final merge are the selected blocks.
    """
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    nvb = tl.load(nvb_ptr + pid_q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = s_ptr + pid_q * stride_s_q + pid_h * stride_s_h + off_k * stride_s_b

    topk_score = tl.full((BLOCK_SIZE_K,), _NEG_SCORE, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    local_start = tl.maximum(nvb - local_blocks, 0)

    for i in tl.range(0, nvb, BLOCK_SIZE_K):
        blk = i + off_k
        causal_mask = blk < nvb
        init_mask = blk < init_blocks
        local_mask = blk >= local_start
        score = tl.load(s_ptrs, mask=causal_mask, other=_NEG_SCORE).to(tl.float32)
        score = tl.where(score != score, _NEG_SCORE, score)
        s_ptrs = s_ptrs + stride_s_b * BLOCK_SIZE_K
        # Init outranks a real score and local outranks init, so local is applied last.
        score = tl.where(causal_mask & init_mask, INIT_SCORE, score)
        score = tl.where(causal_mask & local_mask, LOCAL_SCORE, score)

        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = tl.where(causal_mask, blk + 1, 0), topk_idx
        for j in tl.static_range(1, N_DIMS_K):
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), j, 2, N_DIMS_K)
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), N_DIMS_K, False, N_DIMS_K
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (1 - left_half_mask)
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (1 - left_half_mask)
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), N_DIMS_K, True, N_DIMS_K
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), N_DIMS_K, True, N_DIMS_K
            )

    # First BLOCK_SIZE_T entries are the top blocks, 1-indexed with 0 for invalid.
    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    sel = tl.sum(
        extract_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    valid_mask = off_t < nvb
    sel = tl.where(valid_mask, sel, -1)

    ti_ptrs = ti_ptr + pid_q * stride_ti_q + pid_h * stride_ti_h + off_t * stride_ti_t
    store_mask = off_t < topk
    tl.store(ti_ptrs, sel.to(ti_ptr.dtype.element_ty), mask=store_mask)


def _select_block_size_k(n_blocks: int, block_size_t: int) -> int:
    """Pick a power-of-two BLOCK_SIZE_K from the n_blocks shape constant.

    Chosen from n_blocks, a tensor shape rather than a host sync, so the launch
    config is fixed within a captured CUDA graph. Capped at 2048; the streaming
    loop covers larger block counts.
    """
    bs_k = min(2048, triton.next_power_of_2(max(n_blocks, 1)))
    # _compare_and_swap reshapes BLOCK_SIZE_K into (rows, BLOCK_SIZE_T); the
    # top-k extraction needs at least two rows, so keep BLOCK_SIZE_K above
    # BLOCK_SIZE_T.
    return max(bs_k, block_size_t * 2)


def triton_select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select per-query top-k blocks from per-KV-head block scores.

    Inputs:
        max_score_kv: [num_kv_heads, n_blocks, total_q] per-block max scores,
            amax-reduced to KV-head granularity.
        n_valid_blocks: [total_q] per-query valid block count.

    Returns [total_q, num_kv_heads, topk] int32 block ids with -1 tail padding.
    Init and local blocks are force-selected; blocks beyond a query's causal
    extent are masked out.
    """
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    out = torch.empty((total_q, num_kv_heads, topk), dtype=torch.int32, device=device)
    if total_q == 0 or num_kv_heads == 0 or topk == 0:
        return out
    if n_blocks == 0:
        out.fill_(-1)
        return out

    # The kernel reads scores as [num_kv_heads, total_q, n_blocks] via strides,
    # so a transpose view without a copy suffices.
    score = max_score_kv.transpose(1, 2)
    nvb = n_valid_blocks.to(device=device, dtype=torch.int32)

    block_size_t = triton.next_power_of_2(topk)
    block_size_k = _select_block_size_k(n_blocks, block_size_t)
    n_dims_k = int(block_size_k).bit_length() - 1
    num_warps = 8 if block_size_k >= 256 else (4 if block_size_k >= 128 else 2)

    grid = (total_q, num_kv_heads)
    _msa_topk_index_kernel[grid](
        score,
        nvb,
        out,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        INIT_SCORE=_INIT_SCORE,
        LOCAL_SCORE=_LOCAL_SCORE,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_T=block_size_t,
        N_DIMS_K=n_dims_k,
        num_warps=num_warps,
    )
    return out


__all__ = ["triton_select_blocks_from_maxscore"]
