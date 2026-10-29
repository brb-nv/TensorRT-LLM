# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ancestral sampling of a DSpark Markov block, one fused draw per position.

Position k is drawn from ``q_k = softmax((base_k + bias(d_{k-1})) / T)`` by Gumbel-max: the token is the argmax of
``(base_k + bias) / T + g``, with ``g`` Gumbel noise from a Philox stream keyed by the request's draft
``(seed, offset)`` of position k, so every TP rank and every graph replay draws the same token. The draw kernel also
writes the float32 biased logits and per-chunk softmax statistics; after the block, one kernel writes every ``q_k``
into the proposal buffer.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

_BLOCK = 2048
# Counter word of these draws; the samplers' curand counters never reach it (their offsets stay below 2**63).
_PHILOX_TAG = 0x9E3779B9


@triton.jit
def _draw_lane(
    logits_ptr,
    bias_ptr,
    out_ptr,
    cols,
    vocab,
    is_padding,
    inv_t,
    bits,
    best_key,
    best_index,
    run_max,
    run_sum,
):
    mask = cols < vocab
    base = tl.load(logits_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # A CUDA-graph padding row holds uninitialized logits: it draws from the bias alone.
    base = tl.where(is_padding, 0.0, base)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    logits = base + bias.to(tl.float32)
    tl.store(out_ptr + cols, logits, mask=mask)
    # Rounded product (no FMA contraction), so the proposal kernel recomputes the same values bit for bit.
    scaled = tl.where(mask, libdevice.mul_rn(logits, inv_t), float("-inf"))
    # Gumbel noise -log(E), E = -log(1 - u) ~ Exp(1), from a uniform u fine-grained near 0.
    uniform = tl.minimum((bits.to(tl.float32) + 0.5) * 2.3283064365386963e-10, 0.99999994)
    key = scaled - tl.log(-libdevice.log1p(-uniform))
    lane_key = tl.max(key, 0)
    lane_index = tl.min(cols, 0) + tl.argmax(key, 0)
    best_index = tl.where(lane_key > best_key, lane_index, best_index)
    best_key = tl.maximum(best_key, lane_key)
    new_max = tl.maximum(run_max, tl.max(scaled, 0))
    run_sum = run_sum * tl.exp(run_max - new_max) + tl.sum(tl.exp(scaled - new_max), 0)
    return best_key, best_index, new_max, run_sum


@triton.jit(do_not_specialize=["pos"])
def _draw_kernel(
    logits_ptr,
    out_ptr,
    bias_ptr,
    temp_ptr,
    seed_ptr,
    offset_ptr,
    slot_ptr,
    key_ptr,
    index_ptr,
    max_ptr,
    sum_ptr,
    pos,
    num_pos,
    offset_step,
    padding_slot,
    vocab,
    num_chunks,
    BLOCK: tl.constexpr,
    TAG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    QUARTER: tl.constexpr = BLOCK // 4
    inv_t = 1.0 / tl.load(temp_ptr + row)
    is_padding = tl.load(slot_ptr + row) == padding_slot
    offset = tl.load(offset_ptr + row) + pos * offset_step
    lanes = tl.arange(0, QUARTER)
    counter = (chunk * QUARTER + lanes).to(tl.uint32)
    zero = tl.zeros_like(counter)
    # One Philox call gives four lanes' uniforms: (counter i, output j) <-> column chunk * BLOCK + j * QUARTER + i.
    r0, r1, r2, r3 = tl.philox(
        tl.load(seed_ptr + row),
        counter,
        tl.full([QUARTER], TAG, tl.uint32),
        zero + (offset & 0xFFFFFFFF).to(tl.uint32),
        zero + ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32),
    )
    row_start = (row * num_pos + pos) * vocab
    bias_row = bias_ptr + row * vocab
    cols = chunk * BLOCK + lanes
    best_key = tl.full([], float("-inf"), tl.float32)
    best_index = tl.full([], 0, tl.int32)
    run_max = tl.full([], float("-inf"), tl.float32)
    run_sum = tl.full([], 0.0, tl.float32)
    for j in tl.static_range(4):
        bits = r0 if j == 0 else (r1 if j == 1 else (r2 if j == 2 else r3))
        best_key, best_index, run_max, run_sum = _draw_lane(
            logits_ptr + row_start, bias_row, out_ptr + row_start, cols + j * QUARTER, vocab, is_padding,
            inv_t, bits,
            best_key, best_index, run_max, run_sum,
        )  # fmt: skip
    tl.store(key_ptr + row * num_chunks + chunk, best_key)
    tl.store(index_ptr + row * num_chunks + chunk, best_index)
    stat = (row * num_pos + pos) * num_chunks + chunk
    tl.store(max_ptr + stat, run_max)
    tl.store(sum_ptr + stat, run_sum)


@triton.jit(do_not_specialize=["pos"])
def _pick_kernel(
    key_ptr,
    index_ptr,
    token_ptr,
    pos,
    num_pos,
    w1_ptr,
    w1_stride,
    emb_ptr,
    rank,
    vocab,
    num_chunks,
    FROM_DRAW: tl.constexpr,
    CHUNKS: tl.constexpr,
    RANK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if FROM_DRAW:
        chunks = tl.arange(0, CHUNKS)
        keys = tl.load(
            key_ptr + row * num_chunks + chunks, mask=chunks < num_chunks, other=float("-inf")
        )
        token = tl.load(index_ptr + row * num_chunks + tl.argmax(keys, 0)).to(tl.int64)
        tl.store(token_ptr + row * num_pos + pos, token)
        valid = token >= 0
    else:
        # The anchor: an out-of-vocab one (rejection pads with -1) contributes no bias.
        token = tl.load(token_ptr + row).to(tl.int64)
        valid = (token >= 0) & (token < vocab)
        token = tl.where(valid, token, 0)
    ranks = tl.arange(0, RANK)
    emb = tl.load(w1_ptr + token * w1_stride + ranks, mask=(ranks < rank) & valid, other=0.0)
    tl.store(emb_ptr + row * rank + ranks, emb, mask=ranks < rank)


@triton.jit
def _proposal_kernel(
    logits_ptr,
    temp_ptr,
    max_ptr,
    sum_ptr,
    slot_ptr,
    probs_ptr,
    probs_slot_stride,
    probs_pos_stride,
    vocab,
    num_chunks,
    num_pos,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_pos = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    row = row_pos // num_pos
    inv_t = 1.0 / tl.load(temp_ptr + row)
    chunks = tl.arange(0, CHUNKS)
    run_max = tl.load(
        max_ptr + row_pos * num_chunks + chunks, mask=chunks < num_chunks, other=float("-inf")
    )
    run_sum = tl.load(sum_ptr + row_pos * num_chunks + chunks, mask=chunks < num_chunks, other=0.0)
    row_max = tl.max(run_max, 0)
    total = tl.sum(run_sum * tl.exp(run_max - row_max), 0)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab
    logits = tl.load(logits_ptr + row_pos * vocab + cols, mask=mask, other=0.0)
    probs = tl.exp(libdevice.mul_rn(logits, inv_t) - row_max) / total
    dst = tl.load(slot_ptr + row) * probs_slot_stride + (row_pos % num_pos) * probs_pos_stride
    tl.store(probs_ptr + dst + cols, probs, mask=mask)


def sample_markov_block(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    offset_step: int,
    draft_probs,
    slot_ids: torch.Tensor,
    padding_slot: int,
) -> torch.Tensor:
    """Draw the ``[num_gens, K]`` int32 block from contiguous ``[num_gens, K, vocab]`` base ``logits`` (read only).
    Position k draws at Philox offset ``offsets + k * offset_step``;
    ``q_k`` goes to ``draft_probs[slot_ids, k, :vocab]`` unless ``draft_probs`` is None. No host synchronization."""
    num_gens, num_pos, vocab = logits.shape
    rank = markov_w1.shape[1]
    num_chunks = triton.cdiv(vocab, _BLOCK)
    chunks_p2 = triton.next_power_of_2(num_chunks)
    rank_p2 = triton.next_power_of_2(rank)
    device = logits.device
    tokens = torch.empty(num_gens, num_pos, dtype=torch.int32, device=device)
    emb = torch.empty(num_gens, rank, dtype=markov_w1.dtype, device=device)
    keys = torch.empty(num_gens, num_chunks, dtype=torch.float32, device=device)
    indices = torch.empty(num_gens, num_chunks, dtype=torch.int32, device=device)
    run_max = torch.empty(num_gens, num_pos, num_chunks, dtype=torch.float32, device=device)
    run_sum = torch.empty_like(run_max)
    # The biased logits the draws scale, kept in float32 for the proposal kernel to recompute exactly.
    biased = torch.empty(num_gens, num_pos, vocab, dtype=torch.float32, device=device)
    _pick_kernel[(num_gens,)](
        keys, indices, anchors.contiguous(), 0, num_pos, markov_w1, markov_w1.stride(0), emb, rank, vocab,
        num_chunks, FROM_DRAW=False, CHUNKS=chunks_p2, RANK=rank_p2,
    )  # fmt: skip
    for k in range(num_pos):
        bias = F.linear(emb, markov_w2)
        _draw_kernel[(num_gens, num_chunks)](
            logits, biased, bias, temperatures, seeds, offsets, slot_ids, keys, indices, run_max, run_sum,
            k, num_pos, offset_step, padding_slot, vocab, num_chunks, BLOCK=_BLOCK, TAG=_PHILOX_TAG,
        )  # fmt: skip
        _pick_kernel[(num_gens,)](
            keys, indices, tokens, k, num_pos, markov_w1, markov_w1.stride(0), emb, rank, vocab,
            num_chunks, FROM_DRAW=True, CHUNKS=chunks_p2, RANK=rank_p2,
        )  # fmt: skip
    if draft_probs is not None:
        _proposal_kernel[(num_gens * num_pos, num_chunks)](
            biased, temperatures, run_max, run_sum, slot_ids, draft_probs, draft_probs.stride(0),
            draft_probs.stride(1), vocab, num_chunks, num_pos, CHUNKS=chunks_p2, BLOCK=_BLOCK,
        )  # fmt: skip
    return tokens
