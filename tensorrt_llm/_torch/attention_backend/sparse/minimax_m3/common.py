# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared building blocks for the MiniMax-M3 sparse attention backends.

Both the Triton reference and the MSA (fmha_sm100) path share these
backend-neutral pieces: the lowered parameter and per-rank kernel config
bundles, the lazy fmha_sm100 import guard, kernel precondition constants,
block-priority sentinels, paged cache layout adapters, KV-slot writers,
per-query valid-block counting, and torch top-k block selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch

from ..params import SparseParams

# fmha_sm100 ships only head_dim 128 variants and the MiniMax-M3 checkpoint
# selects topk 16. Callers enforce these early so a misconfiguration fails
# with a clear message rather than a cryptic shape error inside the kernel.
MSA_REQUIRED_TOPK = 16
MSA_REQUIRED_HEAD_DIM = 128

# Sentinel scores that force init and local blocks into the top-k regardless
# of their computed score. Init outranks local.
_INIT_SCORE = 1e30
_LOCAL_SCORE = 1e29


@dataclass(frozen=True)
class MiniMaxM3SparseParams(SparseParams):
    """Lowered runtime parameters for the MiniMax-M3 sparse backend."""

    algorithm: Literal["minimax_m3"] = field(init=False, default="minimax_m3")
    num_index_heads: int = 4
    sparse_index_dim: int = 128
    block_size: int = 128
    topk: int = 16
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"
    disable_index_value: bool = True
    # Select the MSA (fmha_sm100) kernels instead of the Triton reference.
    # Requires an SM100 GPU and the fmha_sm100 package.
    use_msa: bool = False

    @property
    def indices_block_size(self) -> int:
        """Block granularity of the selected sparse indices.

        Read by the shared TrtllmAttention forward when publishing the
        sparse prediction. It equals the per-block scoring size.
        """
        return self.block_size


@dataclass(frozen=True)
class MiniMaxM3SparseConfig:
    """Per-rank kernel parameter bundle for MiniMax-M3 sparse attention.

    This is **not** a user-facing config (use
    :class:`tensorrt_llm.llmapi.llm_args.MiniMaxM3SparseAttentionConfig`
    for that). It is the layer-invariant, post-TP-shard parameter bundle
    that backend kernels and reference helpers consume. The user knobs
    come from :class:`MiniMaxM3SparseParams`; ``num_q_heads`` /
    ``num_kv_heads`` / ``head_dim`` come from the per-rank model
    geometry and must be supplied by the caller (typically via
    :meth:`from_sparse_params`).
    """

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_index_heads: int
    sparse_index_dim: int
    block_size: int
    topk: int
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"

    def __post_init__(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.num_index_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_index_heads ({self.num_index_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.topk <= 0:
            raise ValueError(f"topk must be > 0, got {self.topk}")
        if self.init_blocks < 0:
            raise ValueError(f"init_blocks must be >= 0, got {self.init_blocks}")
        if self.local_blocks < 0:
            raise ValueError(f"local_blocks must be >= 0, got {self.local_blocks}")
        if self.score_type != "max":
            # SGLang exposes only "max" today and that is what the MiniMax-M3
            # checkpoint config specifies. Reject anything else explicitly so
            # a config drift surfaces immediately.
            raise ValueError(
                f"score_type={self.score_type!r} is not supported "
                "(only 'max' matches the SGLang reference)"
            )

    @classmethod
    def from_sparse_params(
        cls,
        sparse_params: "MiniMaxM3SparseParams",
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> "MiniMaxM3SparseConfig":
        """Build a kernel param bundle from lowered ``MiniMaxM3SparseParams``
        and the per-rank model geometry.
        """
        return cls(
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_index_heads=int(sparse_params.num_index_heads),
            sparse_index_dim=int(sparse_params.sparse_index_dim),
            block_size=int(sparse_params.block_size),
            topk=int(sparse_params.topk),
            init_blocks=int(sparse_params.init_blocks),
            local_blocks=int(sparse_params.local_blocks),
            score_type=str(sparse_params.score_type),
        )


def require_msa_module():
    """Import fmha_sm100 and raise a clear error when it is missing.

    The import is deferred to first kernel use so the MSA backend can be
    advertised in the config schema on systems where fmha_sm100 is absent.
    """
    try:
        import fmha_sm100
    except ImportError as exc:
        raise RuntimeError(
            "MiniMax-M3 MSA attention requires the external fmha_sm100 package "
            "(https://github.com/MiniMax-AI/MSA). Install it, or unset "
            "sparse_use_msa to use the Triton reference path."
        ) from exc
    return fmha_sm100


def write_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write per-token values into a K, V, or index-K cache at given slots.

    Supports a 3-D flat-slot cache [num_slots, num_heads, channel] used by
    focused tests, and a 4-D paged view [num_pages, tokens_per_block,
    num_heads, channel]. The paged view is non-contiguous, so a plain
    index_copy_ would lose the write; decompose the slot id into (page,
    within) and use multi-dim assignment so the write reaches the pool.
    """
    with torch.no_grad():
        if cache.ndim >= 4:
            tokens_per_block = int(cache.shape[1])
            out_long = out_cache_loc.to(torch.long)
            page = out_long // tokens_per_block
            within = out_long % tokens_per_block
            cache[page, within] = values.to(cache.dtype)
        else:
            cache.index_copy_(0, out_cache_loc.to(torch.long), values.to(cache.dtype))


def cache_view_to_msa_paged(cache_view: torch.Tensor) -> torch.Tensor:
    """Convert a KV cache view to the fmha_sm100 HND paged layout.

    A 4-D paged view [num_pages, page_size, num_heads, head_dim] permutes to
    [num_pages, num_heads, page_size, head_dim]. A 3-D flat-slot cache
    [num_slots, num_heads, head_dim] is treated as one virtual page, giving
    [1, num_heads, num_slots, head_dim].
    """
    if cache_view.dim() == 4:
        return cache_view.permute(0, 2, 1, 3).contiguous()
    if cache_view.dim() == 3:
        return cache_view.permute(1, 0, 2).unsqueeze(0).contiguous()
    raise ValueError(
        f"Unsupported cache view rank {cache_view.dim()} for MSA paged conversion; "
        "expected 3 (flat-slot) or 4 (paged)."
    )


def msa_paged_kv(kv_cache_manager, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer paged K and V in fmha_sm100 HND layout."""
    buffers = kv_cache_manager.get_buffers(layer_idx)
    return cache_view_to_msa_paged(buffers[:, 0]), cache_view_to_msa_paged(buffers[:, 1])


def write_msa_main_kv(
    kv_cache_manager,
    layer_idx: int,
    out_cache_loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Write new-token K and V into the paged main cache at out_cache_loc.

    fmha_sm100 reads the paged cache directly, so the new-token K and V must
    be resident before the sparse GQA runs.
    """
    buffers = kv_cache_manager.get_buffers(layer_idx)
    k_view, v_view = buffers[:, 0], buffers[:, 1]
    num_kv_heads = int(k_view.shape[2])
    head_dim = int(k_view.shape[3])
    num_tokens = int(k.shape[0])
    write_kv_slots(k_view, out_cache_loc, k.reshape(num_tokens, num_kv_heads, head_dim))
    write_kv_slots(v_view, out_cache_loc, v.reshape(num_tokens, num_kv_heads, head_dim))


def build_kv_page_indices(
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Build the flattened per-request page table fmha_sm100 consumes.

    Returns int32 global page ids concatenated per request. A request's
    pages come from the first slot of each page in its req_to_token row.
    Page ids are global and non-contiguous in production, so they are not
    clamped to a per-request bound.
    """
    device = req_to_token.device
    req_rows = req_to_token.index_select(0, slot_ids.to(torch.long)).to(torch.long)
    batch = int(req_rows.shape[0])
    kv_lens_list = kv_lens_cpu.to(torch.long).tolist()

    page_lists = []
    for b in range(batch):
        kv_len = int(kv_lens_list[b])
        if kv_len <= 0:
            continue
        num_pages = (kv_len + page_size - 1) // page_size
        page_starts = torch.arange(num_pages, device=device, dtype=torch.long) * page_size
        page_ids = req_rows[b].gather(0, page_starts) // page_size
        page_lists.append(page_ids.to(torch.int32))

    if page_lists:
        return torch.cat(page_lists, dim=0)
    return torch.empty(0, dtype=torch.int32, device=device)


def per_token_valid_blocks(
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    *,
    causal: bool,
    block_size: int,
) -> torch.Tensor:
    """Return the per-query number of valid KV blocks, on CPU.

    Expands per-request lengths and offsets to a per-token vector so block
    selection can honour each query token's own causal extent.
    """
    qo = qo_lens_cpu.to(torch.long)
    kv = kv_lens_cpu.to(torch.long)
    batch = int(qo.shape[0])
    total = int(qo.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.long)
    batch_row = torch.repeat_interleave(torch.arange(batch, dtype=torch.long), qo)
    starts = torch.zeros(batch, dtype=torch.long)
    if batch > 1:
        starts[1:] = torch.cumsum(qo, 0)[:-1]
    intra = torch.arange(total, dtype=torch.long) - starts[batch_row]
    kv_per = kv[batch_row]
    if causal:
        if qo_offset_cpu is not None:
            off = qo_offset_cpu.to(torch.long)[batch_row]
        else:
            off = (kv - qo)[batch_row]
        eff = torch.minimum(off + intra + 1, kv_per)
    else:
        eff = kv_per
    return (eff + block_size - 1) // block_size


def select_blocks_from_maxscore(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Select per-query top-k blocks from per-KV-head block scores.

    Applies init and local forced blocks and per-query valid-block masking
    on the amax-reduced scores [num_kv_heads, n_blocks, total_q]. Returns
    [total_q, num_kv_heads, topk] int32 ascending block ids with -1 tail
    padding.
    """
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    scores = max_score_kv.permute(2, 0, 1).to(torch.float32).clone()
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.long)
    nvb = n_valid_blocks.to(device=device, dtype=torch.long)

    if init_blocks > 0:
        init_mask = block_ids.view(1, 1, -1) < init_blocks
        scores = torch.where(init_mask, torch.full_like(scores, _INIT_SCORE), scores)
    if local_blocks > 0:
        local_start = (nvb - local_blocks).clamp_min(0)
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < nvb.view(-1, 1)
        )
        scores = torch.where(local_mask.unsqueeze(1), torch.full_like(scores, _LOCAL_SCORE), scores)
    block_valid = block_ids.view(1, -1) < nvb.view(-1, 1)
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    k = min(topk, n_blocks)
    vals, idx = scores.topk(k=k, dim=-1)
    idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
    sort_key = torch.where(idx < 0, torch.full_like(idx, n_blocks), idx)
    sort_key, _ = torch.sort(sort_key, dim=-1)
    idx = torch.where(sort_key >= n_blocks, torch.full_like(sort_key, -1), sort_key)
    if k < topk:
        pad = torch.full((total_q, num_kv_heads, topk - k), -1, dtype=idx.dtype, device=device)
        idx = torch.cat([idx, pad], dim=-1)
    return idx.to(torch.int32)


__all__ = [
    "MSA_REQUIRED_HEAD_DIM",
    "MSA_REQUIRED_TOPK",
    "MiniMaxM3SparseConfig",
    "MiniMaxM3SparseParams",
    "build_kv_page_indices",
    "cache_view_to_msa_paged",
    "msa_paged_kv",
    "per_token_valid_blocks",
    "require_msa_module",
    "select_blocks_from_maxscore",
    "write_kv_slots",
    "write_msa_main_kv",
]
