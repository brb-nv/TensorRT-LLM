# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 index-K storage and the MiniMax MSA-v1 Q8KV4 indexer for ``indexer_kv_dtype='nvfp4'``.

Each sparse layer's INDEX_KEY page holds the NVFP4 index-K of its 128 tokens, 72 bytes per
token instead of the 128 of E4M3:

* ``[0, 8192)``: packed E2M1 rows ``[128, 64]``, two elements per byte, low nibble first;
* ``[8192, 9216)``: their E4M3 block scales ``[128, 8]``, one per 16 elements, token-major.

The global scale is 1. This is the layout MiniMax MSA's ``msa_v1.indexer.decode.q8kv4`` reads (vendored at
``fmha/msa_kernels/msa_v1``, vendor entry msa-v1-q8kv4) and
the fused NVFP4 horizontal producer writes.

Generation rows (decode and verify) are scored by the Q8KV4 decode indexer straight from that
layout. A plan per (page table, lengths, query length, head count) is built outside capture and
cached; its length-dependent schedule is refreshed by one ``update()`` per forward, issued by the
first sparse layer that runs it, so CUDA graphs replay it. Context rows are scored by the
fmha_sm100 proxy pass over an E4M3 copy of only the pages it reads, which
``dequantize_index_k_pages`` writes once per layer into a step workspace.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

PLAN_CACHE_KEY = "minimax_m3_msa_v1_q8kv4_indexer"

PAGE = 128
HEAD_DIM = 128
TOPK = 16
SF_VEC = 16
NVFP4_DATA_PAGE_BYTES = PAGE * HEAD_DIM // 2
NVFP4_SCALE_PAGE_BYTES = PAGE * HEAD_DIM // SF_VEC
# Bytes per token of an NVFP4 INDEX_KEY page: 64 bytes of E2M1 plus 8 E4M3 scales.
NVFP4_INDEX_K_BYTES_PER_TOKEN = (NVFP4_DATA_PAGE_BYTES + NVFP4_SCALE_PAGE_BYTES) // PAGE
SUPPORTED_INDEX_HEADS = (1, 2, 4)
MAX_QUERY_LEN = 16
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
_E4M3_MAX = 448.0


@functools.lru_cache(maxsize=1)
def _msa_q8kv4_classes():
    """(Q8KV4 decode wrapper class, shared plan class), or None when the vendored indexer is not importable."""
    try:
        from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.msa_v1.indexer.decode import (
            BatchDecodeIndexerPlan,
        )
        from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.msa_v1.indexer.decode.q8kv4 import (
            BatchDecodeIndexerWithPagedKVCacheWrapper,
        )
    except ImportError:
        return None
    return BatchDecodeIndexerWithPagedKVCacheWrapper, BatchDecodeIndexerPlan


def q8kv4_available() -> bool:
    """Whether the MiniMax MSA-v1 Q8KV4 decode indexer is importable."""
    return _msa_q8kv4_classes() is not None


def require_q8kv4() -> None:
    """Raise unless the Q8KV4 decode indexer is importable."""
    if not q8kv4_available():
        raise ImportError(
            "MiniMax-M3 indexer_kv_dtype='nvfp4' scores decode with the MiniMax MSA-v1 Q8KV4 "
            "indexer, but the vendored msa_kernels.msa_v1.indexer.decode.q8kv4 is not importable "
            "(its CuTe DSL, tvm_ffi or jinja2 dependency is missing); use indexer_kv_dtype='fp8'."
        )


def split_nvfp4_index_k(raw: torch.Tensor):
    """Split an NVFP4 INDEX_KEY view into its E2M1 rows and E4M3 scales.

    Args:
        raw: [num_pages, 1, 128, 72] uint8 view of the INDEX_KEY pool (any page stride,
            each page contiguous).

    Returns:
        (data [num_pages, 128, 64] uint8, scale [num_pages, 128, 8] float8_e4m3fn), both
        sharing the pool's page stride.
    """
    if raw.dim() != 4 or tuple(raw.shape[1:]) != (1, PAGE, NVFP4_INDEX_K_BYTES_PER_TOKEN):
        raise ValueError(f"unexpected NVFP4 index-K view shape {tuple(raw.shape)}")
    if (
        raw.dtype != torch.uint8
        or raw.stride(3) != 1
        or raw.stride(2) != NVFP4_INDEX_K_BYTES_PER_TOKEN
    ):
        raise ValueError("NVFP4 index-K view must be uint8 with contiguous pages")
    num_pages, page_stride, base = int(raw.shape[0]), raw.stride(0), raw.storage_offset()
    data = torch.as_strided(
        raw, (num_pages, PAGE, HEAD_DIM // 2), (page_stride, HEAD_DIM // 2, 1), base
    )
    scale = torch.as_strided(
        raw,
        (num_pages, PAGE, HEAD_DIM // SF_VEC),
        (page_stride, HEAD_DIM // SF_VEC, 1),
        base + NVFP4_DATA_PAGE_BYTES,
    ).view(torch.float8_e4m3fn)
    return data, scale


@triton.jit
def _e2m1_value(code):
    """Signed value of a 4-bit E2M1 code (bit 3 sign, bits 2:1 exponent, bit 0 mantissa)."""
    exponent = (code >> 1) & 3
    mantissa = (code & 1).to(tl.float32)
    magnitude = tl.where(
        exponent == 0,
        mantissa * 0.5,
        (1.0 + mantissa * 0.5) * tl.exp2((exponent - 1).to(tl.float32)),
    )
    return tl.where((code & 8) != 0, -magnitude, magnitude)


@triton.jit
def _dequantize_pages_kernel(
    data_ptr,
    scale_ptr,
    src_pages_ptr,
    out_ptr,
    data_page_stride,
    scale_page_stride,
    out_page_stride,
    TOKENS: tl.constexpr,
    HALF_D: tl.constexpr,
    GROUPS: tl.constexpr,
    BYTES_PER_GROUP: tl.constexpr,
    E4M3_MAX: tl.constexpr,
):
    page = tl.program_id(0)
    token_block = tl.program_id(1)
    src = tl.load(src_pages_ptr + page).to(tl.int64)
    tokens = token_block * TOKENS + tl.arange(0, TOKENS)[:, None]
    byte = tl.arange(0, HALF_D)[None, :]
    packed = tl.load(data_ptr + src * data_page_stride + tokens * HALF_D + byte).to(tl.int32)
    scale = (
        tl.load(scale_ptr + src * scale_page_stride + tokens * GROUPS + byte // BYTES_PER_GROUP)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    lo = tl.clamp(_e2m1_value(packed & 15) * scale, -E4M3_MAX, E4M3_MAX)
    hi = tl.clamp(_e2m1_value(packed >> 4) * scale, -E4M3_MAX, E4M3_MAX)
    dst = out_ptr + page.to(tl.int64) * out_page_stride + tokens * (2 * HALF_D) + 2 * byte
    tl.store(dst, lo.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    tl.store(dst + 1, hi.to(tl.float8e4nv).to(tl.uint8, bitcast=True))


def dequantize_index_k_pages(
    data: torch.Tensor,
    scale: torch.Tensor,
    src_pages: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Dequantize the listed NVFP4 index-K pages into a compact E4M3 page table.

    Page ``i`` of ``out`` receives pool page ``src_pages[i]``, so a page table over the
    listed pages addresses ``out`` with ``arange(len(src_pages))``.

    Args:
        data: [num_pool_pages, 128, 64] uint8 E2M1 rows (page-strided).
        scale: [num_pool_pages, 128, 8] float8_e4m3fn block scales (page-strided).
        src_pages: [n] int32 pool page ids.
        out: [>= n, 1, 128, 128] float8_e4m3fn, contiguous pages.

    Returns:
        out[:n].
    """
    num_pages = int(src_pages.shape[0])
    if num_pages > int(out.shape[0]):
        raise ValueError(f"workspace holds {int(out.shape[0])} pages, {num_pages} requested")
    if num_pages == 0:
        return out[:0]
    if data.stride(2) != 1 or data.stride(1) != HEAD_DIM // 2:
        raise ValueError("NVFP4 index-K rows must be contiguous within a page")
    if scale.stride(2) != 1 or scale.stride(1) != HEAD_DIM // SF_VEC:
        raise ValueError("NVFP4 index-K scales must be contiguous within a page")
    if out.stride(3) != 1 or out.stride(2) != HEAD_DIM:
        raise ValueError("E4M3 workspace rows must be contiguous")
    tokens = 32
    _dequantize_pages_kernel[(num_pages, PAGE // tokens)](
        data,
        scale.view(torch.uint8),
        src_pages,
        out.view(torch.uint8),
        data.stride(0),
        scale.stride(0),
        out.stride(0),
        TOKENS=tokens,
        HALF_D=HEAD_DIM // 2,
        GROUPS=HEAD_DIM // SF_VEC,
        BYTES_PER_GROUP=SF_VEC // 2,
        E4M3_MAX=_E4M3_MAX,
        num_warps=4,
    )
    return out[:num_pages]


@triton.jit
def _sort_topk_rows_kernel(table_ptr, num_rows, K: tl.constexpr, ROWS: tl.constexpr):
    """Sort each K-wide row ascending in place with -1 entries moved to the end."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    k = tl.arange(0, K)[None, :]
    mask = r < num_rows
    x = tl.load(table_ptr + r * K + k, mask=mask, other=-1)
    big = 2147483647
    key = tl.where(x < 0, big, x)
    key = tl.sort(key, dim=1)
    tl.store(table_ptr + r * K + k, tl.where(key == big, -1, key), mask=mask)


class _ScoreArena:
    """Process-wide FP32 proxy-score buffers shared by every Q8KV4 plan on a device.

    Each plan's wrapper writes [heads, rows, max_pages] scores that its own select consumes
    right away, and all plans run on one stream, so they can share storage. A buffer is
    never freed or resized, since captured graphs keep its address; a plan that needs more
    than the largest one gets a new, larger buffer.
    """

    def __init__(self):
        self._buffers: dict[torch.device, list[torch.Tensor]] = {}

    def view(self, shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        numel = shape[0] * shape[1] * shape[2]
        buffers = self._buffers.setdefault(device, [])
        for buffer in buffers:
            if buffer.numel() >= numel:
                return buffer[:numel].view(shape)
        buffer = torch.empty(numel, dtype=torch.float32, device=device)
        buffers.append(buffer)
        return buffer.view(shape)


_SCORE_ARENA = _ScoreArena()


@dataclass
class Q8KV4Plan:
    """One planned Q8KV4 decode-indexer launch geometry."""

    wrapper: object
    shared_plan: object
    batch: int
    query_len: int
    num_heads: int
    out: torch.Tensor  # [num_heads, batch * query_len, 16] int32
    first_layer: Optional[int] = None


def q8kv4_supported(
    *, num_heads: int, num_kv_heads: int, query_len: int, max_pages: int, device: torch.device
) -> bool:
    """Whether the Q8KV4 decode indexer serves this geometry."""
    return (
        num_heads == num_kv_heads
        and num_heads in SUPPORTED_INDEX_HEADS
        and 1 <= query_len <= MAX_QUERY_LEN
        and 1 <= max_pages <= 16384
        and torch.cuda.get_device_capability(device) in SUPPORTED_CAPABILITIES
    )


def get_or_plan(
    plan_cache: dict,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    num_heads: int,
) -> Q8KV4Plan:
    """Return the cached plan for this geometry, building it (outside capture) if missing.

    Args:
        plan_cache: Per-metadata dict the plans live in.
        block_table: [batch, max_pages] int32 page table, contiguous; bound by address.
        seq_lens: [batch] int32 lengths including this step's tokens, contiguous; bound by
            address and re-read by every update().
        query_len: Query tokens per request.
        num_heads: Index heads on this rank, one per KV head.
    """
    key = (
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(query_len),
        int(num_heads),
    )
    plan = plan_cache.get(key)
    if plan is not None:
        return plan
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MiniMax-M3 NVFP4 index-K: no Q8KV4 indexer plan for batch "
            f"{key[0]} x {query_len} queries before CUDA graph capture; prepare() builds it."
        )
    if not (block_table.is_contiguous() and seq_lens.is_contiguous()):
        raise ValueError("Q8KV4 indexer metadata must be contiguous")
    if block_table.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("Q8KV4 indexer metadata must be int32")
    require_q8kv4()
    wrapper_cls, plan_cls = _msa_q8kv4_classes()
    batch, max_pages = int(block_table.shape[0]), int(block_table.shape[1])
    shared_plan = plan_cls(
        seq_lens,
        max_pages=max_pages,
        num_index_heads=int(num_heads),
        query_length=int(query_len),
    )
    wrapper = wrapper_cls(
        use_cuda_graph=True, page_table_buffer=block_table, seq_lens_buffer=seq_lens
    )
    # The wrapper keeps a [heads, rows, max_pages] FP32 score buffer per plan unless one of
    # that shape is already in place; hand it a view of the shared arena instead, as one plan
    # per CUDA graph batch size would otherwise hold its own worst-case copy.
    wrapper._proxy_scores = _SCORE_ARENA.view(
        (int(num_heads), batch * int(query_len), max_pages), block_table.device
    )
    wrapper.plan(
        block_table,
        seq_lens,
        num_index_heads=int(num_heads),
        query_length=int(query_len),
        shared_plan=shared_plan,
    )
    out = torch.empty(
        (int(num_heads), batch * int(query_len), TOPK), dtype=torch.int32, device=block_table.device
    )
    plan = Q8KV4Plan(
        wrapper=wrapper,
        shared_plan=shared_plan,
        batch=batch,
        query_len=int(query_len),
        num_heads=int(num_heads),
        out=out,
    )
    plan_cache[key] = plan
    return plan


def run_q8kv4(
    plan: Q8KV4Plan,
    idx_q: torch.Tensor,
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    layer_idx: int,
) -> torch.Tensor:
    """Select this layer's blocks for the planned generation rows.

    Args:
        plan: From get_or_plan for these rows.
        idx_q: [batch * query_len, num_heads, 128] float8_e4m3fn index queries, request-major.
        data / scale: The layer's NVFP4 index-K from split_nvfp4_index_k.
        layer_idx: The calling layer; the first one to run the plan refreshes its schedule
            once per forward.

    Returns:
        [batch * query_len, num_heads, 16] int32 on a head-major backing, ascending with -1
        padding: the contract MsaIndexer.select_blocks returns.
    """
    total_q = plan.batch * plan.query_len
    if tuple(idx_q.shape) != (total_q, plan.num_heads, HEAD_DIM):
        raise ValueError(
            f"index Q {tuple(idx_q.shape)} does not match the planned "
            f"({total_q}, {plan.num_heads}, {HEAD_DIM})"
        )
    if plan.first_layer is None:
        plan.first_layer = int(layer_idx)
    if int(layer_idx) == plan.first_layer:
        plan.shared_plan.update()
    q = idx_q if idx_q.is_contiguous() else idx_q.contiguous()
    plan.wrapper.run(
        q.view(plan.batch, plan.query_len, plan.num_heads, HEAD_DIM),
        data,
        k_scale=scale,
        out=plan.out,
    )
    rows = plan.num_heads * total_q
    block_rows = 16
    _sort_topk_rows_kernel[(triton.cdiv(rows, block_rows),)](
        plan.out, rows, K=TOPK, ROWS=block_rows, num_warps=1
    )
    return plan.out.permute(1, 0, 2)


class DequantWorkspace:
    """Grow-only E4M3 page workspace for the context rows' index-K, plus its identity page table.

    Only eager (context-carrying) steps use it, so growing it reallocates freely.
    """

    _GRANULE_PAGES = 256

    def __init__(self):
        self._pages: Optional[torch.Tensor] = None
        self._identity: Optional[torch.Tensor] = None

    def pages(self, num_pages: int, device: torch.device) -> torch.Tensor:
        """[>= num_pages, 1, 128, 128] float8_e4m3fn."""
        if self._pages is None or int(self._pages.shape[0]) < num_pages:
            # Grow by at least half again so a slowly lengthening context does not reallocate
            # (and fragment the allocator's headroom outside the KV pool) every few steps.
            current = 0 if self._pages is None else int(self._pages.shape[0])
            wanted = max(num_pages, 1, current + current // 2)
            capacity = -(-wanted // self._GRANULE_PAGES) * self._GRANULE_PAGES
            self._pages = None
            self._pages = torch.empty(
                (capacity, 1, PAGE, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device
            )
        return self._pages

    def identity(self, num_pages: int, device: torch.device) -> torch.Tensor:
        """arange(>= num_pages) int32."""
        if self._identity is None or int(self._identity.shape[0]) < num_pages:
            capacity = -(-max(num_pages, 1) // self._GRANULE_PAGES) * self._GRANULE_PAGES
            self._identity = torch.arange(capacity, dtype=torch.int32, device=device)
        return self._identity


__all__ = [
    "NVFP4_INDEX_K_BYTES_PER_TOKEN",
    "PLAN_CACHE_KEY",
    "DequantWorkspace",
    "Q8KV4Plan",
    "dequantize_index_k_pages",
    "get_or_plan",
    "q8kv4_available",
    "q8kv4_supported",
    "require_q8kv4",
    "run_q8kv4",
    "split_nvfp4_index_k",
]
