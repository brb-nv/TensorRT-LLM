# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HiSparse decode-residency coordinator for MiniMax-M3.

Keeps only a bounded per-request window of *sparse main-KV* blocks resident on
device during decode. The full sparse main-KV history is parked on a pinned host
pool; each decode step brings the per-(sparse-layer) top-k selected blocks into a
device "hot buffer" via the block-granular swap-in kernel
(:func:`torch.ops.trtllm.hisparse_swap_in_blocks`, a port of SGLang HiSparse's
``load_cache_to_device_buffer_kernel``).

Ownership boundary:
  * This coordinator owns the device hot buffer, the pinned host pool, and the
    per-(layer, request) residency bookkeeping (``device_buffer_blocks``,
    ``device_buffer_locs``, ``lru_slots``) required by the kernel.
  * Index-K (sparse layers) and dense KV stay fully resident in the normal
    KVCacheManagerV2 pool groups and are untouched here.
  * Prefill and prefix reuse are unchanged; this only fires for decode + sparse
    layers + sparse main KV.

Indexing conventions mirror the kernel contract exactly:
  * ``top_k_blocks`` / ``top_k_device_locs`` are indexed by the *in-batch*
    request id ``bid`` (grid/block id), rows = ``num_reqs``.
  * ``device_buffer_blocks`` / ``device_buffer_locs`` / ``host_block_locs`` /
    ``lru_slots`` are indexed by the *request-pool* id ``rid`` =
    ``req_pool_indices[bid]``, rows = ``max_reqs``.
  * A "block" here is one sparse block == one KV page (MiniMax-M3 forces
    ``sparse_block_size == tokens_per_block == 128``).
  * ``seq_lens_blocks`` is measured in *blocks*, so the newest (current) block id
    is ``seq_lens_blocks - 1`` and is always bound to the reserved slot.

All state tensors are allocated once and mutated in place, so the swap-in call is
CUDA-graph capturable: the padded (``bid >= num_real_reqs``) rows early-return.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch

__all__ = ["MiniMaxM3HiSparseCoordinator"]


class MiniMaxM3HiSparseCoordinator:
    """Owns sparse main-KV hot-window residency for MiniMax-M3 decode.

    Args:
        num_sparse_layers: Number of sparse local layers on this rank.
        max_reqs: Request-pool capacity (rows of the per-request state).
        hot_window_blocks: Blocks kept GPU-resident per request (the swap-in
            hot-buffer size); must be ``>= topk``.
        topk: Number of top-k blocks selected per sparse layer per step.
        max_blocks_per_seq: Upper bound on a request's block count; sizes the
            per-request host region and ``host_block_locs`` width.
        page_shape: Per-block page shape of one main-K (== one main-V) page in
            the manager's layout (HND for MSA: ``[num_kv_heads, tokens_per_block,
            head_dim]``). K and V share this shape.
        page_dtype: Storage dtype of a main-KV page (FP8/plain; NVFP4 is gated
            off upstream because its block scales live in separate pools).
        device: CUDA device for the hot buffer / bookkeeping tensors.
        cuda_block_size: Threads per CUDA block for the swap-in kernel (multiple
            of 32, ``<= 1024``).
    """

    def __init__(
        self,
        *,
        num_sparse_layers: int,
        max_reqs: int,
        hot_window_blocks: int,
        topk: int,
        max_blocks_per_seq: int,
        page_shape: Sequence[int],
        page_dtype: torch.dtype,
        device: torch.device | str = "cuda",
        cuda_block_size: int = 128,
    ) -> None:
        if hot_window_blocks < topk:
            raise ValueError(
                f"hot_window_blocks ({hot_window_blocks}) must be >= topk ({topk})."
            )
        if cuda_block_size <= 0 or cuda_block_size % 32 != 0 or cuda_block_size > 1024:
            raise ValueError(
                "cuda_block_size must be a positive multiple of 32 and <= 1024, "
                f"got {cuda_block_size}.")

        self.num_sparse_layers = int(num_sparse_layers)
        self.max_reqs = int(max_reqs)
        self.hot_window_blocks = int(hot_window_blocks)
        self.topk = int(topk)
        self.max_blocks_per_seq = int(max_blocks_per_seq)
        self.page_shape = tuple(int(d) for d in page_shape)
        self.page_dtype = page_dtype
        self.cuda_block_size = int(cuda_block_size)
        self.device = torch.device(device)

        # Reserved "newest" slot lives at index hot_window_blocks; it is excluded
        # from the LRU set (kernel binds the current block to it every step).
        self.padded_buffer_blocks = self.hot_window_blocks + 1

        page_numel = math.prod(self.page_shape) if self.page_shape else 1
        elem_bytes = torch.empty((), dtype=self.page_dtype).element_size()
        self.item_size_bytes = int(page_numel * elem_bytes)
        if self.item_size_bytes <= 0 or self.item_size_bytes % 8 != 0:
            raise ValueError(
                "HiSparse main-KV page byte size must be a positive multiple of "
                f"8, got {self.item_size_bytes} (page_shape={self.page_shape}, "
                f"dtype={self.page_dtype}).")

        L, R = self.num_sparse_layers, self.max_reqs
        pad = self.padded_buffer_blocks

        # --- Device hot buffer (one K and one V pool per sparse layer) ---
        # Physical pages laid out request-major: page = rid * pad + slot.
        num_hot_pages = R * pad
        self.device_buffer_k: List[torch.Tensor] = [
            torch.zeros((num_hot_pages, *self.page_shape),
                        dtype=self.page_dtype,
                        device=self.device) for _ in range(L)
        ]
        self.device_buffer_v: List[torch.Tensor] = [
            torch.zeros((num_hot_pages, *self.page_shape),
                        dtype=self.page_dtype,
                        device=self.device) for _ in range(L)
        ]

        # --- Pinned host pool (full sparse main-KV history) ---
        # Per-request contiguous region: host page = rid * max_blocks_per_seq +
        # block_id, encoded in host_block_locs (identity here). One K and one V
        # pool per sparse layer.
        num_host_pages = R * self.max_blocks_per_seq
        self.host_cache_k: List[torch.Tensor] = [
            torch.zeros((num_host_pages, *self.page_shape),
                        dtype=self.page_dtype,
                        device="cpu",
                        pin_memory=True) for _ in range(L)
        ]
        self.host_cache_v: List[torch.Tensor] = [
            torch.zeros((num_host_pages, *self.page_shape),
                        dtype=self.page_dtype,
                        device="cpu",
                        pin_memory=True) for _ in range(L)
        ]

        # --- Per-(layer, request) residency bookkeeping (device) ---
        # device_buffer_locs[layer, rid, slot] = physical hot-buffer page id.
        base = torch.arange(R, dtype=torch.int32,
                            device=self.device).view(R, 1) * pad
        slot = torch.arange(pad, dtype=torch.int32, device=self.device).view(1, pad)
        locs_2d = (base + slot).contiguous()  # [R, pad]
        self.device_buffer_locs = locs_2d.unsqueeze(0).repeat(L, 1, 1).contiguous()
        # device_buffer_blocks[layer, rid, slot] = resident block id (-1 empty).
        self.device_buffer_blocks = torch.full((L, R, pad),
                                               -1,
                                               dtype=torch.int32,
                                               device=self.device)
        # lru_slots[layer, rid, :] = LRU ordering over the hot_window_blocks slots
        # (front = least-recently-used). Reserved newest slot excluded.
        self.lru_slots = (torch.arange(self.hot_window_blocks,
                                       dtype=torch.int16,
                                       device=self.device).view(
                                           1, 1, -1).repeat(L, R, 1).contiguous())

        # host_block_locs[rid, block_id] = host page id (identity mapping).
        hb = torch.arange(self.max_blocks_per_seq,
                          dtype=torch.int64,
                          device=self.device).view(1, -1)
        self.host_block_locs = (
            hb + torch.arange(R, dtype=torch.int64, device=self.device).view(
                R, 1) * self.max_blocks_per_seq).contiguous()

        # --- Per-step I/O buffers (graph-safe; filled out-of-graph) ---
        self.top_k_device_locs = torch.full((R, self.topk),
                                            -1,
                                            dtype=torch.int32,
                                            device=self.device)
        self.req_pool_indices = torch.zeros((R, ),
                                            dtype=torch.int32,
                                            device=self.device)
        self.seq_lens_blocks = torch.zeros((R, ),
                                           dtype=torch.int32,
                                           device=self.device)
        self.num_real_reqs = torch.zeros((1, ),
                                         dtype=torch.int32,
                                         device=self.device)

    # ------------------------------------------------------------------
    # Per-request lifecycle
    # ------------------------------------------------------------------
    def reset_request(self, rid: int) -> None:
        """Clear a request-pool slot's residency state before (re)use.

        Marks every hot-buffer slot empty and resets the LRU order for the given
        ``rid`` across all sparse layers. Physical page assignments
        (``device_buffer_locs``, ``host_block_locs``) are fixed and untouched.
        """
        self.device_buffer_blocks[:, rid, :].fill_(-1)
        self.lru_slots[:, rid, :] = torch.arange(self.hot_window_blocks,
                                                 dtype=torch.int16,
                                                 device=self.device)

    def newest_slot_page(self, layer_idx: int, rid: int) -> int:
        """Physical hot-buffer page id of the reserved newest slot for ``rid``.

        The current (being-generated) block's main KV must be written here each
        decode step so the kernel can bind it as a resident hit.
        """
        return int(self.device_buffer_locs[layer_idx, rid,
                                           self.hot_window_blocks])

    def host_page(self, rid: int, block_id: int) -> int:
        """Host page id backing ``(rid, block_id)`` in the pinned host pool."""
        return int(self.host_block_locs[rid, block_id])

    # ------------------------------------------------------------------
    # Per-step bookkeeping (call OUTSIDE the captured CUDA graph)
    # ------------------------------------------------------------------
    def prepare_decode_step(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens_blocks: torch.Tensor,
        num_real_reqs: int,
    ) -> None:
        """Publish this step's batch mapping into the static I/O buffers.

        Args:
            req_pool_indices: ``[num_reqs]`` in-batch -> request-pool id.
            seq_lens_blocks: ``[num_reqs]`` per-request block count (blocks, not
                tokens). The newest block id is ``seq_lens_blocks - 1``.
            num_real_reqs: Non-padded batch size; padded rows early-return in the
                kernel so a fixed captured grid stays valid.
        """
        n = int(req_pool_indices.shape[0])
        if n > self.max_reqs:
            raise ValueError(
                f"batch {n} exceeds coordinator max_reqs {self.max_reqs}.")
        self.req_pool_indices[:n].copy_(req_pool_indices.to(torch.int32),
                                        non_blocking=True)
        self.seq_lens_blocks[:n].copy_(seq_lens_blocks.to(torch.int32),
                                       non_blocking=True)
        self.num_real_reqs.fill_(int(num_real_reqs))

    # ------------------------------------------------------------------
    # Swap-in (safe to call INSIDE the captured CUDA graph)
    # ------------------------------------------------------------------
    def swap_in(self, layer_idx: int, top_k_blocks: torch.Tensor) -> torch.Tensor:
        """Ensure ``top_k_blocks`` for ``layer_idx`` are resident; return locs.

        Runs the fused hit/LRU/miss swap-in for one sparse layer:
          * hits: already-resident selected blocks keep their device page;
          * newest block: bound to the reserved slot;
          * misses: LRU-evict a slot and copy the block's K+V page host->device.

        Args:
            layer_idx: Sparse-layer index (0..num_sparse_layers-1).
            top_k_blocks: ``[num_reqs, topk]`` int32 selected block ids per
                in-batch request (``-1`` pads unused slots).

        Returns:
            ``self.top_k_device_locs[:num_reqs]`` (a view): physical hot-buffer
            page id per selected block, ``-1`` where unresolved/padded. This is
            the block table the MSA sparse decode consumes (top-k position ->
            device page).
        """
        num_reqs = int(top_k_blocks.shape[0])
        if top_k_blocks.shape[1] != self.topk:
            raise ValueError(
                f"top_k_blocks second dim {top_k_blocks.shape[1]} != topk "
                f"{self.topk}.")
        out = self.top_k_device_locs[:num_reqs]
        torch.ops.trtllm.hisparse_swap_in_blocks(
            top_k_blocks,
            self.device_buffer_blocks[layer_idx],
            self.host_block_locs,
            self.device_buffer_locs[layer_idx],
            self.host_cache_k[layer_idx],
            self.host_cache_v[layer_idx],
            self.device_buffer_k[layer_idx],
            self.device_buffer_v[layer_idx],
            out,
            self.req_pool_indices[:num_reqs],
            self.seq_lens_blocks[:num_reqs],
            self.lru_slots[layer_idx],
            self.num_real_reqs,
            self.topk,
            self.hot_window_blocks,
            self.item_size_bytes,
            self.cuda_block_size,
        )
        return out

    # ------------------------------------------------------------------
    # Host staging helpers (device->host backup of full sparse main KV)
    # ------------------------------------------------------------------
    def backup_block_to_host(
        self,
        layer_idx: int,
        rid: int,
        block_id: int,
        src_k_page: torch.Tensor,
        src_v_page: torch.Tensor,
    ) -> None:
        """Copy one main-KV block's K and V pages device->host pinned pool.

        Used at prefill completion (and when a decode block is finalized) to park
        the block in the host pool so it can later be swapped back in. Shapes must
        match ``page_shape``.
        """
        host_pg = self.host_page(rid, block_id)
        self.host_cache_k[layer_idx][host_pg].copy_(src_k_page, non_blocking=True)
        self.host_cache_v[layer_idx][host_pg].copy_(src_v_page, non_blocking=True)
