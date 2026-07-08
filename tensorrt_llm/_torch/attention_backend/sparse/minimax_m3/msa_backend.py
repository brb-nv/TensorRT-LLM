# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the `TrtllmAttention` stack.

Mimics `DSATrtllmAttention`:

  * `MiniMaxM3MSATrtllmAttention` subclasses `TrtllmAttention` and reuses
    its inherited `forward` (the standard `fmha_libs` dispatch loop). It
    overrides only the sparse hooks (`sparse_attn_predict`,
    `sparse_kv_predict`) and owns an `MsaIndexer`. It does not inherit the
    Triton reference backend.
  * The main sparse GQA runs through the registered `MsaSparseGqaFmha`
    (an `Fmha` that does its own whole-batch dispatch) selected by the
    dispatch loop.
  * The indexer calls `fmha_sm100` directly (prefill) or the graph-safe
    decode driver (decode) to produce the per-query selected block
    indices, which the model layer threads through
    `forward_args.topk_indices`.
  * `MiniMaxM3MSATrtllmAttentionMetadata` subclasses
    `TrtllmAttentionMetadata`, so the inherited `forward` reads the
    standard metadata fields. It owns the CUDA-graph-stable page-table
    buffers and the built attachment; the paged K/V views and sparse GQA
    dispatch are handled directly in `MsaSparseGqaFmha`.

The backend and metadata classes are defined inside
`get_minimax_m3_msa_attention_backend_cls` (with a lazy `trtllm` import)
so that importing this module during attention-backend package init does
not pull in `trtllm` -> `fmha` -> `interface` and form an import cycle.
This mirrors the Triton reference backend's
`get_minimax_m3_attention_backend_cls` factory.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple

import torch

from .common import (
    _MSA_REQUIRED_HEAD_DIM,
    _MSA_REQUIRED_TOPK,
    build_kv_indices_and_lens,
    msa_paged_kv,
    write_main_kv_slots,
)
from .indexer import MsaIndexer
from .metadata import MiniMaxM3SparseConfig, build_m3_sparse_metadata_and_plans

if TYPE_CHECKING:
    from .metadata import MiniMaxM3SparseAttentionMetadata, MsaGeometry


def _whole_batch_lens(
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Whole-batch `(qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, kv_indices)`.

    Prefers the pre-staged `msa_plans` (CUDA-graph-stable buffers) and
    falls back to an eager rebuild for focused tests and the first eager
    warmup pass.
    """
    msa_plans = getattr(m3_meta, "msa_plans", None)
    if msa_plans is not None:
        return (
            msa_plans["qo_lens_cpu"],
            msa_plans["kv_lens_cpu"],
            msa_plans["qo_offset_cpu"],
            msa_plans["kv_indices"],
        )
    seq_lens_cpu = m3_meta.seq_lens_cpu.to(torch.int32)
    if m3_meta.is_prefill:
        if m3_meta.extend_seq_lens_cpu is None or m3_meta.prefix_lens is None:
            raise RuntimeError("prefill metadata requires extend_seq_lens_cpu / prefix_lens")
        qo_lens_cpu = torch.tensor(m3_meta.extend_seq_lens_cpu, dtype=torch.int32)
        qo_offset_cpu = m3_meta.prefix_lens.detach().to(device="cpu", dtype=torch.int32)
    else:
        batch = int(seq_lens_cpu.shape[0])
        qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
        qo_offset_cpu = (seq_lens_cpu - 1).to(torch.int32)
    kv_indices, _ = build_kv_indices_and_lens(m3_meta, page_size)
    return qo_lens_cpu, seq_lens_cpu, qo_offset_cpu, kv_indices


def run_msa_sparse_decode(
    config: "MiniMaxM3SparseConfig",
    kv_cache_manager,
    layer_idx: int,
    m3_meta: "MiniMaxM3SparseAttentionMetadata",
    q: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """CUDA-graph-safe decode sparse GQA via the in-tree driver.

    Decode is CUDA-graph captured, so it must not go through the eager
    `fmha_sm100_plan` host driver, which uses unpinned H2D staging,
    per-call device allocations, and a device-side cost sweep. The staged
    page tables come from the metadata's pre-built `msa_plans`. Returns
    `[num_tokens, num_q_heads * head_dim]`.
    """
    from .decode_wrapper.dispatch import M3DecodeGeometry, get_decode_driver

    msa_plans = getattr(m3_meta, "msa_plans", None)
    if msa_plans is None:
        raise RuntimeError(
            "MiniMax-M3 MSA decode requires pre-staged msa_plans; "
            "prepare() did not build them (missing geometry / use_msa)."
        )
    k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
    page_size = int(k_paged.shape[2])
    seq_lens = m3_meta.seq_lens.to(torch.int32)
    batch = int(q.shape[0])
    geometry = M3DecodeGeometry(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        num_index_heads=config.num_index_heads,
        head_dim=config.head_dim,
        page_size=page_size,
        topk=config.topk,
        init_blocks=config.init_blocks,
        local_blocks=config.local_blocks,
        max_batch=int(msa_plans.get("max_batch") or 0) or max(64, 1 << (batch - 1).bit_length()),
        max_kv_len=int(msa_plans.get("max_kv_len") or 0) or int(m3_meta.req_to_token.shape[1]),
    )
    driver = get_decode_driver(geometry, q.device)
    out = driver.sparse_attention(
        q,
        k_paged,
        v_paged,
        kv_block_indexes,
        seq_lens=seq_lens,
        kv_page_indptr=msa_plans["kv_page_indptr"],
        kv_indices=msa_plans["kv_indices"],
        sm_scale=sm_scale,
    )
    return out.reshape(batch, config.num_q_heads * config.head_dim)


@functools.lru_cache(maxsize=1)
def get_minimax_m3_msa_attention_backend_cls():
    """Return `MiniMaxM3MSATrtllmAttention` (selection entry point).

    Defined lazily (with a deferred `trtllm` import) so importing this
    module during attention-backend package init does not form the
    `trtllm` -> `fmha` -> `interface` import cycle.
    """
    from dataclasses import dataclass

    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    @dataclass(init=False)
    class MiniMaxM3MSATrtllmAttentionMetadata(TrtllmAttentionMetadata):
        """`TrtllmAttentionMetadata` for MiniMax-M3 MSA sparse layers.

        Holds the MSA staging (page-table buffers, per-request CPU
        lengths, index-K writes) on top of the standard metadata. The
        whole-batch paged K/V views and sparse GQA dispatch live in
        `MsaSparseGqaFmha`; this class only owns the CUDA-graph-stable
        buffers and the built attachment.
        """

        # Published by the model layer's first sparse forward. Set on the
        # class so CUDA-graph metadata clones see it before capture.
        _msa_geometry: ClassVar[Optional["MsaGeometry"]] = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.minimax_m3: Optional[dict] = None
            self._m3_static_buffers: Optional[dict] = None
            self._msa_kv_indices_buf = None
            self._msa_kv_page_indptr_buf = None

        # lifecycle

        def _cache_device(self) -> torch.device:
            kv_cache_manager = getattr(self, "kv_cache_manager", None)
            if kv_cache_manager is not None:
                try:
                    return kv_cache_manager.get_buffers(0).device
                except Exception:
                    pass
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        def _maybe_static_buffers(self, cache_device: torch.device) -> Optional[dict]:
            """Return persistent M3 buffers when CUDA-graph stability is needed."""
            need_static = bool(getattr(self, "is_cuda_graph", False)) or (
                self._m3_static_buffers is not None
            )
            if not need_static:
                return None
            if (
                self._m3_static_buffers is not None
                and self._m3_static_buffers.get("device") == cache_device
            ):
                return self._m3_static_buffers
            placeholder: dict = {
                "device": cache_device,
                "max_num_sequences_hint": int(
                    getattr(self, "max_num_sequences", None) or self.max_num_requests
                ),
                "max_num_tokens_hint": int(
                    getattr(self, "max_num_tokens", None)
                    or (int(getattr(self, "max_num_sequences", None) or self.max_num_requests))
                ),
            }
            self._m3_static_buffers = placeholder
            return placeholder

        def prepare(self) -> None:
            super().prepare()
            self.minimax_m3 = None
            geometry = type(self)._msa_geometry
            if geometry is None:
                from .metadata import get_global_msa_geometry

                geometry = get_global_msa_geometry()
            cache_device = self._cache_device()
            static_buffers = self._maybe_static_buffers(cache_device)
            (
                attachment,
                self._msa_kv_indices_buf,
                self._msa_kv_page_indptr_buf,
            ) = build_m3_sparse_metadata_and_plans(
                self,
                static_buffers=static_buffers,
                kv_indices_buf=self._msa_kv_indices_buf,
                kv_page_indptr_buf=self._msa_kv_page_indptr_buf,
                geometry=geometry,
            )
            self.minimax_m3 = attachment

        # internal accessors

        def _require_attachment(self) -> dict:
            if self.minimax_m3 is None:
                raise RuntimeError(
                    "MiniMaxM3MSATrtllmAttentionMetadata.minimax_m3 is not built; "
                    "prepare() must run before the sparse forward."
                )
            return self.minimax_m3

        @property
        def m3_meta(self) -> "MiniMaxM3SparseAttentionMetadata":
            return self._require_attachment()["metadata"]

        @property
        def m3_out_cache_loc(self) -> torch.Tensor:
            return self._require_attachment()["out_cache_loc"]

        # Index-K cache access + write (consumed by the MsaIndexer via
        # run_indexer). The main-KV paged views and sparse GQA dispatch
        # are handled directly in MsaSparseGqaFmha.

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            """Raw paged index-K view for the indexer; HND conversion is done there."""
            return self.kv_cache_manager.get_index_k_buffer(layer_idx)

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            idx_cache = self.msa_idx_k_cache(layer_idx)
            sparse_index_dim = int(idx_cache.shape[-1])
            num_tokens = int(idx_k.shape[0])
            write_main_kv_slots(
                idx_cache, self.m3_out_cache_loc, idx_k.reshape(num_tokens, 1, sparse_index_dim)
            )

    class MiniMaxM3MSATrtllmAttention(TrtllmAttention):
        """MSA-backed MiniMax-M3 sparse attention (mimics `DSATrtllmAttention`)."""

        Metadata = MiniMaxM3MSATrtllmAttentionMetadata

        def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config=None,
            *,
            sparse_params,
            **kwargs,
        ):
            TrtllmAttention.__init__(
                self,
                layer_idx,
                num_heads,
                head_dim,
                num_kv_heads=num_kv_heads,
                quant_config=quant_config,
                sparse_params=sparse_params,
                **kwargs,
            )
            self.m3_config = MiniMaxM3SparseConfig.from_sparse_params(
                sparse_params,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads or num_heads,
                head_dim=head_dim,
            )
            self.disable_index_value = bool(sparse_params.disable_index_value)
            self._validate_msa_preconditions()
            self.indexer = MsaIndexer(self.m3_config)
            self._register_global_geometry()

        def _validate_msa_preconditions(self) -> None:
            if not self.disable_index_value:
                raise NotImplementedError(
                    "MSA backend requires disable_index_value=True (MSA's proxy pass "
                    "only consumes max_score; an index-V path is not implemented)."
                )
            if self.m3_config.head_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires head_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.head_dim}."
                )
            if self.m3_config.sparse_index_dim != _MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires sparse_index_dim={_MSA_REQUIRED_HEAD_DIM}, "
                    f"got {self.m3_config.sparse_index_dim}."
                )
            if self.m3_config.topk != _MSA_REQUIRED_TOPK:
                raise NotImplementedError(
                    f"MSA backend requires topk={_MSA_REQUIRED_TOPK}, got {self.m3_config.topk}."
                )

        def _register_global_geometry(self) -> None:
            # Register the per-rank sparse geometry process-wide at
            # construction, before any forward or CUDA graph capture, so
            # every metadata instance's prepare() can pre-build the MSA
            # plans.
            from .metadata import MsaGeometry, set_global_msa_geometry

            set_global_msa_geometry(
                MsaGeometry(
                    num_q_heads=int(self.m3_config.num_q_heads),
                    num_kv_heads=int(self.m3_config.num_kv_heads),
                    num_index_heads=int(self.m3_config.num_index_heads),
                    head_dim=int(self.m3_config.head_dim),
                    block_size=int(self.m3_config.block_size),
                    topk=int(self.m3_config.topk),
                    init_blocks=int(self.m3_config.init_blocks),
                    local_blocks=int(self.m3_config.local_blocks),
                )
            )

        @classmethod
        def support_fused_rope(cls) -> bool:
            # The MiniMax-M3 model layer applies partial RoPE to both the
            # main and index branches explicitly.
            return False

        def create_fmha_libs(self) -> None:
            # The MSA layer runs its main attention exclusively through the
            # fmha_sm100 block-sparse GQA kernel. Restrict the dispatch loop
            # to MsaSparseGqaFmha so the standard libs (flashinfer
            # trtllm-gen, fallback), which come first in the registry and
            # would otherwise claim the request but cannot handle the M3
            # sparse contract (unfused q/k/v plus selected block indices),
            # never win.
            from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha

            self.fmha_libs = [MsaSparseGqaFmha(self)]

        # indexer entry point (called by the model layer, DSA-style)

        def run_indexer(
            self,
            idx_q: torch.Tensor,
            idx_k: torch.Tensor,
            metadata,
            *,
            idx_sm_scale: Optional[float] = None,
        ) -> torch.Tensor:
            """Write the index-K cache and return the selected block indices.

            Mirrors DSA's `indexer.sparse_attn_indexer`: the model layer
            runs this before `forward` and threads the result through
            `forward_args.topk_indices`. Returns `[total_q, num_kv_heads,
            topk]`.
            """
            config = self.m3_config
            idx_sm_scale = (
                idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
            )
            num_tokens = int(idx_q.shape[0])
            idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
            idx_k_view = idx_k.view(num_tokens, 1, config.sparse_index_dim)

            metadata.msa_write_idx_k(self.layer_idx, idx_k_view)
            idx_k_cache = metadata.msa_idx_k_cache(self.layer_idx)
            m3_meta = metadata.m3_meta
            # The cache's tokens_per_block equals the sparse block size
            # (enforced by the M3 KV cache manager), so read it from the
            # config instead of the cache view.
            page_size = self.m3_config.block_size

            if m3_meta.is_prefill:
                qo, kv, qo_off, kv_indices = _whole_batch_lens(m3_meta, page_size)
                return self.indexer.select_blocks_prefill(
                    idx_q_view,
                    idx_k_cache,
                    m3_meta,
                    idx_sm_scale=idx_sm_scale,
                    qo_lens_cpu=qo,
                    kv_lens_cpu=kv,
                    qo_offset_cpu=qo_off,
                    kv_indices=kv_indices,
                )
            return self.indexer.select_blocks_decode(
                idx_q_view,
                idx_k_cache,
                m3_meta,
                idx_sm_scale=idx_sm_scale,
                page_size=page_size,
            )

        # sparse hooks (consumed by inherited TrtllmAttention.forward)

        def sparse_attn_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            # The model layer runs run_indexer() and passes the selected
            # block indices through forward_args.topk_indices. Publish them
            # as the sparse attention indices the MsaSparseGqaFmha reads.
            return getattr(forward_args, "topk_indices", None), None

        def sparse_kv_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            return None, None

    return MiniMaxM3MSATrtllmAttention


__all__ = [
    "get_minimax_m3_msa_attention_backend_cls",
]
