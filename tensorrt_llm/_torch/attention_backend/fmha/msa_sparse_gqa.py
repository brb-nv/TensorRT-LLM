# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-sparse GQA FMHA backed by MSA's fmha_sm100 kernel.

MsaSparseGqaFmha wraps the fmha_sm100 paged sparse GQA kernel and
participates in the standard TrtllmAttention.forward dispatch loop. The
owning MiniMax-M3 MSA attention layer runs an MsaIndexer to select the
per-query KV blocks and publishes them on forward_args.sparse_prediction;
this class attends over them.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


def _msa_metadata_cls() -> type:
    """The concrete metadata class that drives MsaSparseGqaFmha.

    Resolved lazily because the class is built inside a factory with a
    deferred trtllm import, which would otherwise form an import cycle at
    attention-backend package init.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
        get_minimax_m3_msa_attention_backend_cls,
    )

    return get_minimax_m3_msa_attention_backend_cls().Metadata


def run_msa_sparse_gqa(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    kv_block_indexes: Optional[torch.Tensor] = None,
    *,
    kv_indices: torch.Tensor,
    sm_scale: float,
    qo_lens_cpu: Optional[torch.Tensor] = None,
    kv_lens_cpu: Optional[torch.Tensor] = None,
    qo_offset_cpu: Optional[torch.Tensor] = None,
    causal: bool = True,
    head_dim: int = 128,
    plan: Optional[tuple] = None,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Run fmha_sm100 paged GQA (plan/run split).

    `kv_block_indexes` selects the mode. When provided, it is the per-query
    top-k block table and the kernel attends only those blocks (MiniMax-M3
    sparse layers); the plan is built with a fixed `kv_block_num=topk`. When
    None, the kernel attends every page listed in `kv_indices` for each
    request (MiniMax-M3 dense layers); the plan omits `kv_block_num`, so
    each request may use a different number of KV pages with no top-k limit.

    `plan` is the fmha_sm100 execution plan. When None, the plan is built
    inline from qo_lens_cpu/kv_lens_cpu/qo_offset_cpu; this is used for prefill
    and focused tests, which run eagerly rather than inside CUDA graph capture.
    CUDA-graph decode passes a plan prebuilt in metadata.prepare(), so planning
    stays outside capture and the captured region only runs the kernel. `out`,
    when provided, receives the result in place; otherwise a fresh output
    tensor is allocated and returned.
    """
    import fmha_sm100

    if q.dim() != 3:
        raise ValueError(
            f"MsaSparseGqaFmha expects q [total_q, num_qo_heads, head_dim]; got {tuple(q.shape)}."
        )
    if q.shape[-1] != head_dim:
        raise NotImplementedError(
            f"MsaSparseGqaFmha supports head_dim={head_dim}; got {q.shape[-1]}."
        )
    if k_paged.dim() != 4 or v_paged.dim() != 4:
        raise ValueError(
            "MsaSparseGqaFmha expects paged KV [num_pages, num_kv_heads, page_size, head_dim]; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )
    if k_paged.shape != v_paged.shape:
        raise ValueError(
            f"MsaSparseGqaFmha requires k and v to share shape; "
            f"got k={tuple(k_paged.shape)}, v={tuple(v_paged.shape)}."
        )

    if plan is None:
        # kv_block_num is planned only for the sparse (block-indexed) path;
        # dense paged GQA leaves it unset and attends the full page table.
        kv_block_num = int(kv_block_indexes.shape[-1]) if kv_block_indexes is not None else -1
        plan = fmha_sm100.fmha_sm100_plan(
            qo_lens_cpu,
            kv_lens_cpu,
            int(q.shape[1]),  # num query heads.
            num_kv_heads=int(k_paged.shape[1]),
            qo_offset=qo_offset_cpu,
            page_size=int(k_paged.shape[2]),
            kv_block_num=kv_block_num,
            causal=causal,
            num_kv_splits=1,
        )
    out_result, _ = fmha_sm100.fmha_sm100(
        q,
        k_paged,
        v_paged,
        plan,
        kv_indices=kv_indices,
        kv_block_indexes=kv_block_indexes,
        out=out,
        sm_scale=sm_scale,
        output_maxscore=False,
    )
    return out_result


def run_msa_paged_gqa(
    attn: "TrtllmAttention",
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    metadata: "TrtllmAttentionMetadata",
    output: torch.Tensor,
    *,
    kv_block_indexes: Optional[torch.Tensor],
    plan: Optional[tuple],
) -> None:
    """Write the new-token main K/V, then run paged GQA into output in place.

    Shared by the sparse layers (kv_block_indexes is the per-query top-k table,
    with the sparse plan) and the dense layers (kv_block_indexes None, with the
    dense plan, attending the full page table). fmha_sm100 reads the paged cache
    directly, so the new-token K/V must be resident before the run.
    """
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
        msa_paged_kv,
        write_msa_main_kv,
    )

    layer_idx = attn.layer_idx
    head_dim = attn.head_dim
    kv_cache_manager = metadata.kv_cache_manager
    num_tokens = int(q.shape[0])
    if k is not None and v is not None:
        write_msa_main_kv(
            kv_cache_manager, layer_idx, metadata.msa_out_cache_loc[:num_tokens], k, v
        )

    q_view = q.view(num_tokens, attn.num_heads, head_dim)
    out_view = output.view(num_tokens, attn.num_heads, head_dim)
    k_paged, v_paged = msa_paged_kv(kv_cache_manager, layer_idx)
    sm_scale = (head_dim**-0.5) / float(attn.q_scaling)

    run_msa_sparse_gqa(
        q_view,
        k_paged,
        v_paged,
        kv_block_indexes,
        kv_indices=metadata.msa_kv_indices,
        sm_scale=sm_scale,
        qo_lens_cpu=metadata.msa_qo_lens_cpu,
        kv_lens_cpu=metadata.msa_kv_lens_cpu,
        qo_offset_cpu=metadata.msa_qo_offset_cpu,
        causal=True,
        head_dim=head_dim,
        plan=plan,
        out=out_view,
    )


class MsaSparseGqaFmha(Fmha):
    """SM100 paged GQA FMHA powered by MSA's fmha_sm100 kernel.

    Handles every MiniMax-M3 MSA layer. Sparse layers pass the indexer's
    selected KV block indices on forward_args.sparse_prediction.sparse_attn_indices
    and attend those blocks; dense layers leave the indices None and attend the
    full page table.

        Inherits Fmha rather than PhasedFmha: fmha_sm100 takes a single plan and
        the selected block indices span the whole batch, so it handles a mixed
        context and generation batch in one call and there is no
        context/generation split from PhasedFmha to reuse. Requires head_dim 128
        and 4-D HND paged K/V.
    """

    HEAD_DIM = 128
    REQUIRES_PAGED_KV = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.kv_factor = 2
        self.generation_out_head_size = self.HEAD_DIM
        self.context_out_head_size = self.HEAD_DIM

    @classmethod
    def is_available(cls, attn: Optional["TrtllmAttention"] = None) -> bool:
        # Only the MiniMax-M3 MSA attention layer uses this library. Filtering
        # on the owning type lets the base create_fmha_libs add it to that
        # layer alone, so no create_fmha_libs override is needed. Availability
        # of the fmha_sm100 package and an SM100 device is gated earlier, when
        # the MSA backend is selected.
        from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_backend import (
            get_minimax_m3_msa_attention_backend_cls,
        )

        return isinstance(attn, get_minimax_m3_msa_attention_backend_cls())

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> bool:
        # Claims every MiniMax-M3 MSA forward. Sparse layers carry the per-query
        # top-k table in sparse_attn_indices; dense layers leave it None and
        # attend the full page table. Both run fmha_sm100 through this lib.
        return isinstance(metadata, _msa_metadata_cls())

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: "AttentionForwardArgs",
    ) -> None:
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires an output buffer.")

        # Sparse layers attend the per-query top-k blocks with the sparse plan;
        # dense layers leave the indices None and attend the full page table
        # with the dense plan. The shared helper writes the main K/V cache and
        # runs the paged GQA either way.
        kv_block_indexes = forward_args.sparse_prediction.sparse_attn_indices
        plan = (
            metadata.msa_decode_gqa_plan
            if kv_block_indexes is not None
            else metadata.msa_decode_dense_plan
        )
        run_msa_paged_gqa(
            self.attn,
            q,
            k,
            v,
            metadata,
            output,
            kv_block_indexes=kv_block_indexes,
            plan=plan,
        )


__all__ = ["MsaSparseGqaFmha", "run_msa_paged_gqa", "run_msa_sparse_gqa"]
