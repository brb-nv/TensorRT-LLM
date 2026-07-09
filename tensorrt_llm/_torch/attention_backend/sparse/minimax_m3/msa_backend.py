# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA-backed MiniMax-M3 sparse attention on the TrtllmAttention stack.

Mimics DSATrtllmAttention:

  * MsaMinimaxM3Attention subclasses TrtllmAttention and reuses its
    inherited forward, overriding only the sparse hooks and owning an
    MsaIndexer.
  * The main sparse GQA runs through the registered MsaSparseGqaFmha.
  * The indexer calls fmha_sm100 directly to produce the per-query selected
    block indices, which the model layer threads through
    forward_args.topk_indices.
  * MsaMinimaxM3AttentionMetadata subclasses TrtllmAttentionMetadata and
    declares the per-forward MSA fields directly, built from the standard
    attention metadata and the KV cache manager.

The classes are defined inside get_minimax_m3_msa_attention_backend_cls with
a deferred trtllm import, avoiding an import cycle at package init.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from .common import MSA_REQUIRED_HEAD_DIM, MSA_REQUIRED_TOPK, build_kv_page_indices, write_kv_slots
from .indexer import MsaIndexer
from .metadata import MiniMaxM3SparseConfig, build_runtime_metadata_from_kv_manager

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs


def _cache_device(meta) -> torch.device:
    """Device hosting the paged KV buffers, else the current CUDA device."""
    kv_cache_manager = meta.kv_cache_manager
    if kv_cache_manager is not None:
        try:
            return kv_cache_manager.get_buffers(0).device
        except Exception:
            pass
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@functools.lru_cache(maxsize=1)
def get_minimax_m3_msa_attention_backend_cls():
    """Return MsaMinimaxM3Attention (the MSA backend selection entry point)."""
    from dataclasses import dataclass

    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

    @dataclass(init=False)
    class MsaMinimaxM3AttentionMetadata(TrtllmAttentionMetadata):
        """TrtllmAttentionMetadata for MiniMax-M3 MSA sparse layers.

        The per-forward MSA state is declared flat on this class and built
        in prepare() from the standard attention metadata plus the KV cache
        manager. Nothing here is nested or attached dynamically.
        """

        msa_is_prefill: bool = False
        msa_req_to_token: Optional[torch.Tensor] = None
        msa_slot_ids: Optional[torch.Tensor] = None
        msa_kv_lens_dev: Optional[torch.Tensor] = None
        msa_kv_lens_cpu: Optional[torch.Tensor] = None
        msa_qo_lens_cpu: Optional[torch.Tensor] = None
        msa_qo_offset_cpu: Optional[torch.Tensor] = None
        msa_kv_indices: Optional[torch.Tensor] = None
        msa_out_cache_loc: Optional[torch.Tensor] = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._clear_msa_fields()

        def _clear_msa_fields(self) -> None:
            self.msa_is_prefill = False
            self.msa_req_to_token = None
            self.msa_slot_ids = None
            self.msa_kv_lens_dev = None
            self.msa_kv_lens_cpu = None
            self.msa_qo_lens_cpu = None
            self.msa_qo_offset_cpu = None
            self.msa_kv_indices = None
            self.msa_out_cache_loc = None

        def prepare(self) -> None:
            super().prepare()
            self._build_msa_fields()

        def _build_msa_fields(self) -> None:
            """Derive the flat MSA per-forward tensors for this step.

            Reuses build_runtime_metadata_from_kv_manager for the tested
            req_to_token and out_cache_loc derivation, then computes the
            fmha_sm100 page table and per-request lengths. The transient
            builder result is not retained; only the flat fields are.
            """
            self._clear_msa_fields()
            kv_cache_manager = self.kv_cache_manager
            if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_index_k_buffer"):
                return
            request_ids = self.request_ids
            seq_lens = self.seq_lens
            if request_ids is None or seq_lens is None:
                return
            batch_size = int(seq_lens.shape[0])
            if batch_size == 0:
                return

            num_contexts = int(self.num_contexts or 0)
            cache_device = _cache_device(self)
            page_size = int(kv_cache_manager.tokens_per_block)

            seq_lens_cpu = self.seq_lens_cpu
            if seq_lens_cpu is None:
                seq_lens_cpu = seq_lens.detach().to("cpu")

            kv_cache_params = self.kv_cache_params
            num_cached = (
                kv_cache_params.num_cached_tokens_per_seq
                if kv_cache_params is not None
                else [0] * batch_size
            )
            kv_lens_list = [
                int(num_cached[b]) + int(seq_lens_cpu[b].item()) for b in range(batch_size)
            ]
            kv_lens_cpu = torch.tensor(kv_lens_list, dtype=torch.int32)
            kv_lens_dev = kv_lens_cpu.to(device=cache_device, non_blocking=True)

            is_prefill = num_contexts > 0
            if not is_prefill and int(seq_lens_cpu[:batch_size].max().item()) > 1:
                raise NotImplementedError(
                    "MiniMax-M3 MSA attention does not support speculative decoding "
                    "(multiple query tokens per decode step). Disable speculative "
                    "decoding or use the non-MSA MiniMax-M3 backend."
                )

            if is_prefill:
                prefix_lens_list = [int(num_cached[b]) for b in range(batch_size)]
                extend_seq_lens_cpu = [
                    kv_lens_list[b] - prefix_lens_list[b] for b in range(batch_size)
                ]
                prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=cache_device)
                m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                    kv_cache_manager=kv_cache_manager,
                    request_ids=request_ids,
                    seq_lens=kv_lens_dev,
                    seq_lens_cpu=kv_lens_cpu,
                    is_prefill=True,
                    prefix_lens=prefix_lens,
                    extend_seq_lens_cpu=extend_seq_lens_cpu,
                    device=cache_device,
                )
                qo_lens_cpu = torch.tensor(extend_seq_lens_cpu, dtype=torch.int32)
            else:
                m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                    kv_cache_manager=kv_cache_manager,
                    request_ids=request_ids,
                    seq_lens=kv_lens_dev,
                    seq_lens_cpu=kv_lens_cpu,
                    is_prefill=False,
                    device=cache_device,
                )
                qo_lens_cpu = torch.ones(batch_size, dtype=torch.int32)

            qo_offset_cpu = (kv_lens_cpu.to(torch.long) - qo_lens_cpu.to(torch.long)).to(
                torch.int32
            )
            kv_indices = build_kv_page_indices(
                m3_meta.req_to_token, m3_meta.slot_ids, kv_lens_cpu, page_size
            )

            self.msa_is_prefill = is_prefill
            self.msa_req_to_token = m3_meta.req_to_token
            self.msa_slot_ids = m3_meta.slot_ids
            self.msa_kv_lens_dev = kv_lens_dev
            self.msa_kv_lens_cpu = kv_lens_cpu
            self.msa_qo_lens_cpu = qo_lens_cpu
            self.msa_qo_offset_cpu = qo_offset_cpu
            self.msa_kv_indices = kv_indices
            self.msa_out_cache_loc = out_cache_loc

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            """Paged index-K view for the indexer; HND conversion is done there."""
            return self.kv_cache_manager.get_index_k_buffer(layer_idx)

        def msa_write_idx_k(self, layer_idx: int, idx_k: torch.Tensor) -> None:
            """Write the new-token index-K into the side cache at out_cache_loc."""
            cache = self.msa_idx_k_cache(layer_idx)
            sparse_index_dim = int(cache.shape[-1])
            num_tokens = int(idx_k.shape[0])
            write_kv_slots(
                cache, self.msa_out_cache_loc, idx_k.reshape(num_tokens, 1, sparse_index_dim)
            )

    class MsaMinimaxM3Attention(TrtllmAttention):
        """MSA-backed MiniMax-M3 sparse attention (mimics DSATrtllmAttention)."""

        Metadata = MsaMinimaxM3AttentionMetadata

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

        def _validate_msa_preconditions(self) -> None:
            config = self.m3_config
            if not self.disable_index_value:
                raise NotImplementedError(
                    "MSA backend requires disable_index_value=True; the proxy pass "
                    "consumes only the max score and has no index-V path."
                )
            if config.head_dim != MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires head_dim={MSA_REQUIRED_HEAD_DIM}, got {config.head_dim}."
                )
            if config.sparse_index_dim != MSA_REQUIRED_HEAD_DIM:
                raise NotImplementedError(
                    f"MSA backend requires sparse_index_dim={MSA_REQUIRED_HEAD_DIM}, "
                    f"got {config.sparse_index_dim}."
                )
            if config.topk != MSA_REQUIRED_TOPK:
                raise NotImplementedError(
                    f"MSA backend requires topk={MSA_REQUIRED_TOPK}, got {config.topk}."
                )

        @classmethod
        def support_fused_rope(cls) -> bool:
            # The MiniMax-M3 model layer applies partial RoPE to the main and
            # index branches explicitly.
            return False

        def run_indexer(
            self,
            idx_q: torch.Tensor,
            idx_k: torch.Tensor,
            metadata,
            *,
            idx_sm_scale: Optional[float] = None,
        ) -> torch.Tensor:
            """Write the index-K cache and return the selected block indices.

            Mirrors DSA's indexer entry point: the model layer runs this
            before forward and threads the result through
            forward_args.topk_indices. Returns [total_q, num_kv_heads, topk].
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
            return self.indexer.select_blocks(
                idx_q_view,
                idx_k_cache,
                idx_sm_scale=idx_sm_scale,
                qo_lens_cpu=metadata.msa_qo_lens_cpu,
                kv_lens_cpu=metadata.msa_kv_lens_cpu,
                qo_offset_cpu=metadata.msa_qo_offset_cpu,
                kv_indices=metadata.msa_kv_indices,
            )

        def sparse_attn_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args: "AttentionForwardArgs",
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            # The model layer runs run_indexer and passes the selected block
            # indices through forward_args.topk_indices. Publish them as the
            # sparse attention indices MsaSparseGqaFmha reads.
            return forward_args.topk_indices, None

        def sparse_kv_predict(
            self,
            q: torch.Tensor,
            k: Optional[torch.Tensor],
            metadata,
            forward_args: "AttentionForwardArgs",
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            return None, None

    return MsaMinimaxM3Attention


__all__ = ["get_minimax_m3_msa_attention_backend_cls"]
