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
    stores its per-forward MSA tensors in CUDA-graph-stable buffers.
    Following DSAtrtllmAttentionMetadata, the buffers are allocated once in
    __post_init__ via get_empty(capture_graph=...), and prepare() copies the
    per-step values into them. The standard CUDAGraphRunner clones one
    metadata per graph batch size (create_cuda_graph_metadata), so no
    per-batch-size cache is needed here.

The classes are defined inside get_minimax_m3_msa_attention_backend_cls with
a deferred trtllm import, avoiding an import cycle at package init.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._utils import prefer_pinned

from .common import (
    MSA_REQUIRED_HEAD_DIM,
    MSA_REQUIRED_TOPK,
    build_kv_page_indices,
    per_token_valid_blocks,
    require_msa_module,
    write_kv_slots,
)
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


# Per-step fmha_sm100 plan tensors that must live in CUDA-graph-stable buffers.
# At num_kv_splits=1 the plan carries no split-KV workspaces, and
# cute_workspace_buffer is the vendor's cached scratch (kept by reference, not
# copied). Validated by scripts_local/msa_2b_probe.py on SM100.
_MSA_PLAN_STABLE_KEYS = (
    "packed_work_range",
    "packed_work_info",
    "qo_segment_offsets",
    "kv_segment_offsets",
    "kv_page_indptr",
    "qo_segment_lens",
    "kv_segment_lens",
    "qo_offset",
)
_MSA_PLAN_INT64_KEYS = ("packed_work_range", "packed_work_info")
# fmha_sm100 sizes packed_work_info at 131072 * max(num_kv_splits, 1); forcing
# num_kv_splits=1 pins this worklist width.
_MSA_PACKED_WORK_INFO_LEN = 131072
_MSA_SPLIT_KV_KEYS = (
    "kv_tile_begin_indices",
    "kv_tile_end_indices",
    "kv_split_indices",
    "num_kv_splits_per_row",
    "workspace_o",
    "workspace_lse",
)


class _MsaGraphSafePlan:
    """CUDA-graph-stable mirror of one fmha_sm100 decode plan.

    Owns fixed device buffers for the per-step plan worklists. refresh() copies
    a freshly built plan into them and returns a plan tuple pointing at the
    stable buffers, so the captured fmha_sm100 run reads addresses that do not
    change across replays. Mirrors FlashInfer's fixed indptr/indices buffers.

    Only valid at num_kv_splits=1: the plan then has no split-KV workspaces
    (refresh() asserts this), and cute_workspace_buffer and the scalar fields
    pass through unchanged.
    """

    def __init__(self, metadata, name: str, *, max_batch: int, num_ctas: int, capture_graph: bool):
        buffers = metadata.cuda_graph_buffers
        self._buf = {}
        for key in _MSA_PLAN_STABLE_KEYS:
            if key == "packed_work_range":
                shape = (num_ctas,)
            elif key == "packed_work_info":
                shape = (_MSA_PACKED_WORK_INFO_LEN,)
            elif key in ("qo_segment_offsets", "kv_segment_offsets", "kv_page_indptr"):
                shape = (max_batch + 1,)
            else:
                shape = (max_batch,)
            dtype = torch.int64 if key in _MSA_PLAN_INT64_KEYS else torch.int32
            self._buf[key] = metadata.get_empty(
                buffers,
                shape,
                cache_name=f"{name}_{key}",
                dtype=dtype,
                capture_graph=capture_graph,
            )

    def refresh(self, plan_tuple) -> tuple:
        has_mixed, split, batch, decode, prefill = plan_tuple
        if has_mixed:
            raise RuntimeError(
                "MSA decode expects a single (non-mixed) fmha_sm100 plan; a decode "
                "batch must be pure decode."
            )
        for key in _MSA_SPLIT_KV_KEYS:
            if decode.get(key) is not None:
                raise RuntimeError(
                    f"MSA decode plan used split-KV workspace {key!r}; num_kv_splits=1 "
                    "is required for graph-safe decode."
                )
        rebuilt = dict(decode)
        for key in _MSA_PLAN_STABLE_KEYS:
            src = decode.get(key)
            if src is None:
                continue
            n = int(src.shape[0])
            dst = self._buf[key]
            if n > dst.shape[0]:
                raise ValueError(
                    f"MSA plan buffer {key} ({dst.shape[0]}) is smaller than the plan tensor ({n})."
                )
            dst[:n].copy_(src, non_blocking=True)
            rebuilt[key] = dst[:n]
        return (has_mixed, split, batch, rebuilt, prefill)


def _lookup_msa_attention_layer(layer_idx: int):
    """Resolve the MSA sparse backend for a layer via the model layer registry.

    Returns the MsaMinimaxM3Attention backend (which owns the geometry) or None
    when the registry is unavailable, e.g. focused tests that build metadata
    without running the model. Reuses the same per-layer registry the attention
    custom op uses; it is not a separate geometry singleton.
    """
    from tensorrt_llm._torch.utils import get_model_extra_attrs

    extra_attrs = get_model_extra_attrs()
    if not extra_attrs:
        return None
    attn_layers = extra_attrs.get("attn_layers")
    if not attn_layers:
        return None
    ref = attn_layers.get(str(layer_idx))
    model_layer = ref() if ref is not None else None
    if model_layer is None:
        return None
    backend = getattr(model_layer, "attn", None)
    if backend is not None and hasattr(backend, "build_decode_plans"):
        return backend
    return None


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

        Only the tensors the sparse GQA and cache write actually consume are
        stored: the per-new-token cache slots (msa_out_cache_loc), the paged
        page table (msa_kv_indices), and the per-request lengths and causal
        offset that fmha_sm100_plan reads (msa_kv_lens_cpu, msa_qo_lens_cpu,
        msa_qo_offset_cpu).

        Buffers are allocated once in __post_init__ and populated in prepare()
        via copy_, so their addresses stay stable across CUDA-graph replays.
        The device tensors come from get_empty on the shared graph buffer pool;
        the host tensors are pinned so the plan build stays sync-free.
        """

        msa_out_cache_loc: Optional[torch.Tensor] = None
        msa_kv_indices: Optional[torch.Tensor] = None
        msa_kv_lens_cpu: Optional[torch.Tensor] = None
        msa_qo_lens_cpu: Optional[torch.Tensor] = None
        msa_qo_offset_cpu: Optional[torch.Tensor] = None

        # Dense layers (0-2) of MiniMax-M3 run the shared dense oracle path in
        # the model, which reads this attachment (metadata + per-new-token
        # cache slots). It is the same contract the Triton reference builds; the
        # sparse layers ignore it and use the msa_* fields above.
        minimax_m3: Optional[dict] = None

        # Prebuilt graph-safe decode plans (set by the sparse layer's
        # build_decode_plans during prepare(); None for prefill/mixed batches,
        # which run eagerly). The forward reads these instead of planning inside
        # the captured region.
        msa_decode_proxy_plan: Optional[tuple] = None
        msa_decode_gqa_plan: Optional[tuple] = None
        msa_max_score: Optional[torch.Tensor] = None
        msa_n_valid_blocks: Optional[torch.Tensor] = None

        # Persistent backing storage for the fields above. Declared so they are
        # never attached dynamically; populated in _create_msa_buffers().
        _msa_buffers_ready: bool = False
        _msa_out_cache_loc_buf: Optional[torch.Tensor] = None
        _msa_kv_indices_buf: Optional[torch.Tensor] = None
        _msa_kv_lens_cpu_buf: Optional[torch.Tensor] = None
        _msa_qo_lens_cpu_buf: Optional[torch.Tensor] = None
        _msa_qo_offset_cpu_buf: Optional[torch.Tensor] = None
        # Lazily created by the sparse layer's build_decode_plans (needs layer
        # geometry not visible here), then reused across steps.
        _msa_proxy_plan: Optional["_MsaGraphSafePlan"] = None
        _msa_gqa_plan: Optional["_MsaGraphSafePlan"] = None
        _msa_max_score_buf: Optional[torch.Tensor] = None
        _msa_n_valid_blocks_buf: Optional[torch.Tensor] = None
        # Persistent staging buffers for the dense oracle attachment (m3_meta +
        # out_cache_loc). Under CUDA graph these keep the dense path's tensor
        # addresses stable across replays, matching the Triton reference. One
        # dict per metadata clone (create_cuda_graph_metadata clones per batch
        # size); it is not keyed by batch size.
        _m3_static_buffers: Optional[dict] = None

        def __post_init__(self) -> None:
            super().__post_init__()
            self._create_msa_buffers()

        def _create_msa_buffers(self) -> None:
            """Allocate the CUDA-graph-stable MSA buffers (mirrors DSA's
            create_buffers_for_indexer).

            Device buffers come from the shared graph buffer pool so they are
            reserved under capture. Host buffers are pinned. Sizing follows the
            worst-case graph geometry: max_num_sequences requests, up to
            max_blocks_per_seq pages each, and max_num_tokens new tokens.
            """
            kv_cache_manager = self.kv_cache_manager
            self._msa_buffers_ready = False
            if kv_cache_manager is None or not hasattr(kv_cache_manager, "get_index_k_buffer"):
                return
            capture_graph = self.is_cuda_graph
            buffers = self.cuda_graph_buffers
            max_num_sequences = int(self.max_num_sequences)
            max_blocks_per_seq = int(kv_cache_manager.max_blocks_per_seq)
            max_total_pages = max_num_sequences * max_blocks_per_seq
            max_num_tokens = int(self.max_num_tokens)

            self._msa_out_cache_loc_buf = self.get_empty(
                buffers,
                (max_num_tokens,),
                cache_name="msa_out_cache_loc",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self._msa_kv_indices_buf = self.get_empty(
                buffers,
                (max_total_pages,),
                cache_name="msa_kv_indices",
                dtype=torch.int32,
                capture_graph=capture_graph,
            )
            self._msa_kv_lens_cpu_buf = torch.zeros(
                (max_num_sequences,), dtype=torch.int32, pin_memory=prefer_pinned()
            )
            self._msa_qo_lens_cpu_buf = torch.zeros(
                (max_num_sequences,), dtype=torch.int32, pin_memory=prefer_pinned()
            )
            self._msa_qo_offset_cpu_buf = torch.zeros(
                (max_num_sequences,), dtype=torch.int32, pin_memory=prefer_pinned()
            )
            self._msa_buffers_ready = True

        def prepare(self) -> None:
            super().prepare()
            self._build_msa_fields()
            self._build_decode_plans()

        def _build_decode_plans(self) -> None:
            """Build the graph-safe decode plans for this step, outside capture.

            The plan is layer-invariant for MiniMax-M3, so it is built once per
            step by the first sparse layer (which owns the geometry) and shared
            by every layer. Prefill/mixed batches leave the plans as None and run
            eagerly. Mirrors DSA's Indexer.prepare(metadata=self).
            """
            self.msa_decode_proxy_plan = None
            self.msa_decode_gqa_plan = None
            self.msa_max_score = None
            self.msa_n_valid_blocks = None
            if self.msa_qo_lens_cpu is None:
                return
            # A decode batch is pure generation (no context requests).
            if int(self.num_contexts or 0) > 0:
                return
            # Planning does host->device work and cannot run under capture.
            if torch.cuda.is_current_stream_capturing():
                return
            kv_cache_manager = self.kv_cache_manager
            sparse_layer_ids = getattr(kv_cache_manager, "sparse_layer_ids", None)
            if not sparse_layer_ids:
                return
            layer = _lookup_msa_attention_layer(int(sparse_layer_ids[0]))
            if layer is None:
                return
            layer.build_decode_plans(self)

        def _maybe_get_m3_static_buffers(self, cache_device) -> Optional[dict]:
            """Return persistent staging buffers for the dense oracle attachment.

            Used only to keep the dense path's m3_meta / out_cache_loc addresses
            stable under CUDA graph. Returns None in eager mode (fresh per-call
            tensors are fine there). The first graph-mode call installs a
            placeholder dict; build_runtime_metadata_from_kv_manager allocates
            the persistent tensors lazily on first use and refreshes them in
            place afterwards.
            """
            need_static = bool(self.is_cuda_graph) or self._m3_static_buffers is not None
            if not need_static:
                return None
            if self._m3_static_buffers is not None and (
                self._m3_static_buffers.get("device") == cache_device
            ):
                return self._m3_static_buffers
            max_num_sequences = int(
                getattr(self, "max_num_sequences", None) or self.max_num_requests
            )
            max_num_tokens = int(getattr(self, "max_num_tokens", None) or max_num_sequences)
            self._m3_static_buffers = {
                "device": cache_device,
                "max_num_sequences_hint": max_num_sequences,
                "max_num_tokens_hint": max_num_tokens,
            }
            return self._m3_static_buffers

        def _build_msa_fields(self) -> None:
            """Populate the MSA buffers for this step and expose sliced views.

            The page table and per-new-token cache slots are derived via the
            tested build_runtime_metadata_from_kv_manager path, then copied into
            the persistent buffers. Only the sliced views are exposed as the
            public msa_* fields; the transient builder result is discarded.
            """
            self.msa_out_cache_loc = None
            self.msa_kv_indices = None
            self.msa_kv_lens_cpu = None
            self.msa_qo_lens_cpu = None
            self.msa_qo_offset_cpu = None
            self.minimax_m3 = None
            if not self._msa_buffers_ready:
                return
            request_ids = self.request_ids
            seq_lens = self.seq_lens
            if request_ids is None or seq_lens is None:
                return
            batch_size = int(seq_lens.shape[0])
            if batch_size == 0:
                return

            kv_cache_manager = self.kv_cache_manager
            num_contexts = int(self.num_contexts or 0)
            cache_device = _cache_device(self)
            page_size = int(kv_cache_manager.tokens_per_block)

            # AttentionMetadata.seq_lens is the host tensor of per-request
            # query lengths; keep a CPU copy for the sync-free length math.
            seq_lens_cpu = seq_lens if seq_lens.device.type == "cpu" else seq_lens.cpu()

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

            # Under CUDA graph the dense oracle attachment (m3_meta +
            # out_cache_loc) must keep stable tensor addresses across replays,
            # so build it into persistent staging buffers, matching the Triton
            # reference. The sparse path copies what it needs into its own
            # buffers below, so this only affects the dense attachment.
            static_buffers = self._maybe_get_m3_static_buffers(cache_device)

            if is_prefill:
                prefix_lens_list = [int(num_cached[b]) for b in range(batch_size)]
                qo_lens_list = [kv_lens_list[b] - prefix_lens_list[b] for b in range(batch_size)]
                prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, device=cache_device)
                m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                    kv_cache_manager=kv_cache_manager,
                    request_ids=request_ids,
                    seq_lens=kv_lens_dev,
                    seq_lens_cpu=kv_lens_cpu,
                    is_prefill=True,
                    prefix_lens=prefix_lens,
                    extend_seq_lens_cpu=qo_lens_list,
                    device=cache_device,
                    static_buffers=static_buffers,
                )
            else:
                qo_lens_list = [1] * batch_size
                m3_meta, out_cache_loc = build_runtime_metadata_from_kv_manager(
                    kv_cache_manager=kv_cache_manager,
                    request_ids=request_ids,
                    seq_lens=kv_lens_dev,
                    seq_lens_cpu=kv_lens_cpu,
                    is_prefill=False,
                    device=cache_device,
                    static_buffers=static_buffers,
                )

            qo_lens_cpu = torch.tensor(qo_lens_list, dtype=torch.int32)
            qo_offset_cpu = kv_lens_cpu - qo_lens_cpu
            kv_indices = build_kv_page_indices(
                m3_meta.req_to_token, m3_meta.slot_ids, kv_lens_cpu, page_size
            )

            # Dense layers reuse the shared dense oracle path, which reads the
            # freshly built metadata and cache slots off this attachment.
            self.minimax_m3 = {"metadata": m3_meta, "out_cache_loc": out_cache_loc}

            total_new_tokens = int(out_cache_loc.shape[0])
            total_pages = int(kv_indices.shape[0])
            if total_new_tokens > self._msa_out_cache_loc_buf.shape[0]:
                raise ValueError(
                    f"MSA out_cache_loc buffer ({self._msa_out_cache_loc_buf.shape[0]}) is "
                    f"smaller than the step's new-token count ({total_new_tokens})."
                )
            if total_pages > self._msa_kv_indices_buf.shape[0]:
                raise ValueError(
                    f"MSA kv_indices buffer ({self._msa_kv_indices_buf.shape[0]}) is "
                    f"smaller than the step's page count ({total_pages})."
                )

            self._msa_out_cache_loc_buf[:total_new_tokens].copy_(out_cache_loc, non_blocking=True)
            self._msa_kv_indices_buf[:total_pages].copy_(kv_indices, non_blocking=True)
            self._msa_kv_lens_cpu_buf[:batch_size].copy_(kv_lens_cpu)
            self._msa_qo_lens_cpu_buf[:batch_size].copy_(qo_lens_cpu)
            self._msa_qo_offset_cpu_buf[:batch_size].copy_(qo_offset_cpu)

            self.msa_out_cache_loc = self._msa_out_cache_loc_buf[:total_new_tokens]
            self.msa_kv_indices = self._msa_kv_indices_buf[:total_pages]
            self.msa_kv_lens_cpu = self._msa_kv_lens_cpu_buf[:batch_size]
            self.msa_qo_lens_cpu = self._msa_qo_lens_cpu_buf[:batch_size]
            self.msa_qo_offset_cpu = self._msa_qo_offset_cpu_buf[:batch_size]

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

        def build_decode_plans(self, metadata) -> None:
            """Build the graph-safe decode plans and buffers on the metadata.

            Invoked from MsaMinimaxM3AttentionMetadata.prepare() (outside CUDA
            graph capture) by the first sparse layer, since the plan is
            layer-invariant. Uses this layer's geometry plus the metadata's
            per-request lengths to build the proxy max-score plan and the sparse
            GQA plan, mirror them into CUDA-graph-stable buffers, and precompute
            the per-query valid-block count. Mimics FlashInfer's plan() split.
            """
            fmha_sm100 = require_msa_module()
            config = self.m3_config
            qo_lens_cpu = metadata.msa_qo_lens_cpu
            kv_lens_cpu = metadata.msa_kv_lens_cpu
            qo_offset_cpu = metadata.msa_qo_offset_cpu
            if qo_lens_cpu is None or kv_lens_cpu is None or qo_offset_cpu is None:
                return
            batch = int(qo_lens_cpu.shape[0])
            device = _cache_device(metadata)
            page_size = int(metadata.kv_cache_manager.tokens_per_block)
            capture_graph = metadata.is_cuda_graph
            max_batch = int(metadata.max_num_sequences)

            proxy_plan = fmha_sm100.fmha_sm100_plan(
                qo_lens_cpu,
                kv_lens_cpu,
                config.num_index_heads,
                num_kv_heads=1,
                qo_offset=qo_offset_cpu,
                page_size=page_size,
                output_maxscore=True,
                num_kv_splits=1,
                causal=True,
            )
            gqa_plan = fmha_sm100.fmha_sm100_plan(
                qo_lens_cpu,
                kv_lens_cpu,
                config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                qo_offset=qo_offset_cpu,
                page_size=page_size,
                kv_block_num=config.topk,
                num_kv_splits=1,
                causal=True,
            )

            if metadata._msa_proxy_plan is None:
                num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
                metadata._msa_proxy_plan = _MsaGraphSafePlan(
                    metadata,
                    "msa_proxy_plan",
                    max_batch=max_batch,
                    num_ctas=num_ctas,
                    capture_graph=capture_graph,
                )
                metadata._msa_gqa_plan = _MsaGraphSafePlan(
                    metadata,
                    "msa_gqa_plan",
                    max_batch=max_batch,
                    num_ctas=num_ctas,
                    capture_graph=capture_graph,
                )
                # max_k_tiles is constant over the decode kv-length range, so the
                # proxy max_score buffer keeps a stable shape across replays.
                max_k_tiles = int(proxy_plan[3]["max_k_tiles"])
                metadata._msa_max_score_buf = metadata.get_empty(
                    metadata.cuda_graph_buffers,
                    (config.num_index_heads, max_k_tiles, max_batch),
                    cache_name="msa_max_score",
                    dtype=torch.float32,
                    capture_graph=capture_graph,
                )
                metadata._msa_n_valid_blocks_buf = metadata.get_empty(
                    metadata.cuda_graph_buffers,
                    (max_batch,),
                    cache_name="msa_n_valid_blocks",
                    dtype=torch.int32,
                    capture_graph=capture_graph,
                )

            metadata.msa_decode_proxy_plan = metadata._msa_proxy_plan.refresh(proxy_plan)
            metadata.msa_decode_gqa_plan = metadata._msa_gqa_plan.refresh(gqa_plan)
            metadata.msa_max_score = metadata._msa_max_score_buf[:, :, :batch]

            n_valid = per_token_valid_blocks(
                qo_lens_cpu, kv_lens_cpu, qo_offset_cpu, causal=True, block_size=page_size
            )
            metadata._msa_n_valid_blocks_buf[:batch].copy_(
                n_valid.to(torch.int32), non_blocking=True
            )
            metadata.msa_n_valid_blocks = metadata._msa_n_valid_blocks_buf[:batch]

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
            Decode uses the prebuilt graph-safe proxy plan; prefill plans
            eagerly.
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

            if metadata.msa_decode_proxy_plan is not None:
                return self.indexer.select_blocks_decode(
                    idx_q_view,
                    idx_k_cache,
                    proxy_plan=metadata.msa_decode_proxy_plan,
                    kv_indices=metadata._msa_kv_indices_buf,
                    max_score=metadata.msa_max_score,
                    n_valid_blocks=metadata.msa_n_valid_blocks,
                    idx_sm_scale=idx_sm_scale,
                )
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
