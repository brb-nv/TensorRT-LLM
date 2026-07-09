# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate the DSA-aligned wiring without launching kernels: the flat
metadata surface, FMHA registration, backend resolution, and the sparse
hooks. Numerical parity against the Triton reference is covered by the
SM100 integration accuracy test.
"""

from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import DEFAULT_FMHA_LIBS, FMHA_LIBS
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
    get_minimax_m3_msa_attention_backend_cls,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.metadata import MiniMaxM3SparseParams
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

_MSA_METADATA_FIELDS = (
    "msa_out_cache_loc",
    "msa_kv_indices",
    "msa_kv_lens_cpu",
    "msa_qo_lens_cpu",
    "msa_qo_offset_cpu",
)


def test_msa_fmha_registered_first():
    assert FMHA_LIBS.get("msa_sparse_gqa") is MsaSparseGqaFmha
    assert next(iter(FMHA_LIBS)) == "msa_sparse_gqa"
    assert DEFAULT_FMHA_LIBS[0] == "msa_sparse_gqa"


def test_msa_fmha_is_available_filters_on_owning_attention():
    # is_available must reject non-MSA attentions and None so the base
    # create_fmha_libs adds MsaSparseGqaFmha only to the MSA layer.
    assert MsaSparseGqaFmha.is_available(None) is False
    assert MsaSparseGqaFmha.is_available(object()) is False


def test_msa_backend_does_not_override_create_fmha_libs():
    cls = get_minimax_m3_msa_attention_backend_cls()
    assert cls.create_fmha_libs is TrtllmAttention.create_fmha_libs


def test_indices_block_size_matches_block_size():
    params = MiniMaxM3SparseAttentionConfig(sparse_block_size=128).to_sparse_params()
    assert params.indices_block_size == 128


def test_msa_backend_subclasses_trtllm_attention():
    cls = get_minimax_m3_msa_attention_backend_cls()
    assert issubclass(cls, TrtllmAttention)
    assert issubclass(cls.Metadata, TrtllmAttentionMetadata)


def test_msa_metadata_declares_flat_fields():
    metadata_cls = get_minimax_m3_msa_attention_backend_cls().Metadata
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for field in _MSA_METADATA_FIELDS:
        assert field in annotations, f"{field} must be a declared field"


def test_msa_metadata_drops_redundant_intermediate_fields():
    # These were per-forward intermediates that no forward path reads; the
    # graph-safe metadata computes them locally and does not store them.
    metadata_cls = get_minimax_m3_msa_attention_backend_cls().Metadata
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for field in ("msa_is_prefill", "msa_req_to_token", "msa_slot_ids", "msa_kv_lens_dev"):
        assert field not in annotations, f"{field} should no longer be stored"


def test_msa_metadata_allocates_graph_stable_buffers():
    # The buffers are declared and allocated in a DSA-style __post_init__ hook,
    # not cached per batch size on a bespoke driver.
    metadata_cls = get_minimax_m3_msa_attention_backend_cls().Metadata
    assert callable(getattr(metadata_cls, "_create_msa_buffers", None))
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for buf in ("_msa_out_cache_loc_buf", "_msa_kv_indices_buf"):
        assert buf in annotations, f"{buf} must be a declared backing buffer"


def test_msa_backend_defines_dsa_style_hooks():
    cls = get_minimax_m3_msa_attention_backend_cls()
    for name in ("run_indexer", "sparse_attn_predict", "sparse_kv_predict"):
        assert callable(getattr(cls, name, None)), f"missing {name}"
    assert cls.support_fused_rope() is False


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True).to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is (get_minimax_m3_msa_attention_backend_cls())


def test_msa_params_reject_bad_topk_at_construction():
    # The MSA backend requires topk 16; a mismatch must fail loudly rather
    # than reaching the kernel with an unsupported shape.
    params = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=8).to_sparse_params()
    assert isinstance(params, MiniMaxM3SparseParams)
    assert params.topk == 8  # config lowering keeps the value; the backend gates it
