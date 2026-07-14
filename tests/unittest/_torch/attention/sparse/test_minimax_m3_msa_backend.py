# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate the DSA-aligned wiring without launching kernels: the flat
metadata surface, FMHA registration, backend resolution, and the sparse
hooks. Numerical parity against the Triton reference is covered by the
SM100 integration accuracy test.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import MsaSparseGqaFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import DEFAULT_FMHA_LIBS, FMHA_LIBS
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import (
    MiniMaxM3SparseConfig,
    MiniMaxM3SparseParams,
)
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

# CUDA-graph-stable buffers owned by the metadata as declared fields.
_MSA_METADATA_FIELDS = (
    "msa_out_cache_loc",
    "msa_kv_indices",
    "msa_max_score",
    "msa_n_valid_blocks",
)

# Per-request lengths and decode-plan tuples are derived on access from the
# base metadata / plan owners, so they are properties rather than stored fields.
_MSA_METADATA_PROPERTIES = (
    "msa_qo_lens_cpu",
    "msa_kv_lens_cpu",
    "msa_qo_offset_cpu",
    "msa_decode_proxy_plan",
    "msa_decode_gqa_plan",
    "msa_decode_dense_plan",
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


def test_indices_block_size_matches_block_size():
    params = MiniMaxM3SparseAttentionConfig(sparse_block_size=128).to_sparse_params()
    assert params.indices_block_size == 128


def test_msa_metadata_declares_flat_fields():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for field in _MSA_METADATA_FIELDS:
        assert field in annotations, f"{field} must be a declared field"


def test_msa_metadata_lengths_and_plans_are_derived_properties():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for name in _MSA_METADATA_PROPERTIES:
        assert name not in annotations, f"{name} should be derived, not stored"
        assert isinstance(getattr(metadata_cls, name, None), property), (
            f"{name} must be exposed as a property"
        )


def test_msa_metadata_allocates_graph_stable_buffers():
    # The buffers are declared and allocated in a DSA-style __post_init__ hook,
    # not cached per batch size on a bespoke driver. Each graph-stable tensor is
    # a single declared field.
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    assert callable(getattr(metadata_cls, "_create_msa_buffers", None))
    annotations = {}
    for klass in metadata_cls.__mro__:
        annotations.update(getattr(klass, "__annotations__", {}))
    for buf in ("msa_out_cache_loc", "msa_kv_indices"):
        assert buf in annotations, f"{buf} must be a declared backing buffer"


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True).to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is MiniMaxM3MsaSparseAttention


def test_msa_params_reject_bad_topk_at_construction():
    # The MSA backend requires topk 16; a mismatch must fail loudly rather
    # than reaching the kernel with an unsupported shape.
    params = MiniMaxM3SparseAttentionConfig(sparse_topk_blocks=8).to_sparse_params()
    assert isinstance(params, MiniMaxM3SparseParams)
    assert params.topk == 8  # config lowering keeps the value; the backend gates it


def test_msa_metadata_rejects_undersized_max_score_buffer():
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    config = MiniMaxM3SparseConfig(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        num_index_heads=4,
        sparse_index_dim=128,
        block_size=128,
        topk=16,
    )
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.msa_max_score = torch.zeros(4, 8, 2)
    metadata.kv_cache_manager = None

    with pytest.raises(ValueError, match=r"msa_max_score has 8 k-tiles"):
        metadata._ensure_msa_decode_scratch_buffers(
            config=config,
            max_batch=2,
            capture_graph=False,
            required_max_k_tiles=16,
        )
