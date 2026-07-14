# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the MiniMax-M3 MSA sparse attention backend.

These validate backend selection and decode scratch-buffer sizing without
launching kernels. Numerical parity against the Triton reference is covered
by the SM100 integration accuracy test.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_resolver_selects_msa_backend_when_available(monkeypatch):
    import tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability as avail

    monkeypatch.setattr(avail, "ensure_msa_available", lambda: None)
    params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True).to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is MiniMaxM3MsaSparseAttention


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
