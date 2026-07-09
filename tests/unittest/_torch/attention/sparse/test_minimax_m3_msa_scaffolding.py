# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-plumbing tests for MiniMax-M3 MSA backend selection.

These cover the scaffolding only: the user-facing sparse_use_msa flag, its
lowering into MiniMaxM3SparseParams, and the backend resolver's gating. They
do not launch kernels and run without an SM100 GPU.
"""

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import (
    get_minimax_m3_msa_attention_backend_cls,
    get_minimax_m3_triton_attention_backend_cls,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseParams
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_availability import (
    is_msa_available,
)
from tensorrt_llm._torch.attention_backend.sparse.utils import _resolve_minimax_m3_backend_cls
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig


def test_use_msa_defaults_to_false():
    config = MiniMaxM3SparseAttentionConfig()
    assert config.sparse_use_msa is False
    assert config.to_sparse_params().use_msa is False


def test_use_msa_lowers_into_params():
    config = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True)
    params = config.to_sparse_params()
    assert isinstance(params, MiniMaxM3SparseParams)
    assert params.use_msa is True


def test_resolver_returns_triton_backend_when_msa_disabled():
    params = MiniMaxM3SparseAttentionConfig().to_sparse_params()
    assert _resolve_minimax_m3_backend_cls(params) is (
        get_minimax_m3_triton_attention_backend_cls()
    )


def test_resolver_does_not_fall_back_to_triton_when_msa_requested():
    params = MiniMaxM3SparseAttentionConfig(sparse_use_msa=True).to_sparse_params()
    try:
        resolved = _resolve_minimax_m3_backend_cls(params)
    except RuntimeError:
        return
    assert resolved is get_minimax_m3_msa_attention_backend_cls()
    assert resolved is not get_minimax_m3_triton_attention_backend_cls()


def test_is_msa_available_returns_bool():
    assert isinstance(is_msa_available(), bool)
