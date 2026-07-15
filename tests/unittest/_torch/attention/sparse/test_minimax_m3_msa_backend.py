# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural and top-k tests for the MiniMax-M3 MSA sparse attention backend.

The structural tests validate backend selection and decode scratch-buffer
sizing without launching kernels. The top-k test checks the Triton block
selector against a reference PyTorch implementation and needs a CUDA device.
End-to-end numerical parity is covered by the SM100 integration accuracy test.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    select_blocks_from_maxscore,
)
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


def _reference_select_blocks(
    max_score_kv: torch.Tensor,
    *,
    topk: int,
    n_valid_blocks: torch.Tensor,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """PyTorch reference for the block selector.

    Mirrors the forcing and masking rules the Triton kernel implements: init
    and local blocks get sentinel scores, blocks beyond a query's valid extent
    are masked out, and the top-k blocks are selected per query and KV head.
    Returns [total_q, num_kv_heads, topk] block ids with -1 padding, unsorted.
    """
    num_kv_heads, n_blocks, total_q = max_score_kv.shape
    device = max_score_kv.device
    scores = max_score_kv.permute(2, 0, 1).to(torch.float32).clone()
    block_ids = torch.arange(n_blocks, device=device, dtype=torch.long)
    nvb = n_valid_blocks.to(device=device, dtype=torch.long)

    if init_blocks > 0:
        init_mask = block_ids.view(1, 1, -1) < init_blocks
        scores = torch.where(init_mask, torch.full_like(scores, 1e30), scores)
    if local_blocks > 0:
        local_start = (nvb - local_blocks).clamp_min(0)
        local_mask = (block_ids.view(1, -1) >= local_start.view(-1, 1)) & (
            block_ids.view(1, -1) < nvb.view(-1, 1)
        )
        scores = torch.where(local_mask.unsqueeze(1), torch.full_like(scores, 1e29), scores)
    block_valid = block_ids.view(1, -1) < nvb.view(-1, 1)
    scores = scores.masked_fill(~block_valid.unsqueeze(1), float("-inf"))

    k = min(topk, n_blocks)
    vals, idx = scores.topk(k=k, dim=-1)
    idx = torch.where(vals != float("-inf"), idx, torch.full_like(idx, -1))
    if k < topk:
        pad = torch.full((total_q, num_kv_heads, topk - k), -1, dtype=idx.dtype, device=device)
        idx = torch.cat([idx, pad], dim=-1)
    return idx.to(torch.int32)


def _selected_sets(indices: torch.Tensor):
    """Per (query, KV head) set of selected block ids, dropping -1 padding."""
    total_q, num_kv_heads, _ = indices.shape
    rows = indices.tolist()
    return [
        [set(b for b in rows[q][h] if b >= 0) for h in range(num_kv_heads)] for q in range(total_q)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a CUDA device")
@pytest.mark.parametrize(
    "num_kv_heads,n_blocks,total_q,topk,init_blocks,local_blocks",
    [
        (4, 40, 37, 16, 0, 1),  # default MiniMax-M3 forcing
        (1, 40, 8, 16, 0, 1),  # single KV head, decode-like
        (2, 64, 20, 16, 2, 2),  # non-trivial init and local
        (4, 8, 12, 16, 0, 1),  # fewer blocks than topk
        (2, 3000, 5, 16, 0, 1),  # streaming across several BLOCK_SIZE_K chunks
        (4, 40, 16, 16, 0, 0),  # no forced blocks
    ],
)
def test_triton_select_blocks_matches_reference(
    num_kv_heads, n_blocks, total_q, topk, init_blocks, local_blocks
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    max_score_kv = torch.randn(num_kv_heads, n_blocks, total_q, device=device, dtype=torch.float32)
    n_valid_blocks = torch.randint(1, n_blocks + 1, (total_q,), device=device, dtype=torch.int32)

    out = select_blocks_from_maxscore(
        max_score_kv,
        topk=topk,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    ref = _reference_select_blocks(
        max_score_kv,
        topk=topk,
        n_valid_blocks=n_valid_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )

    assert out.shape == (total_q, num_kv_heads, topk)
    assert out.dtype == torch.int32

    nvb = n_valid_blocks.tolist()
    for q in range(total_q):
        expected_valid = min(topk, nvb[q])
        for h in range(num_kv_heads):
            row = out[q, h].tolist()
            valid = [b for b in row if b >= 0]
            assert len(valid) == expected_valid
            assert all(0 <= b < nvb[q] for b in valid)
            assert len(set(valid)) == len(valid)

    # Order within a query row is irrelevant to the downstream gather, so
    # compare the selected block ids as sets.
    assert _selected_sets(out) == _selected_sets(ref)
