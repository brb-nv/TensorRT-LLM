# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 index-K (indexer_kv_dtype='nvfp4') and the MSA-v1 Q8KV4 decode indexer that scores it.

Covers the page layout, the context-row dequantization the fmha_sm100 proxy reads, the step
staging prepare() does for it, the Q8KV4 selection contract and its agreement with the E4M3
CuTe DSL scorer over the same (dequantized) keys, CUDA graph replay, the cache manager
geometry, and the config validation. Q8KV4 tests skip unless MiniMax MSA is importable.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_v1_indexer
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttention,
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import (
    MsaIndexer,
    _cutedsl_score,
)
from tensorrt_llm._utils import get_sm_version

PAGE = 128
D = 128
BYTES_PER_TOKEN = msa_v1_indexer.NVFP4_INDEX_K_BYTES_PER_TOKEN

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="NVFP4 index-K requires SM100/SM103.",
)


@pytest.fixture
def q8kv4():
    if not msa_v1_indexer.q8kv4_available():
        pytest.skip("MiniMax MSA (inference.msa_v1) is not importable.")


def _reference_dequant(data, scale):
    """E2M1 x E4M3 scale in FP32: [..., 64] uint8, [..., 8] E4M3 -> [..., 128]."""
    table = torch.tensor(E2M1, device=data.device)
    values = torch.stack([table[(data & 0xF).long()], table[(data >> 4).long()]], -1).flatten(-2)
    return values * scale.float().repeat_interleave(16, dim=-1)


def _nvfp4_pool(num_pages, *, structured, gen):
    """A raw NVFP4 index-K pool quantized from normalized keys as the fused producer does."""
    raw = torch.zeros(num_pages, 1, PAGE, BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda")
    data, scale = msa_v1_indexer.split_nvfp4_index_k(raw)
    k = torch.randn(num_pages, PAGE, D, device="cuda", generator=gen)
    if structured:
        # Pages share a direction, as normalized index keys of related tokens do.
        center = torch.randn(num_pages, 1, D, device="cuda", generator=gen)
        k = 0.8 * center + 0.6 * k
    k = k / k.norm(dim=-1, keepdim=True) * math.sqrt(D) * 0.5
    packed, scales = torch.ops.trtllm.fp4_quantize(
        k.reshape(-1, D).to(torch.bfloat16), torch.ones(1, device="cuda"), 16, False, False
    )
    data.copy_(packed.view(torch.uint8).view(num_pages, PAGE, D // 2))
    scale.view(torch.uint8).copy_(scales.view(torch.uint8).view(num_pages, PAGE, D // 16))
    return raw, data, scale


@blackwell
def test_split_views_carve_rows_then_scales_out_of_each_page():
    raw = torch.zeros(3, 1, PAGE, BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda")
    data, scale = msa_v1_indexer.split_nvfp4_index_k(raw)
    assert tuple(data.shape) == (3, PAGE, D // 2) and tuple(scale.shape) == (3, PAGE, D // 16)
    assert data.stride(0) == scale.stride(0) == PAGE * BYTES_PER_TOKEN
    data[1, 5, 7] = 0xAB
    scale.view(torch.uint8)[2, 9, 3] = 0x38
    flat = raw.view(3, -1)
    assert int(flat[1, 5 * 64 + 7]) == 0xAB
    assert int(flat[2, PAGE * 64 + 9 * 8 + 3]) == 0x38
    assert int(torch.count_nonzero(flat)) == 2


@blackwell
def test_dequantize_pages_matches_reference_in_listed_order():
    gen = torch.Generator(device="cuda").manual_seed(0)
    num_pages = 9
    raw = torch.randint(
        0,
        256,
        (num_pages, 1, PAGE, BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    data, scale = msa_v1_indexer.split_nvfp4_index_k(raw)
    # Positive normal E4M3 scales, some large enough that 6 x scale saturates E4M3.
    scale.copy_(
        (torch.rand(scale.shape, device="cuda", generator=gen) * 100 + 0.02).to(torch.float8_e4m3fn)
    )
    src = torch.tensor([7, 2, 2, 0, 8], dtype=torch.int32, device="cuda")
    workspace = msa_v1_indexer.DequantWorkspace()
    out = workspace.pages(int(src.shape[0]), raw.device)
    got = msa_v1_indexer.dequantize_index_k_pages(data, scale, src, out)
    assert tuple(got.shape) == (5, 1, PAGE, D) and got.dtype == torch.float8_e4m3fn
    expected = (
        _reference_dequant(data[src.long()], scale[src.long()])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    assert torch.equal(got[:, 0].view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(
        workspace.identity(5, raw.device)[:5].cpu(), torch.arange(5, dtype=torch.int32)
    )


def _step_metadata(*, num_contexts, qo_lens, kv_lens, device):
    metadata_cls = MiniMaxM3MsaSparseAttention.Metadata
    metadata = metadata_cls.__new__(metadata_cls)
    metadata.mapping = None
    # Set before the decode span, which bounds generation rows by 1 + max_total_draft_tokens.
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=PAGE,
        dtype=None,
        indexer_kv_dtype="nvfp4",
        index_k_is_nvfp4=True,
        max_total_draft_tokens=7,
    )
    metadata._seq_lens = torch.tensor(qo_lens, dtype=torch.int32)
    metadata.num_contexts = num_contexts
    metadata.kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    metadata.is_cuda_graph = False
    metadata._stage_host_lengths()
    metadata._set_decode_span()
    metadata._msa_fields_ready = True
    metadata._msa_params = SimpleNamespace(
        sharded_index_head_count=lambda mapping: 2,
        sharded_head_counts=lambda mapping: (32, 2),
        topk=16,
    )
    metadata.msa_policy = SimpleNamespace(short_sequence_identity=False)
    num_pages = sum((k + PAGE - 1) // PAGE for k in kv_lens)
    metadata.msa_kv_indices = torch.arange(
        1000, 1000 + 4 * num_pages, dtype=torch.int32, device=device
    )
    metadata.msa_block_table = torch.zeros(8, 64, dtype=torch.int32, device=device)
    metadata.msa_seq_lens_cuda = torch.zeros(8, dtype=torch.int32, device=device)
    metadata.fmha_plan_caches = {}
    return metadata


@blackwell
@pytest.mark.parametrize("segmented", [False, True], ids=["unsplit", "kv-split"])
def test_step_plans_stage_the_context_pages_and_the_generation_plan(monkeypatch, segmented):
    """Context rows read a compact E4M3 copy of their own pages; generation rows get a plan."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend, msa_indexer

    def fmha_sm100_plan(qo_lens, kv_lens, **kwargs):
        return dict(kwargs, qo_lens=qo_lens.tolist(), kv_lens=kv_lens.tolist())

    fake = SimpleNamespace(fmha_sm100_plan=fmha_sm100_plan)
    monkeypatch.setattr(msa_backend, "require_msa_module", lambda: fake)
    monkeypatch.setattr(msa_backend, "_cache_device", lambda metadata: torch.device("cuda"))
    planned = []

    def get_or_plan(cache, *, block_table, seq_lens, query_len, num_heads):
        planned.append((block_table.data_ptr(), int(block_table.shape[0]), query_len, num_heads))
        return "plan"

    monkeypatch.setattr(msa_v1_indexer, "get_or_plan", get_or_plan)
    monkeypatch.setattr(msa_v1_indexer, "q8kv4_supported", lambda **kwargs: True)
    # One context row (300-token prompt, or a 200-token chunk over a 9800-token prefix, which
    # the proxy plans as KV-split segments) ahead of two verify rows of 8 tokens.
    qo_lens = (200, 8, 8) if segmented else (300, 8, 8)
    kv_lens = (10_000, 1000, 2000) if segmented else (300, 1000, 2000)
    metadata = _step_metadata(num_contexts=1, qo_lens=qo_lens, kv_lens=kv_lens, device="cuda")

    metadata._build_step_plans()

    ctx_pages = (kv_lens[0] + PAGE - 1) // PAGE
    assert torch.equal(metadata._msa_idx_k_dequant_src, metadata.msa_kv_indices[:ctx_pages])
    proxy_indices = metadata.msa_prefill_proxy_kv_indices
    if segmented:
        # Same gather the pool-addressed split plan does, expressed in workspace pages.
        _, pool_indices = msa_indexer._segmented_proxy_plan(
            fake,
            [qo_lens[0]],
            [kv_lens[0] - qo_lens[0]],
            num_index_heads=2,
            page_size=PAGE,
            num_kv_splits=metadata.msa_prefill_proxy_plan["num_kv_splits"],
            kv_indices=metadata.msa_kv_indices,
        )
        assert torch.equal(metadata._msa_idx_k_dequant_src[proxy_indices.long()], pool_indices)
    else:
        assert torch.equal(proxy_indices.cpu(), torch.arange(ctx_pages, dtype=torch.int32))
    assert planned == [(metadata.msa_block_table[1:3].data_ptr(), 2, 8, 2)]
    assert metadata.msa_q8kv4_plan == "plan"


def _decode_case(batch, q_len, heads, ctx, *, structured, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages_per_req = -(-ctx // PAGE)
    num_pages = batch * pages_per_req + 2
    raw, data, scale = _nvfp4_pool(num_pages, structured=structured, gen=gen)
    block_table = (
        torch.randperm(num_pages, device="cuda", generator=gen)[: batch * pages_per_req]
        .to(torch.int32)
        .view(batch, pages_per_req)
        .contiguous()
    )
    seq_lens = torch.randint(
        ctx // 2, ctx + 1, (batch,), device="cuda", generator=gen, dtype=torch.int32
    )
    idx_q = torch.randn(batch * q_len, heads, D, device="cuda", generator=gen).to(
        torch.float8_e4m3fn
    )
    return data, scale, block_table, seq_lens, idx_q


def _e4m3_reference(data, scale, idx_q, block_table, seq_lens, q_len, heads):
    """CuTe DSL scorer + select over the E4M3 dequantization of the same NVFP4 keys."""
    workspace = msa_v1_indexer.DequantWorkspace()
    num_pages = int(data.shape[0])
    src = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    fp8 = msa_v1_indexer.dequantize_index_k_pages(
        data, scale, src, workspace.pages(num_pages, src.device)
    )
    cfg = MiniMaxM3SparseConfig(
        num_q_heads=16 * heads,
        num_kv_heads=heads,
        head_dim=D,
        num_index_heads=heads,
        sparse_index_dim=D,
        block_size=PAGE,
        topk=16,
    )
    total_q = idx_q.shape[0]
    max_score = torch.empty(heads, block_table.shape[1], total_q, device="cuda")
    assert _cutedsl_score(
        idx_q,
        fp8,
        max_score,
        block_table=block_table,
        seq_lens_cuda=seq_lens,
        decode_query_len=q_len,
    )
    qi = torch.arange(q_len, device="cuda", dtype=torch.int32)
    n_valid = ((seq_lens[:, None] - q_len + qi[None]) // PAGE + 1).reshape(-1).to(torch.int32)
    return MsaIndexer(cfg)._select(max_score, n_valid)


def _q8kv4(data, scale, idx_q, block_table, seq_lens, q_len, heads, plan_cache, layer_idx=0):
    plan = msa_v1_indexer.get_or_plan(
        plan_cache, block_table=block_table, seq_lens=seq_lens, query_len=q_len, num_heads=heads
    )
    return msa_v1_indexer.run_q8kv4(plan, idx_q, data, scale, layer_idx=layer_idx)


def _check_contract(table, seq_lens, q_len):
    """Ascending valid prefix, -1 tail, local page last, count = min(16, valid blocks)."""
    t = table.permute(1, 0, 2).cpu()  # [H, T, 16]
    for h in range(t.shape[0]):
        for row in range(t.shape[1]):
            r = t[h, row]
            valid = r[r >= 0]
            assert torch.all(r[: valid.numel()] >= 0) and torch.all(r[valid.numel() :] == -1)
            assert torch.all(valid[1:] > valid[:-1])
            b, qi = divmod(row, q_len)
            local = (int(seq_lens[b]) - q_len + qi) // PAGE
            assert int(valid[-1]) == local
            assert valid.numel() == min(16, local + 1)


@blackwell
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("q_len", [1, 8])
def test_q8kv4_selection_contract_and_agreement(q8kv4, heads, q_len):
    data, scale, bt, sl, idx_q = _decode_case(
        8, q_len, heads, 40_000, structured=True, seed=10 + q_len
    )
    ours = _e4m3_reference(data, scale, idx_q, bt, sl, q_len, heads)
    got = _q8kv4(data, scale, idx_q, bt, sl, q_len, heads, {})
    assert got.shape == ours.shape and got.dtype == torch.int32
    assert got.permute(1, 0, 2).is_contiguous()
    _check_contract(got, sl.cpu(), q_len)
    overlap = 0.0
    for h in range(heads):
        a, b = ours[:, h], got[:, h]
        hits = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
        overlap += (hits.any(-1).sum() / (a >= 0).sum()).item()
    overlap /= heads
    print(f"Q8KV4 vs E4M3 selection overlap heads={heads} q_len={q_len}: {overlap:.4f}")
    # Both score the same NVFP4 keys; only the E4M3 rounding of the dequantized keys differs.
    assert overlap > 0.95


@blackwell
def test_q8kv4_short_contexts_select_every_block(q8kv4):
    data, scale, bt, sl, idx_q = _decode_case(4, 8, 2, 1_500, structured=False, seed=3)
    ours = _e4m3_reference(data, scale, idx_q, bt, sl, 8, 2)
    got = _q8kv4(data, scale, idx_q, bt, sl, 8, 2, {})
    assert torch.equal(ours, got)


@blackwell
def test_q8kv4_cuda_graph_replay_tracks_lengths(q8kv4):
    data, scale, bt, sl, idx_q = _decode_case(4, 8, 2, 20_000, structured=True, seed=7)
    plan_cache = {}
    eager = _q8kv4(data, scale, idx_q, bt, sl, 8, 2, plan_cache).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _q8kv4(data, scale, idx_q, bt, sl, 8, 2, plan_cache)
    graph.replay()
    assert torch.equal(captured, eager)
    sl.sub_(5_000)
    graph.replay()
    expected = _q8kv4(data, scale, idx_q, bt, sl, 8, 2, {}).clone()
    assert torch.equal(captured, expected)
    _check_contract(captured, sl.cpu(), 8)


@blackwell
def test_q8kv4_plans_share_score_storage_and_refuse_capture_without_a_plan(q8kv4, monkeypatch):
    data, scale, bt, sl, idx_q = _decode_case(4, 8, 2, 5_000, structured=False, seed=5)
    # A fresh arena: the process-wide one may already hold buffers from earlier tests.
    monkeypatch.setattr(msa_v1_indexer, "_SCORE_ARENA", msa_v1_indexer._ScoreArena())
    cache = {}
    big = msa_v1_indexer.get_or_plan(cache, block_table=bt, seq_lens=sl, query_len=8, num_heads=2)
    small = msa_v1_indexer.get_or_plan(
        cache, block_table=bt[:2], seq_lens=sl[:2], query_len=8, num_heads=2
    )
    assert big.wrapper._proxy_scores.data_ptr() == small.wrapper._proxy_scores.data_ptr()
    assert (
        msa_v1_indexer.get_or_plan(cache, block_table=bt, seq_lens=sl, query_len=8, num_heads=2)
        is big
    )
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(graph):
            msa_v1_indexer.get_or_plan(
                cache, block_table=bt[:3], seq_lens=sl[:3], query_len=8, num_heads=2
            )


def _make_manager(*, main_dtype, num_layers=4):
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.cache_manager import (
        MiniMaxM3KVCacheManagerV2,
    )
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MiniMaxM3SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=64 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            use_kv_cache_manager_v2=True,
        ),
        CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=2,
        head_dim=D,
        tokens_per_block=PAGE,
        max_seq_len=4096,
        max_batch_size=2,
        max_num_tokens=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=main_dtype,
        vocab_size=16,
        sparse_attention_config=MiniMaxM3SparseAttentionConfig(
            implementation="msa",
            indexer_kv_dtype="nvfp4",
            sparse_disable_index_value=True,
            fuse_qkv_index_projection=True,
        ),
        sparse_layer_ids=list(range(num_layers)),
        sparse_index_dim=D,
    )


@blackwell
def test_cache_manager_stores_72_bytes_per_token_of_nvfp4_index_k(q8kv4):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
    from tensorrt_llm.bindings import DataType

    manager = _make_manager(main_dtype=DataType.NVFP4)
    try:
        assert manager.index_k_is_nvfp4
        assert manager.get_layer_bytes_per_token(0, Role.INDEX_KEY) == 72
        raw = manager.get_index_k_buffer(0)
        data, scale = manager.get_index_k_nvfp4_buffers(0)
        assert raw.dtype == torch.uint8 and tuple(raw.shape[1:]) == (1, PAGE, 72)
        assert tuple(data.shape[1:]) == (PAGE, D // 2) and tuple(scale.shape[1:]) == (PAGE, 8)
        assert raw.shape[0] == data.shape[0] == scale.shape[0]
        assert data.data_ptr() == raw.data_ptr()
        assert scale.data_ptr() == raw.data_ptr() + PAGE * D // 2
        # The MSA backend reaches the views through its metadata.
        via_metadata = MiniMaxM3MsaSparseAttentionMetadata.msa_idx_k_nvfp4(
            SimpleNamespace(kv_cache_manager=manager), 0
        )
        assert via_metadata is not None
        assert via_metadata[0].data_ptr() == data.data_ptr()
        assert via_metadata[1].data_ptr() == scale.data_ptr()
    finally:
        manager.shutdown()


@blackwell
def test_cache_manager_rejects_nvfp4_index_k_without_nvfp4_main_kv(q8kv4):
    from tensorrt_llm.bindings import DataType

    with pytest.raises(ValueError, match="requires an NVFP4 main KV cache"):
        _make_manager(main_dtype=DataType.FP8)


def test_config_accepts_nvfp4_with_its_requirements():
    from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig

    config = MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        indexer_kv_dtype="nvfp4",
        sparse_disable_index_value=True,
        fuse_qkv_index_projection=True,
    )
    assert config.indexer_kv_dtype == "nvfp4"
    with pytest.raises(ValueError, match="fuse_qkv_index_projection=True"):
        MiniMaxM3SparseAttentionConfig(implementation="msa", indexer_kv_dtype="nvfp4")
    with pytest.raises(ValueError, match="'msa' implementation"):
        MiniMaxM3SparseAttentionConfig(
            implementation="triton", indexer_kv_dtype="nvfp4", fuse_qkv_index_projection=True
        )
    with pytest.raises(ValueError, match="sparse_local_blocks=1"):
        MiniMaxM3SparseAttentionConfig(
            implementation="msa",
            indexer_kv_dtype="nvfp4",
            fuse_qkv_index_projection=True,
            sparse_local_blocks=2,
        )
