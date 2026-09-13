# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for MiniMaxM3HiSparseCoordinator.

Drives the coordinator's swap-in path (which wraps
``trtllm::hisparse_swap_in_blocks``) across decode-style steps, asserting the hit
/ newest / miss / fast-path behavior and that missed blocks' K and V pages are
actually copied host->device into the LRU-evicted hot-buffer slot.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers trtllm ops)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.hisparse_coordinator import \
    MiniMaxM3HiSparseCoordinator

DEVICE = "cuda"
PAGE_SHAPE = (2, 4, 4)  # [num_kv_heads, tokens_per_block, head_dim]; 32 elems

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch.ops.trtllm, "hisparse_swap_in_blocks"),
    reason="HiSparse swap-in op requires a CUDA build of libth_common.so.",
)


def _make_coord(hot_window_blocks=4, topk=3, max_blocks_per_seq=16):
    return MiniMaxM3HiSparseCoordinator(
        num_sparse_layers=1,
        max_reqs=2,
        hot_window_blocks=hot_window_blocks,
        topk=topk,
        max_blocks_per_seq=max_blocks_per_seq,
        page_shape=PAGE_SHAPE,
        page_dtype=torch.uint8,
        device=DEVICE,
    )


def _fill_host(coord, layer=0):
    """Give every host block deterministic, distinct bytes for K and V."""
    n = coord.max_reqs * coord.max_blocks_per_seq
    cols = coord.host_cache_k[layer].view(n, -1).shape[1]
    base = torch.arange(cols, dtype=torch.int64).view(1, cols)
    pg = torch.arange(n, dtype=torch.int64).view(n, 1)
    coord.host_cache_k[layer].view(n, -1).copy_(((pg * 7 + base) % 256).to(torch.uint8))
    coord.host_cache_v[layer].view(n, -1).copy_(
        ((pg * 7 + base + 100) % 256).to(torch.uint8))


def _seed_resident(coord, rid, blocks_in_slots, newest_block, layer=0):
    """Place ``blocks_in_slots`` into slots 0.. and bind ``newest_block``.

    Copies the corresponding host pages onto the (identity-mapped) device pages so
    a resident slot's device bytes match its block, mirroring what prefill
    staging + the newest-block KV write would produce.
    """
    for slot, blk in enumerate(blocks_in_slots):
        coord.device_buffer_blocks[layer, rid, slot] = blk
        dev_pg = int(coord.device_buffer_locs[layer, rid, slot])
        host_pg = coord.host_page(rid, blk)
        coord.device_buffer_k[layer][dev_pg].copy_(coord.host_cache_k[layer][host_pg])
        coord.device_buffer_v[layer][dev_pg].copy_(coord.host_cache_v[layer][host_pg])
    # Bind the newest (current) block into the reserved slot's device page.
    newest_pg = coord.newest_slot_page(layer, rid)
    host_pg = coord.host_page(rid, newest_block)
    coord.device_buffer_k[layer][newest_pg].copy_(coord.host_cache_k[layer][host_pg])
    coord.device_buffer_v[layer][newest_pg].copy_(coord.host_cache_v[layer][host_pg])
    torch.cuda.synchronize()


def _step(coord, top_k, seq_len_blocks, num_reqs=1, num_real=None):
    coord.prepare_decode_step(
        req_pool_indices=torch.arange(num_reqs, dtype=torch.int32, device=DEVICE),
        seq_lens_blocks=torch.full((num_reqs, ),
                                   seq_len_blocks,
                                   dtype=torch.int32,
                                   device=DEVICE),
        num_real_reqs=num_reqs if num_real is None else num_real,
    )
    out = coord.swap_in(
        0, torch.tensor(top_k, dtype=torch.int32, device=DEVICE))
    torch.cuda.synchronize()
    return out


def test_fast_path_short_sequence():
    """seq_len_blocks <= hot_window: direct block-id -> device page, no IO."""
    coord = _make_coord()
    _fill_host(coord)
    _seed_resident(coord, rid=0, blocks_in_slots=[0, 1, 2, 3], newest_block=2)
    k_before = coord.device_buffer_k[0].clone()

    out = _step(coord, [[2, 0, 1]], seq_len_blocks=3)

    # Identity locs => device page == block id for the fast path.
    assert torch.equal(out.cpu(), torch.tensor([[2, 0, 1]], dtype=torch.int32))
    assert torch.equal(coord.device_buffer_k[0].cpu(), k_before.cpu())


def test_hits_and_newest_no_io():
    coord = _make_coord()
    _fill_host(coord)
    _seed_resident(coord, rid=0, blocks_in_slots=[1, 4, 2, 5], newest_block=7)
    k_before = coord.device_buffer_k[0].clone()
    v_before = coord.device_buffer_v[0].clone()

    # 4 -> slot1(page1), 2 -> slot2(page2), 7 -> newest reserved slot (page4).
    out = _step(coord, [[4, 2, 7]], seq_len_blocks=8)

    assert torch.equal(out.cpu(), torch.tensor([[1, 2, 4]], dtype=torch.int32))
    assert torch.equal(coord.device_buffer_blocks[0, 0].cpu(),
                       torch.tensor([1, 4, 2, 5, -1], dtype=torch.int32))
    # Pure hits: no host->device copies.
    assert torch.equal(coord.device_buffer_k[0].cpu(), k_before.cpu())
    assert torch.equal(coord.device_buffer_v[0].cpu(), v_before.cpu())


def test_miss_copies_page_from_host():
    coord = _make_coord()
    _fill_host(coord)
    _seed_resident(coord, rid=0, blocks_in_slots=[1, 4, 2, 5], newest_block=7)

    # From fresh LRU [0,1,2,3]: 4 hits slot1(page1), 2 hits slot2(page2),
    # 6 misses -> evicts LRU-front evictable slot0 (page0).
    out = _step(coord, [[4, 2, 6]], seq_len_blocks=8)

    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 2
    assert out[0, 2].item() == 0
    assert coord.device_buffer_blocks[0, 0, 0].item() == 6
    host_pg = coord.host_page(0, 6)
    assert torch.equal(coord.device_buffer_k[0][0].cpu(),
                       coord.host_cache_k[0][host_pg])
    assert torch.equal(coord.device_buffer_v[0][0].cpu(),
                       coord.host_cache_v[0][host_pg])


def test_padded_rows_return_invalid():
    coord = _make_coord()
    _fill_host(coord)
    _seed_resident(coord, rid=0, blocks_in_slots=[1, 4, 2, 5], newest_block=7)
    _seed_resident(coord, rid=1, blocks_in_slots=[1, 4, 2, 5], newest_block=7)

    out = _step(coord, [[4, 2, 7], [4, 2, 7]],
                seq_len_blocks=8,
                num_reqs=2,
                num_real=1)

    assert torch.equal(out[0].cpu(), torch.tensor([1, 2, 4], dtype=torch.int32))
    assert torch.equal(out[1].cpu(), torch.tensor([-1, -1, -1], dtype=torch.int32))
