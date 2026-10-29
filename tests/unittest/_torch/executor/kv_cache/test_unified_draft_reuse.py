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
"""Block reuse of one KVCacheManagerV2 holding a full-attention target layer and a DSpark-like
sliding-window draft layer, admitted through KVCacheV2Scheduler.

A claim the drafter's window does not cover stops where the prefill recomputes everything the
drafter reads: 8-token pages, a 16-token window.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, CapacitySchedulerPolicy, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import DEFAULT_BEAM_INDEX, AttentionReusePolicy

TOKENS_PER_BLOCK = 8
WINDOW = 16
MAX_SEQ_LEN = 256


def _layer(layer_id: int, owner: KVCacheLayerOwner, **kwargs) -> KVCacheLayerSpec:
    return KVCacheLayerSpec(
        global_layer_id=layer_id,
        owner=owner,
        dtype=DataType.HALF,
        num_kv_heads=2,
        head_dim=64,
        buffer_roles=(Role.KEY, Role.VALUE),
        **kwargs,
    )


@pytest.fixture(params=["per_conversation", "all_reusable"])
def unified(request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=True,
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            max_util_for_resume=1.0,
            block_reuse_config=BlockReuseConfig(policy=request.param),
        ),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        max_num_tokens=MAX_SEQ_LEN,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        vocab_size=4096,
        layer_specs=(
            _layer(0, KVCacheLayerOwner.TARGET),
            _layer(
                1,
                KVCacheLayerOwner.DRAFT,
                sliding_window_size=WINDOW,
                reuse_policy=AttentionReusePolicy.OPTIONAL,
            ),
        ),
    )
    scheduler = KVCacheV2Scheduler(
        max_batch_size=4,
        max_num_tokens=MAX_SEQ_LEN,
        kv_cache_manager=manager,
        scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
    )
    try:
        yield scheduler, manager
    finally:
        manager.shutdown()


@dataclass
class _Request:
    py_request_id: int
    tokens: list[int]
    context_current_position: int = 0
    is_first_context_chunk: bool = True
    is_last_context_chunk: bool = True
    context_chunk_size: int = 0
    lora_task_id: int | None = None
    cache_salt: str | None = None
    py_conversation_params: None = None
    py_draft_tokens: None = None
    py_served_prefix_end: int = 0
    py_num_connector_matched_tokens: int = 0
    is_disagg_generation_init_state: bool = False
    is_dummy_request: bool = False
    return_perf_metrics: bool = False
    multimodal_hashes: None = None
    multimodal_positions: None = None
    multimodal_lengths: None = None

    @property
    def prompt_len(self) -> int:
        return len(self.tokens)

    @property
    def context_remaining_length(self) -> int:
        return self.prompt_len - self.context_current_position

    @property
    def is_dummy(self) -> bool:
        return self.is_dummy_request

    def get_tokens(self, beam_id: int = DEFAULT_BEAM_INDEX) -> list[int]:
        return self.tokens

    def get_tokens_view(self, beam_id: int = DEFAULT_BEAM_INDEX) -> np.ndarray:
        return np.asarray(self.tokens, dtype=np.int32)

    def set_prepopulated_prompt_len(self, length: int, tokens_per_block: int) -> None:
        self.context_current_position = length


def _prefill(scheduler, manager, request: _Request, *, stored: bool = True) -> int:
    """Admit and prefill ``request`` in one chunk; return its reuse claim."""
    assert scheduler._prepare_context_pair(request)
    claim = request.context_current_position
    chunk = request.context_remaining_length
    request.context_chunk_size = chunk
    assert scheduler._try_allocate_context(request, chunk)
    if not stored:
        manager.mark_unwritten_prefix(request.py_request_id, float("inf"))
    batch = ScheduledRequests()
    batch.append_context_request(request)
    request.context_current_position += chunk
    manager.update_context_resources(batch)
    return claim


def _finish(manager, *requests: _Request) -> None:
    for request in requests:
        if request.py_request_id in manager.kv_cache_map:
            manager.free_resources(request)


def _prompt(start: int, length: int) -> list[int]:
    return list(range(start, start + length))


def test_continuation_reuses_the_draft_window(unified):
    """Claiming 64 of 69 tokens is only sound with the window at 64 reused."""
    scheduler, manager = unified
    first = _Request(1, _prompt(0, 64))
    turn = _Request(2, first.tokens + _prompt(1000, 5))
    try:
        assert _prefill(scheduler, manager, first) == 0
        _finish(manager, first)
        assert _prefill(scheduler, manager, turn) == 64
    finally:
        _finish(manager, first, turn)


@pytest.mark.parametrize("first_len, shared", [(43, 43), (64, 63)])
def test_a_partial_match_reuses_the_drafters_partial_page(unified, first_len, shared):
    """A continuation of a 43-token prompt, or a 63-token prefix of a 64-token one, is claimed whole:
    the drafter's window there ends in a partial page, which the tree holds."""
    scheduler, manager = unified
    first = _Request(1, _prompt(0, first_len))
    turn = _Request(2, first.tokens[:shared] + _prompt(1000, 5))
    try:
        _prefill(scheduler, manager, first)
        _finish(manager, first)
        assert _prefill(scheduler, manager, turn) == shared
    finally:
        _finish(manager, first, turn)


@pytest.mark.parametrize("unified", ["per_conversation"], indirect=True)
def test_a_branch_point_window_is_recomputed_once_and_then_shared(unified):
    """Only the window at the first prompt's end is committed, none at the shared prefix (page 40
    of 43 tokens): the first branch claims 24, one window before 40, and publishes the window there
    with its page 40, so the third claims all 43. A match ending in the first prompt's page 40, which
    has no drafter page, falls back to 40."""
    scheduler, manager = unified
    first = _Request(1, _prompt(0, 64))
    second = _Request(2, first.tokens[:43] + _prompt(1000, 9))
    third = _Request(3, first.tokens[:43] + _prompt(2000, 9))
    fourth = _Request(4, first.tokens[:44] + _prompt(3000, 9))
    try:
        _prefill(scheduler, manager, first)
        _finish(manager, first)
        assert _prefill(scheduler, manager, second) == 24
        _finish(manager, second)
        assert _prefill(scheduler, manager, third) == 43
        _finish(manager, third)
        assert _prefill(scheduler, manager, fourth) == 40
    finally:
        _finish(manager, first, second, third, fourth)


def test_an_unstored_context_is_never_published(unified):
    scheduler, manager = unified
    first = _Request(1, _prompt(0, 64))
    again = _Request(2, list(first.tokens))
    try:
        _prefill(scheduler, manager, first, stored=False)
        _finish(manager, first)
        assert _prefill(scheduler, manager, again) == 0
    finally:
        _finish(manager, first, again)
