"""Tests for PyExecutor request handling functionality.

This module tests the request handling logic that was moved from ExecutorRequestQueue
to PyExecutor, including:
- _handle_special_queue_items method
- canceled_req_ids management
- waiting_queue management
- is_shutdown state management
- expected_num_active_requests tracking
- _emit_previous_batch_responses / _finalize_previous_batch (overlap-loop split)
"""

from unittest.mock import MagicMock, Mock

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue


class MockPyExecutor:
    """A mock PyExecutor class for testing request handling logic.

    This mock contains only the attributes and methods needed to test
    the _handle_special_queue_items functionality.
    """

    def __init__(self, dist):
        self.dist = dist
        self.canceled_req_ids = []
        self.control_requests = []
        self.request_accumulated = []
        self.is_shutdown = False
        self.expected_num_active_requests = 0
        self.new_active_requests_queue_latency_ms = 0.0
        self.waiting_queue = FCFSWaitingQueue()

    def _handle_special_queue_items(self, new_requests):
        """Handle special signals.

        This method mirrors PyExecutor._handle_special_queue_items.
        """
        accepted_new_requests = []
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1 :])
                break
            else:
                accepted_new_requests.append(req_item)

        return accepted_new_requests

    def update_waiting_queue(self):
        """Update waiting queue to remove canceled requests.

        This method mirrors PyExecutor._handle_canceled_requests.
        """
        if self.canceled_req_ids:
            canceled_set = set(self.canceled_req_ids)
            self.waiting_queue.remove_by_ids(canceled_set)

    def clear_canceled_req_ids(self):
        """Clear the list of canceled request IDs."""
        self.canceled_req_ids.clear()

    def get_canceled_req_ids(self):
        """Get the list of canceled request IDs."""
        return self.canceled_req_ids

    def get_canceled_req_ids_size(self):
        """Get the number of canceled request IDs."""
        return len(self.canceled_req_ids)

    def get_expected_num_active_requests(self):
        """Get the expected number of active requests."""
        return self.expected_num_active_requests

    def get_waiting_queue_size(self):
        """Get the size of the waiting queue."""
        return len(self.waiting_queue)

    def _get_new_active_requests_queue_latency(self):
        """Get the queue latency for new active requests."""
        return self.new_active_requests_queue_latency_ms


@pytest.fixture
def mock_dist():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 1
    return mock_dist


@pytest.fixture
def mock_executor(mock_dist):
    """Create a MockPyExecutor instance for testing."""
    return MockPyExecutor(dist=mock_dist)


def test_handle_special_queue_items(mock_executor):
    """Test special queue item handling."""
    # Create a mock request
    mock_request = Mock()
    if hasattr(mock_request, "sampling_config"):
        delattr(mock_request, "sampling_config")

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    requests = [normal_req, cancel_req, shutdown_req]

    valid_requests = mock_executor._handle_special_queue_items(requests)

    assert len(valid_requests) == 1
    assert valid_requests[0] == normal_req
    assert mock_executor.is_shutdown
    assert 2 in mock_executor.canceled_req_ids


def test_clear_canceled_req_ids(mock_executor):
    """Test clearing canceled request IDs."""
    mock_executor.canceled_req_ids = [1, 2, 3]
    assert len(mock_executor.canceled_req_ids) == 3

    mock_executor.clear_canceled_req_ids()

    assert len(mock_executor.canceled_req_ids) == 0


def test_update_waiting_queue(mock_executor):
    """Test updating waiting queue to remove canceled requests."""
    items = [
        RequestQueueItem(1, Mock()),
        RequestQueueItem(2, Mock()),
        RequestQueueItem(3, Mock()),
    ]
    mock_executor.waiting_queue.extend(items)
    mock_executor.canceled_req_ids = [2]

    mock_executor.update_waiting_queue()

    assert len(mock_executor.waiting_queue) == 2
    remaining_ids = [item.id for item in mock_executor.waiting_queue]
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 2 not in remaining_ids


def test_getter_methods(mock_executor):
    """Test various getter methods."""
    # Test initial values
    assert mock_executor._get_new_active_requests_queue_latency() == 0
    assert mock_executor.get_expected_num_active_requests() == 0
    assert mock_executor.get_canceled_req_ids_size() == 0
    assert mock_executor.get_canceled_req_ids() == []
    assert mock_executor.get_waiting_queue_size() == 0

    # Add some data and test
    mock_executor.canceled_req_ids = [3, 4]
    mock_executor.expected_num_active_requests = 5
    mock_executor.new_active_requests_queue_latency_ms = 10.5
    mock_executor.waiting_queue.append(RequestQueueItem(1, Mock()))

    assert mock_executor.get_canceled_req_ids_size() == 2
    assert mock_executor.get_canceled_req_ids() == [3, 4]
    assert mock_executor.get_expected_num_active_requests() == 5
    assert mock_executor._get_new_active_requests_queue_latency() == 10.5
    assert mock_executor.get_waiting_queue_size() == 1


def _classify_termination(request, enable_partial_reuse_for_disagg, is_vswa, pp_size):
    """Reproduce the termination logic from _handle_responses (py_executor.py).

    Returns:
        "terminate" | "stats_only" | "skip"
    """
    force_terminate_for_partial_reuse = (
        enable_partial_reuse_for_disagg and not is_vswa and pp_size == 1
    )
    if request.is_disagg_context_complete_state:
        return "stats_only"
    elif force_terminate_for_partial_reuse:
        return "terminate"
    elif not request.is_disagg_context_transmission_state:
        return "terminate"
    return "skip"


def _make_request(complete_state, transmission_state):
    req = Mock()
    req.is_disagg_context_complete_state = complete_state
    req.is_disagg_context_transmission_state = transmission_state
    return req


class TestDisaggTerminationGuard:
    """Verify _handle_responses does not double-terminate DISAGG_CONTEXT_COMPLETE
    requests that were already cleaned up by _check_disagg_ctx_cache_transfer_status
    (nvbug/5961736)."""

    def test_normal_path_skips_context_complete(self):
        """Without partial reuse, CONTEXT_COMPLETE goes to stats only."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, False, False, 1) == "stats_only"

    def test_normal_path_skips_transmission_in_progress(self):
        """Without partial reuse, TRANS_IN_PROGRESS is skipped (still in flight)."""
        req = _make_request(complete_state=False, transmission_state=True)
        assert _classify_termination(req, False, False, 1) == "skip"

    def test_normal_path_terminates_regular_request(self):
        """Without partial reuse, a normal finished request is terminated."""
        req = _make_request(complete_state=False, transmission_state=False)
        assert _classify_termination(req, False, False, 1) == "terminate"

    def test_partial_reuse_terminates_non_complete(self):
        """With partial reuse, non-CONTEXT_COMPLETE requests are terminated."""
        for complete, transmission in [(False, True), (False, False)]:
            req = _make_request(complete, transmission)
            assert _classify_termination(req, True, False, 1) == "terminate"

    def test_partial_reuse_skips_context_complete(self):
        """With partial reuse, CONTEXT_COMPLETE still goes to stats only."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, False, 1) == "stats_only"

    def test_partial_reuse_disabled_by_vswa(self):
        """VSWA disables partial reuse path, falling back to normal logic."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, True, 1) == "stats_only"

    def test_partial_reuse_disabled_by_pp(self):
        """PP > 1 disables partial reuse path, falling back to normal logic."""
        req = _make_request(complete_state=True, transmission_state=False)
        assert _classify_termination(req, True, False, 2) == "stats_only"


# ---------------------------------------------------------------------------
# Tests for _compute_scheduled_tokens with KV cache reuse chunk-shift logic
# ---------------------------------------------------------------------------


def _make_ctx_request(
    context_chunk_size,
    context_remaining_length,
    estimated_reusable_tokens=0,
    is_first_context_chunk=True,
    context_current_position=0,
):
    """Helper to create a mock context request for token computation tests."""
    req = Mock()
    req.context_chunk_size = context_chunk_size
    req.context_remaining_length = context_remaining_length
    req.estimated_reusable_tokens = estimated_reusable_tokens
    req.is_first_context_chunk = is_first_context_chunk
    req.context_current_position = context_current_position
    return req


def _make_gen_request(num_draft_tokens=0):
    """Helper to create a mock generation request."""
    req = Mock()
    req.num_draft_tokens = num_draft_tokens
    return req


class TestComputeScheduledTokens:
    """Tests for PyExecutor._compute_scheduled_tokens.

    Validates the chunk-shift aware token accounting: setPrepopulatedPromptLen
    shifts the chunk window right by the reused amount rather than shrinking it.
    Non-last chunks cost chunkSize; only last chunks cost remaining - reusable.
    """

    def test_no_reuse(self):
        """Without reuse, compute = chunk_size."""
        ctx = [_make_ctx_request(context_chunk_size=100, context_remaining_length=100)]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 100

    def test_last_chunk_with_reuse(self):
        """Last chunk (reusable + chunk >= remaining): compute = chunk - reusable."""
        # promptLen=100, reusable=60, chunk=100 (full context)
        # 60 + 100 >= 100 → last chunk → compute = max(1, 100 - 60) = 40
        ctx = [
            _make_ctx_request(
                context_chunk_size=100, context_remaining_length=100, estimated_reusable_tokens=60
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 40

    def test_non_last_chunk_with_reuse(self):
        """Non-last chunk (reusable + chunk < remaining): compute = chunk_size.

        This is the core chunk-shift scenario. The old formula would compute
        max(0, 25 - 30) = 0, but the correct cost is 25 because the chunk
        window shifts right rather than shrinking.
        """
        # promptLen=100, reusable=30, chunk=25
        # 30 + 25 = 55 < 100 → non-last chunk → compute = 25
        ctx = [
            _make_ctx_request(
                context_chunk_size=25, context_remaining_length=100, estimated_reusable_tokens=30
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 25

    def test_non_first_chunk_ignores_reuse(self):
        """Reusable tokens only apply to the first context chunk."""
        ctx = [
            _make_ctx_request(
                context_chunk_size=50,
                context_remaining_length=50,
                estimated_reusable_tokens=30,
                is_first_context_chunk=False,
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 50

    def test_v2_scheduler_position_advanced(self):
        """V2 scheduler: context_current_position already advanced past reuse.

        reusable_in_chunk = max(0, 30 - 30) = 0 → no credit → compute = chunk.
        """
        ctx = [
            _make_ctx_request(
                context_chunk_size=50,
                context_remaining_length=70,
                estimated_reusable_tokens=30,
                context_current_position=30,
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 50

    def test_min_compute_is_one(self):
        """Compute cost is floored at 1 even when reusable >= chunk_size."""
        # chunk=10, remaining=10, reusable=15 → last chunk → max(1, 10-15) = 1
        ctx = [
            _make_ctx_request(
                context_chunk_size=10, context_remaining_length=10, estimated_reusable_tokens=15
            )
        ]
        assert PyExecutor._compute_scheduled_tokens(ctx, []) == 1

    def test_generation_tokens(self):
        """Generation requests contribute 1 + num_draft_tokens each."""
        gen = [_make_gen_request(3), _make_gen_request(0)]
        assert PyExecutor._compute_scheduled_tokens([], gen) == (1 + 3) + (1 + 0)

    def test_mixed_context_and_generation(self):
        """Combined context (with chunk-shift) and generation tokens."""
        # Non-last chunk: compute = 25
        ctx = [
            _make_ctx_request(
                context_chunk_size=25, context_remaining_length=100, estimated_reusable_tokens=30
            )
        ]
        gen = [_make_gen_request(2)]
        # 25 ctx + (1 + 2) gen = 28
        assert PyExecutor._compute_scheduled_tokens(ctx, gen) == 28

    def test_multiple_ctx_requests_mixed_chunks(self):
        """Multiple context requests: one non-last chunk, one last chunk."""
        # req0: non-last chunk → compute = 20
        req0 = _make_ctx_request(
            context_chunk_size=20, context_remaining_length=100, estimated_reusable_tokens=30
        )
        # req1: last chunk (reuse=10, chunk=50, remaining=50) → 10+50>=50
        # → compute = max(1, 50-10) = 40
        req1 = _make_ctx_request(
            context_chunk_size=50, context_remaining_length=50, estimated_reusable_tokens=10
        )
        assert PyExecutor._compute_scheduled_tokens([req0, req1], []) == 20 + 40


# ---------------------------------------------------------------------------
# Tests for the overlap-loop split:
#   _emit_previous_batch_responses (pre-sample_async)
#   _finalize_previous_batch       (post-sample_async)
# ---------------------------------------------------------------------------

BUILT_RESPONSES = [("req-1", "resp-1"), ("req-2", "resp-2")]
TO_TERMINATE = ["term-A"]
FINISHED_BY_TRANSFER = ["xfer-B"]


@pytest.fixture
def overlap_executor_mock():
    """MagicMock stand-in for a PyExecutor when invoking the unbound
    overlap-loop helpers. Child attributes are left auto-created so their
    method calls surface on the parent's `mock_calls` for ordering checks.
    """
    m = MagicMock()
    m._build_responses.return_value = (BUILT_RESPONSES, TO_TERMINATE, FINISHED_BY_TRANSFER)
    m._finalize_handled_responses.return_value = TO_TERMINATE + FINISHED_BY_TRANSFER
    m.model_engine.attn_metadata = "ATTN_META"
    m.model_engine.kv_cache_dtype_byte_size = 0.5
    m.enable_kv_cache_events = False
    m.enable_iter_perf_stats = False
    m.active_requests = ["active-req"]
    return m


def _called_names(mock):
    """Ordered list of method-call names recorded on `mock` and its
    auto-created children, e.g. `'_build_responses'` or
    `'resource_manager.update_resources'`."""
    return [c[0] for c in mock.mock_calls]


class TestEmitPreviousBatchResponses:
    """Pre-sample_async half of the overlap-loop split."""

    def test_call_order(self, overlap_executor_mock):
        PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        assert _called_names(overlap_executor_mock) == [
            "_handle_canceled_requests",
            "_build_responses",
            "_enqueue_responses",
        ]

    def test_enqueues_built_responses(self, overlap_executor_mock):
        PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        overlap_executor_mock._enqueue_responses.assert_called_once_with(BUILT_RESPONSES)

    def test_returns_deferred_termination_state(self, overlap_executor_mock):
        assert PyExecutor._emit_previous_batch_responses(overlap_executor_mock) == (
            TO_TERMINATE,
            FINISHED_BY_TRANSFER,
        )

    def test_does_not_terminate(self, overlap_executor_mock):
        """The emit half must defer termination so the response can still
        find the originating request before its resources are released."""
        PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        overlap_executor_mock._finalize_handled_responses.assert_not_called()
        overlap_executor_mock._terminate_request.assert_not_called()
        overlap_executor_mock.resource_manager.update_resources.assert_not_called()
        overlap_executor_mock._add_kv_cache_events.assert_not_called()
        overlap_executor_mock._process_iter_stats.assert_not_called()

    def test_empty_build_still_enqueues_for_collective(self, overlap_executor_mock):
        """An empty response list still enqueues so the TP-gather collective
        participates on every rank."""
        overlap_executor_mock._build_responses.return_value = ([], [], [])
        result = PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        overlap_executor_mock._enqueue_responses.assert_called_once_with([])
        assert result == ([], [])


class TestFinalizePreviousBatch:
    """Post-sample_async half of the overlap-loop split."""

    def test_call_order(self, overlap_executor_mock):
        PyExecutor._finalize_previous_batch(overlap_executor_mock, ["t"], ["x"])
        assert _called_names(overlap_executor_mock) == [
            "_finalize_handled_responses",
            "resource_manager.update_resources",
        ]

    def test_finalize_forwards_termination_lists(self, overlap_executor_mock):
        PyExecutor._finalize_previous_batch(overlap_executor_mock, ["term-A"], ["xfer-C"])
        overlap_executor_mock._finalize_handled_responses.assert_called_once_with(
            ["term-A"], ["xfer-C"]
        )

    def test_update_resources_forwards_engine_attrs(self, overlap_executor_mock):
        prev_sched = MagicMock(name="prev_sched")
        overlap_executor_mock.previous_batch.scheduled_requests = prev_sched
        overlap_executor_mock.model_engine.attn_metadata = "attn-X"
        overlap_executor_mock.model_engine.kv_cache_dtype_byte_size = 2.0
        PyExecutor._finalize_previous_batch(overlap_executor_mock, [], [])
        overlap_executor_mock.resource_manager.update_resources.assert_called_once_with(
            prev_sched, "attn-X", 2.0
        )

    def test_missing_model_engine_attrs_fall_back_to_none(self, overlap_executor_mock):
        """`getattr(model_engine, '...', None)` takes the default-None
        branch when the attribute is absent."""
        overlap_executor_mock.model_engine = MagicMock(spec=[])
        PyExecutor._finalize_previous_batch(overlap_executor_mock, [], [])
        args, _ = overlap_executor_mock.resource_manager.update_resources.call_args
        # (scheduled_requests, attn_metadata, kv_cache_dtype_byte_size)
        assert args[1] is None
        assert args[2] is None

    @pytest.mark.parametrize("enabled,expected_calls", [(False, 0), (True, 1)])
    def test_kv_cache_events_gated(self, overlap_executor_mock, enabled, expected_calls):
        overlap_executor_mock.enable_kv_cache_events = enabled
        PyExecutor._finalize_previous_batch(overlap_executor_mock, [], [])
        assert overlap_executor_mock._add_kv_cache_events.call_count == expected_calls

    def test_iter_perf_stats_disabled(self, overlap_executor_mock):
        PyExecutor._finalize_previous_batch(overlap_executor_mock, [], [])
        overlap_executor_mock._process_iter_stats.assert_not_called()

    def test_iter_perf_stats_enabled(self, overlap_executor_mock):
        overlap_executor_mock.enable_iter_perf_stats = True
        overlap_executor_mock._finalize_handled_responses.return_value = ["f1", "f2"]
        overlap_executor_mock.active_requests = ["a1"]
        PyExecutor._finalize_previous_batch(overlap_executor_mock, ["t"], ["x"])
        overlap_executor_mock._process_iter_stats.assert_called_once_with(
            ["f1", "f2"], ["a1"], overlap_executor_mock.previous_batch
        )


class TestOverlapSplitInvariants:
    """End-to-end ordering: emit -> (current iter sample_async) -> finalize."""

    def test_full_call_sequence(self, overlap_executor_mock):
        """Responses leave rank 0 before any request is terminated and
        before the current iter's `_sample_async`."""
        pending = PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        overlap_executor_mock._sample_async("scheduled_batch", "batch_outputs")
        PyExecutor._finalize_previous_batch(overlap_executor_mock, *pending)
        assert _called_names(overlap_executor_mock) == [
            "_handle_canceled_requests",
            "_build_responses",
            "_enqueue_responses",
            "_sample_async",
            "_finalize_handled_responses",
            "resource_manager.update_resources",
        ]

    def test_termination_state_threaded_through(self, overlap_executor_mock):
        """Emit's return value is forwarded verbatim into finalize."""
        pending = PyExecutor._emit_previous_batch_responses(overlap_executor_mock)
        PyExecutor._finalize_previous_batch(overlap_executor_mock, *pending)
        overlap_executor_mock._finalize_handled_responses.assert_called_once_with(
            TO_TERMINATE, FINISHED_BY_TRANSFER
        )
