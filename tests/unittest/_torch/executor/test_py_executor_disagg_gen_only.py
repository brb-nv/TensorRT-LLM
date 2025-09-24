"""
Test PyExecutor executor_loop for gen-only disaggregated serving with prefill cache populated.

This test verifies the executor_loop behavior when:
1. Generation-only disaggregated serving is enabled
2. Prefill cache is already populated (simulated)
3. Requests are processed without needing KV cache transfer
"""

import os
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.disaggregated_params import DisaggregatedParams


class MockResourceManager:

    def __init__(self):
        self.prepare_resources = Mock()
        self.update_resources = Mock()
        self.free_resources = Mock()


class MockScheduler:

    def __init__(self):
        self.schedule_request = Mock()


class MockModelEngine:

    def __init__(self):
        self.forward = Mock()
        self.prepare_for_execute = Mock()
        self.iter_states = {'num_ctx_tokens': 0}
        self.spec_config = Mock()
        self.spec_config.max_draft_len = 0
        self.enable_spec_decode = False


class MockSampler:

    def __init__(self):
        self.sample_async = Mock()


class MockDistributed:

    def __init__(self):
        self.rank = 0
        self.tp_size = 1
        self.pp_size = 1
        self.has_pp = False
        self.tp_rank = 0
        self.cp_rank = 0
        self.cp_size = 1
        self.cp_config = {}
        self.is_first_pp_rank = True
        self.is_last_pp_rank = True
        self.next_pp_rank = 1
        self.prev_pp_rank = 0


class MockKvCacheTransceiver:

    def __init__(self):
        self.request_and_receive_async = Mock()
        self.request_and_receive_sync = Mock()
        self.respond_and_send_async = Mock()


@pytest.fixture
def mock_executor_components():
    """Create mock components for PyExecutor."""
    return {
        'resource_manager': MockResourceManager(),
        'scheduler': MockScheduler(),
        'model_engine': MockModelEngine(),
        'sampler': MockSampler(),
        'dist': MockDistributed(),
        'kv_cache_transceiver': MockKvCacheTransceiver(),
    }


@pytest.fixture
def mock_executor_request_queue():
    """Create a mock ExecutorRequestQueue."""
    mock_queue = Mock()
    mock_queue.active = True
    mock_queue.is_shutdown = False
    mock_queue.get_new_active_requests_queue_latency.return_value = 0.0
    mock_queue.waiting_queue = []
    mock_queue.canceled_req_ids = []
    return mock_queue


def create_mock_disagg_gen_request(request_id: int,
                                   first_gen_tokens=None,
                                   encoded_opaque_state=None):
    """Create a mock disaggregated generation request."""
    request = Mock(spec=LlmRequest)
    request.request_id = request_id
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    request.is_disagg_generation_init_state = True
    request.is_disagg_generation_transmission_complete = False
    request.is_disagg_generation_transmission_in_progress = False
    request.is_context_only_request = False
    request.is_generation_only_request = True
    request.is_finished = False
    request.is_canceled = False
    request.seq_slot = None
    request.py_seq_slot = None
    request.py_draft_tokens = []
    request.py_draft_pages_allocated = 0
    request.add_new_token = Mock()
    request.set_first_scheduled_time = Mock()
    request.return_perf_metrics = False

    # Disaggregated parameters for generation-only request
    request.disaggregated_params = DisaggregatedParams(
        request_type="generation_only",
        first_gen_tokens=first_gen_tokens or [7],
        ctx_request_id=1,
        encoded_opaque_state=encoded_opaque_state,
        draft_tokens=None)

    return request


def create_scheduled_batch_with_disagg_gen_requests(generation_requests):
    """Create a ScheduledRequests batch with disaggregated generation requests."""
    scheduled_batch = Mock(spec=ScheduledRequests)
    scheduled_batch.context_requests = []
    scheduled_batch.generation_requests = generation_requests
    scheduled_batch.paused_requests = []
    scheduled_batch.batch_size = len(generation_requests)
    scheduled_batch.all_requests.return_value = generation_requests
    return scheduled_batch


class TestPyExecutorDisaggGenOnly:
    """Test PyExecutor with generation-only disaggregated serving."""

    def test_executor_loop_gen_only_with_prefill_cache(
            self, mock_executor_components, mock_executor_request_queue):
        """
        Test executor_loop with gen-only disaggregated serving and prefill cache populated.

        This test simulates the scenario where:
        1. TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 is set (gen-only mode)
        2. Prefill cache is already populated
        3. Generation requests are processed without KV cache transfer
        """
        # Set up environment for gen-only benchmark mode
        with patch.dict(os.environ, {'TRTLLM_DISAGG_BENCHMARK_GEN_ONLY': '1'}):
            # Create PyExecutor with mocked components
            executor = PyExecutor(
                resource_manager=mock_executor_components['resource_manager'],
                scheduler=mock_executor_components['scheduler'],
                model_engine=mock_executor_components['model_engine'],
                sampler=mock_executor_components['sampler'],
                dist=mock_executor_components['dist'],
                max_num_sequences=8,
                kv_cache_transceiver=mock_executor_components[
                    'kv_cache_transceiver'],
                start_worker=False  # Don't start background worker
            )

            # Replace the executor request queue with our mock
            executor.executor_request_queue = mock_executor_request_queue

            # Create mock disaggregated generation requests
            gen_request_1 = create_mock_disagg_gen_request(1,
                                                           first_gen_tokens=[7])
            gen_request_2 = create_mock_disagg_gen_request(2,
                                                           first_gen_tokens=[8])

            # Set up active requests list
            executor.active_requests = [gen_request_1, gen_request_2]

            # Mock the scheduled batch that will be returned
            scheduled_batch = create_scheduled_batch_with_disagg_gen_requests(
                [gen_request_1, gen_request_2])

            # Mock _prepare_and_schedule_batch to return our scheduled batch
            with patch.object(
                    executor,
                    '_prepare_and_schedule_batch') as mock_prepare_schedule:
                mock_prepare_schedule.side_effect = [
                    (scheduled_batch, None),  # First iteration: return requests
                    (None, None)  # Second iteration: exit loop
                ]

                # Mock other methods that are called during the loop
                with patch.object(executor, '_pause_requests') as mock_pause, \
                     patch.object(executor, '_prepare_disagg_gen_transmission_complete') as mock_prepare_disagg, \
                     patch.object(executor, '_handle_first_token_response') as mock_handle_first_token, \
                     patch.object(executor, '_forward_step') as mock_forward, \
                     patch.object(executor, '_sample_async') as mock_sample, \
                     patch.object(executor, '_update_request_states') as mock_update_states, \
                     patch.object(executor, '_update_requests') as mock_update_requests, \
                     patch.object(executor, '_handle_canceled_requests') as mock_handle_canceled, \
                     patch.object(executor, '_handle_responses') as mock_handle_responses, \
                     patch.object(executor, '_recv_disagg_gen_cache') as mock_recv_cache, \
                     patch.object(executor, '_send_disagg_ctx_cache') as mock_send_cache, \
                     patch.object(executor, '_terminate_ctx_finished_requests') as mock_terminate_ctx, \
                     patch.object(executor, '_kv_connector_terminate_requests') as mock_kv_terminate, \
                     patch.object(executor, '_kv_connector_start_batch') as mock_kv_start, \
                     patch.object(executor, '_add_kv_cache_events') as mock_add_events, \
                     patch('torch.cuda.set_device'), \
                     patch('tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT'):

                    # Configure return values
                    mock_forward.return_value = {
                        'logits': torch.randn(2, 1, 1000)
                    }
                    mock_sample.return_value = Mock()
                    mock_handle_responses.return_value = []
                    mock_send_cache.return_value = []

                    # Set device_id for torch.cuda.set_device
                    executor.device_id = 0
                    executor.enable_iter_perf_stats = False
                    executor.enable_attention_dp = False
                    executor.enable_kv_cache_events = False
                    executor.ctx_in_transmission_requests = []
                    executor.drafter = None
                    executor.guided_decoder = None
                    executor.kv_connector_manager = None
                    executor._profiler = Mock()
                    executor._profiler.return_value.__enter__ = Mock(
                        return_value=Mock())
                    executor._profiler.return_value.__exit__ = Mock(
                        return_value=None)

                    # Run the executor loop
                    executor._executor_loop()

                    # Verify that the gen-only disaggregated requests were processed correctly

                    # 1. Verify _recv_disagg_gen_cache was called to handle generation cache
                    # In gen-only mode, this should mark requests as transmission complete immediately
                    assert mock_recv_cache.called

                    # 2. Verify _prepare_disagg_gen_transmission_complete was called
                    mock_prepare_disagg.assert_called_once_with(scheduled_batch)

                    # 3. Verify _handle_first_token_response was called to return first tokens
                    mock_handle_first_token.assert_called_once_with(
                        scheduled_batch)

                    # 4. Verify resources were prepared
                    mock_executor_components[
                        'resource_manager'].prepare_resources.assert_called_once_with(
                            scheduled_batch)

                    # 5. Verify forward pass was executed
                    mock_forward.assert_called_once_with(scheduled_batch)

                    # 6. Verify sampling was performed
                    mock_sample.assert_called_once()

                    # 7. Verify request states were updated
                    mock_update_states.assert_called_once_with(scheduled_batch)
                    mock_update_requests.assert_called_once()

                    # 8. Verify responses were handled
                    mock_handle_responses.assert_called_once()

    def test_recv_disagg_gen_cache_gen_only_mode(self,
                                                 mock_executor_components):
        """
        Test _recv_disagg_gen_cache specifically in gen-only mode.

        This test verifies that when TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1 is set,
        the method immediately marks requests as transmission complete without
        actual KV cache transfer.
        """
        with patch.dict(os.environ, {'TRTLLM_DISAGG_BENCHMARK_GEN_ONLY': '1'}):
            # Create PyExecutor
            executor = PyExecutor(
                resource_manager=mock_executor_components['resource_manager'],
                scheduler=mock_executor_components['scheduler'],
                model_engine=mock_executor_components['model_engine'],
                sampler=mock_executor_components['sampler'],
                dist=mock_executor_components['dist'],
                max_num_sequences=8,
                kv_cache_transceiver=mock_executor_components[
                    'kv_cache_transceiver'],
                start_worker=False)

            # Create mock generation requests
            gen_request_1 = create_mock_disagg_gen_request(1)
            gen_request_2 = create_mock_disagg_gen_request(2)
            new_gen_reqs = [gen_request_1, gen_request_2]

            # Call _recv_disagg_gen_cache
            executor._recv_disagg_gen_cache(new_gen_reqs)

            # Verify that both requests were marked as transmission complete
            assert gen_request_1.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            assert gen_request_2.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE

            # Verify that no actual KV cache transfer was initiated
            mock_executor_components[
                'kv_cache_transceiver'].request_and_receive_async.assert_not_called(
                )
            mock_executor_components[
                'kv_cache_transceiver'].request_and_receive_sync.assert_not_called(
                )

    def test_prepare_disagg_gen_transmission_complete(self,
                                                      mock_executor_components):
        """
        Test _prepare_disagg_gen_transmission_complete for setting first generation tokens.

        This test verifies that first generation tokens are properly set for
        disaggregated generation requests that have completed transmission.
        """
        # Create PyExecutor
        executor = PyExecutor(
            resource_manager=mock_executor_components['resource_manager'],
            scheduler=mock_executor_components['scheduler'],
            model_engine=mock_executor_components['model_engine'],
            sampler=mock_executor_components['sampler'],
            dist=mock_executor_components['dist'],
            max_num_sequences=8,
            start_worker=False)

        # Create mock requests with transmission complete
        gen_request_1 = create_mock_disagg_gen_request(1,
                                                       first_gen_tokens=[7, 8])
        gen_request_1.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        gen_request_1.is_disagg_generation_transmission_complete = True
        gen_request_1.sampling_config = Mock()
        gen_request_1.sampling_config.num_beams = 2

        gen_request_2 = create_mock_disagg_gen_request(2, first_gen_tokens=[9])
        gen_request_2.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
        gen_request_2.is_disagg_generation_transmission_complete = True
        gen_request_2.sampling_config = Mock()
        gen_request_2.sampling_config.num_beams = 1

        # Create scheduled batch
        scheduled_batch = create_scheduled_batch_with_disagg_gen_requests(
            [gen_request_1, gen_request_2])

        # Call the method
        executor._prepare_disagg_gen_transmission_complete(scheduled_batch)

        # Verify that add_new_token was called with the correct first generation tokens
        # For gen_request_1 with 2 beams and first_gen_tokens=[7, 8]
        gen_request_1.add_new_token.assert_any_call(7, 0)  # beam 0 gets token 7
        gen_request_1.add_new_token.assert_any_call(8, 1)  # beam 1 gets token 8

        # For gen_request_2 with 1 beam and first_gen_tokens=[9]
        gen_request_2.add_new_token.assert_called_once_with(
            9, 0)  # beam 0 gets token 9

    def test_executor_loop_handles_empty_batch(self, mock_executor_components,
                                               mock_executor_request_queue):
        """
        Test that executor_loop handles empty batches gracefully.
        """
        with patch.dict(os.environ, {'TRTLLM_DISAGG_BENCHMARK_GEN_ONLY': '1'}):
            executor = PyExecutor(
                resource_manager=mock_executor_components['resource_manager'],
                scheduler=mock_executor_components['scheduler'],
                model_engine=mock_executor_components['model_engine'],
                sampler=mock_executor_components['sampler'],
                dist=mock_executor_components['dist'],
                max_num_sequences=8,
                kv_cache_transceiver=mock_executor_components[
                    'kv_cache_transceiver'],
                start_worker=False)

            executor.executor_request_queue = mock_executor_request_queue
            executor.active_requests = []

            # Create empty scheduled batch
            empty_scheduled_batch = create_scheduled_batch_with_disagg_gen_requests(
                [])
            empty_scheduled_batch.batch_size = 0

            with patch.object(
                    executor,
                    '_prepare_and_schedule_batch') as mock_prepare_schedule:
                mock_prepare_schedule.side_effect = [
                    (empty_scheduled_batch,
                     None),  # First iteration: empty batch
                    (None, None)  # Second iteration: exit loop
                ]

                with patch.object(executor, '_handle_canceled_requests') as mock_handle_canceled, \
                     patch.object(executor, '_handle_responses') as mock_handle_responses, \
                     patch.object(executor, '_terminate_ctx_finished_requests') as mock_terminate_ctx, \
                     patch.object(executor, '_kv_connector_terminate_requests') as mock_kv_terminate, \
                     patch('torch.cuda.set_device'), \
                     patch('tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT'):

                    mock_handle_responses.return_value = []
                    executor.device_id = 0
                    executor.enable_iter_perf_stats = False
                    executor.enable_attention_dp = False
                    executor.enable_kv_cache_events = False
                    executor.ctx_in_transmission_requests = []
                    executor.drafter = None
                    executor.guided_decoder = None
                    executor.kv_connector_manager = None
                    executor._profiler = Mock()
                    executor._profiler.return_value.__enter__ = Mock(
                        return_value=Mock())
                    executor._profiler.return_value.__exit__ = Mock(
                        return_value=None)

                    # Run the executor loop
                    executor._executor_loop()

                    # Verify that basic cleanup operations were still performed
                    mock_handle_canceled.assert_called_once()
                    mock_handle_responses.assert_called_once()
