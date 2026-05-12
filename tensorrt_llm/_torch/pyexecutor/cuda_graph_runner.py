import bisect
import contextlib
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeAlias

import torch

from tensorrt_llm.llmapi.llm_args import (BaseSparseAttentionConfig,
                                          DecodingBaseConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...inputs.multimodal import MultimodalParams
from ..distributed import Distributed
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import get_memory_buffers
from ..modules.multi_stream_utils import with_multi_stream
from ..speculative.eagle3 import Eagle3ResourceManager
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..speculative.utils import get_draft_kv_cache_manager
from ..utils import make_weak_ref, piecewise_cuda_graph
from .llm_request import get_draft_token_length
from .resource_manager import (BaseResourceManager, ResourceManager,
                               ResourceManagerType)
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests

# A large prime number used for dummy request IDs to avoid collisions
CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 64) - 1
KeyType: TypeAlias = Tuple[int, int, bool, bool]

# Prefill CUDA graph PoC: hardcoded shape (1 ctx req, 100 tokens). Gated on
# at process start by ``TLLM_ENABLE_PREFILL_CUDA_GRAPH=1``; behavior is
# byte-identical to today when the env var is unset.
PREFILL_GRAPH_NUM_TOKENS = 100
PREFILL_GRAPH_BATCH_SIZE = 1
PREFILL_GRAPH_ENV_VAR = "TLLM_ENABLE_PREFILL_CUDA_GRAPH"
# Distinct dummy request id for the prefill CUDA graph warmup so it does not
# collide with the decode-graph dummy requests.
PREFILL_CUDA_GRAPH_DUMMY_REQUEST_ID = (1 << 63) - 1


@dataclass
class CUDAGraphRunnerConfig:
    """Configuration for the CUDAGraphRunner, passed from the ModelEngine."""
    use_cuda_graph: bool
    """
    Master switch controlling the model's execution path.

    This flag determines one of three distinct execution paths for the
    model engine:

    1.  **`False` (Pure Eager Path):**
        * Forces all execution to be in eager mode.
        * The `CUDAGraphRunner` instance is mostly dormant
        * Methods like `maybe_get_cuda_graph` and `pad_batch`
            will return immediately, signaling the model engine to
            run in eager mode.

    2.  **`True` (Eager Fallback Path):**
        * The runner is active and checks for graph eligibility.
        * If a batch is ineligible (e.g., it's a prefill batch,
            stats collection is on, or it's an unsupported batch size),
            the runner signals a fallback to eager mode for that batch.

    3.  **`True` (CUDA Graph Path):**
        * The runner finds an eligible batch and a matching graph.
        * The graph is then captured (if new) or replayed.

    Note: As of this implementation, the model engine *always* calls
    `cuda_graph_runner.pad_batch` and `cuda_graph_runner.maybe_get_cuda_graph`
    even when this is `False`. This could be refactored in the future
    so that the engine bypasses the `CUDAGraphRunner` entirely in Case 1.
    """
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    max_cuda_graph_batch_size: int
    max_beam_width: int
    max_num_tokens: int
    spec_config: Optional[DecodingBaseConfig]
    cuda_graph_mem_pool: Any
    use_mrope: bool
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    is_draft_model: bool
    enable_attention_dp: bool
    batch_size: int
    mapping: Optional[Mapping]
    dist: Optional[Distributed]
    kv_cache_manager_key: Any
    dynamic_draft_len_mapping: Optional[Dict[int, int]] = None
    sparse_attention_config: Optional[BaseSparseAttentionConfig] = None


class CUDAGraphRunner:
    """
    Manages the lifecycle and execution of CUDA graphs for the model engine.

    This unified class handles high-level orchestration (padding, eligibility)
    and low-level execution (capturing, resource management, replaying) for
    multiple graphs, keyed by (batch size, draft_len, is_first_draft).
    """
    WARMUP_STEPS = 2

    def __init__(self, config: CUDAGraphRunnerConfig):
        self.config = config

        # High-level configuration from the config object
        self.enabled = config.use_cuda_graph
        self.padding_enabled = config.cuda_graph_padding_enabled
        self.supported_batch_sizes = config.cuda_graph_batch_sizes
        self.max_supported_batch_size = config.max_cuda_graph_batch_size
        self.max_beam_width = config.max_beam_width
        self.spec_config = config.spec_config
        self.sparse_config = config.sparse_attention_config

        self.graphs: Dict[KeyType, torch.cuda.CUDAGraph] = {}
        self.graph_outputs: Dict[KeyType,
                                 Callable[[], Optional[torch.Tensor]]] = {}
        self.graph_metadata: Dict[KeyType, Dict[str, Any]] = {}
        self.memory_pool = config.cuda_graph_mem_pool
        self.padding_dummy_requests: Dict[int, "Request"] = {}
        self.dynamic_draft_len_mapping = config.dynamic_draft_len_mapping

        self.shared_static_tensors: Dict[str, torch.Tensor] = {}
        if self.enabled:
            self._create_shared_static_tensors()
        self.cuda_graph_meta_buffers = get_memory_buffers()

        # On-the-fly capture is disabled by default to prevent workspace
        # tensor reallocation from invalidating addresses baked into existing
        # CUDA graphs.  Use allow_capture() context manager during warmup.
        self._capture_allowed = False

        # ---- Prefill CUDA graph PoC state -----------------------------------
        # Opt-in proof-of-concept: capture/replay a single full-forward CUDA
        # graph for a context-only iteration of exactly
        # PREFILL_GRAPH_NUM_TOKENS tokens (BS=1). Gated by env var so
        # behavior is byte-identical to today when disabled. Decode-graph
        # data structures above are not touched.
        self.prefill_enabled: bool = self.enabled and bool(
            int(os.environ.get(PREFILL_GRAPH_ENV_VAR, "0")))
        self.prefill_num_tokens: int = PREFILL_GRAPH_NUM_TOKENS
        if self.prefill_enabled:
            logger.info(
                f"[prefill graph] PoC enabled "
                f"(num_tokens={self.prefill_num_tokens}); set "
                f"{PREFILL_GRAPH_ENV_VAR}=0 to disable.")
        self.prefill_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.prefill_graph_outputs: Dict[int, Callable[[],
                                                       Optional[
                                                           torch.Tensor]]] = {}
        self.prefill_graph_metadata: Dict[int, Dict[str, Any]] = {}
        # Static input buffers reused across prefill graph replays.
        if self.prefill_enabled:
            self.prefill_static_input_ids = torch.ones(
                (self.prefill_num_tokens, ),
                device="cuda",
                dtype=torch.int32)
            self.prefill_static_position_ids = torch.zeros(
                (1, self.prefill_num_tokens),
                device="cuda",
                dtype=torch.int32)
        else:
            self.prefill_static_input_ids = None
            self.prefill_static_position_ids = None

    def _create_shared_static_tensors(self):
        """Allocates static tensors sized for the largest possible batch."""
        max_draft_len = self.config.original_max_total_draft_tokens if self.config.spec_config is not None else 0
        token_per_request = max_draft_len + 1
        max_total_tokens = (self.max_supported_batch_size *
                            self.max_beam_width * token_per_request)
        max_total_tokens = min(max_total_tokens, self.config.max_num_tokens)

        self.shared_static_tensors = {
            "input_ids":
            torch.ones((max_total_tokens, ), device="cuda", dtype=torch.int32),
            "position_ids":
            torch.zeros((1, max_total_tokens), device="cuda",
                        dtype=torch.int32),
        }
        if self.config.use_mrope:
            self.shared_static_tensors["position_ids"] = torch.zeros(
                (3, 1, max_total_tokens), device="cuda", dtype=torch.int32)
            self.shared_static_tensors["multimodal_params"] = [
                MultimodalParams(
                    multimodal_data={
                        "mrope_config": {
                            "mrope_position_deltas":
                            torch.zeros(
                                (1, 1), device="cuda", dtype=torch.int32)
                        }
                    }) for _ in range(max_total_tokens)
            ]

    def _get_seq_len_mode(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None):
        if self.sparse_config is not None and self.sparse_config.needs_separate_short_long_cuda_graphs(
        ):
            # Some sparse attention algorithms need to use different forward paths for short and long sequences.
            # For example, the DSA can skip the MQA and Top-K in the indexer for short sequences to reduce the
            # computational overhead. To support this feature, we need to capture separate CUDA graphs for short
            # and long sequences. We need to first collect the sequence length of the requests and then determine
            # the sequence length mode. For long sequences, use the default maximum sequence length. For short
            # sequences, use the sequence length threshold as the maximum sequence length.
            total_seq_lens = []
            new_tokens_device, next_draft_tokens_device = None, None
            if new_tensors_device is not None:
                new_tokens_device = new_tensors_device.new_tokens
                if isinstance(new_tensors_device, SampleStateTensorsSpec):
                    next_draft_tokens_device = new_tensors_device.next_draft_tokens
            overlap_scheduler_enabled = new_tokens_device is not None
            for request in batch.generation_requests:
                is_spec_request = get_draft_token_length(
                    request) > 0 or next_draft_tokens_device is not None
                num_draft_tokens = self.spec_config.max_draft_len if is_spec_request else 0
                # First draft
                if request.py_is_first_draft:
                    total_seq_len = len(request.get_tokens(0))
                # With overlap scheduler disabled or dummy request or not assigned to a batch,
                elif not overlap_scheduler_enabled or request.is_dummy or request.py_batch_idx is None:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens
                # Other cases
                else:
                    total_seq_len = request.max_beam_num_tokens + num_draft_tokens + 1
                total_seq_lens.append(total_seq_len)
            # Determine the sequence length mode.
            from ..speculative import get_num_extra_kv_tokens
            num_extra_kv_tokens = get_num_extra_kv_tokens(self.spec_config)
            max_seq_len = max(total_seq_lens)
            if max_seq_len <= self.sparse_config.seq_len_threshold - num_extra_kv_tokens:
                short_seq_len_mode = True
            else:
                short_seq_len_mode = False
        else:
            # For non-sparse attention or sparse attention that does not need separate short and long CUDA graphs,
            # use the default sequence length mode.
            short_seq_len_mode = False
        return short_seq_len_mode

    def get_graph_key(
            self,
            batch: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            spec_resource_manager: Optional[BaseResourceManager] = None):
        batch_size = batch.batch_size

        # Get the sequence length mode.
        short_seq_len_mode = self._get_seq_len_mode(batch, new_tensors_device)

        if self.config.is_draft_model and spec_resource_manager is not None and isinstance(
                spec_resource_manager, Eagle3ResourceManager):
            # If 'is_first_draft' is True, even with tree decoding, the length of draft_len will only be 'max_draft_len', not 'max_total_draft_token'.
            # Because we will pad the input to 'max_draft_len' length for the first draft layer.
            draft_len = self.config.original_max_draft_len if spec_resource_manager.is_first_draft else 0
            key = (batch_size, draft_len, spec_resource_manager.is_first_draft,
                   short_seq_len_mode)
        else:
            # With dynamic spec decode, the draft length may be zero even when enable_spec_decode is True,
            # so we need to get the draft length from the batch instead of using enable_spec_decode.
            draft_len_list = []
            for request in batch.generation_requests:
                draft_len_list.append(len(request.py_draft_tokens))
            draft_len = max(draft_len_list)
            assert len(
                set(draft_len_list)) == 1, "All draft lengths must be the same"
            key = (batch_size, draft_len, False, short_seq_len_mode)
        return key

    def __del__(self):
        self.clear()

    def maybe_get_cuda_graph(
        self,
        batch: ScheduledRequests,
        enable_spec_decode: bool,
        attn_metadata: Any,
        spec_metadata: Optional[Any] = None,
        draft_tokens_cuda: Optional[torch.Tensor] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_resource_manager: Optional[BaseResourceManager] = None,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Tuple[int, int, bool]]]:
        """
        Determines if the current batch can be run with a CUDA graph.

        Returns a tuple containing:
        - The attn_metadata for the graph, if applicable.
        - The spec_metadata for the graph, if applicable.
        - The key for the graph, if applicable.
        """
        # disable when doing statistic
        if ExpertStatistic.should_record():
            return None, None, None

        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            all_can_graph_batch = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            is_all_gen_only = all(all_can_graph[0]
                                  for all_can_graph in all_can_graph_batch)
            all_batch_size_equal = all(
                all_gen_only[1] == all_can_graph_batch[0][1]
                for all_gen_only in all_can_graph_batch)

            if not is_all_gen_only or not all_batch_size_equal:
                return None, None, None

        if not self.enabled or not can_run_cuda_graph:
            return None, None, None
        key = self.get_graph_key(batch, new_tensors_device,
                                 spec_resource_manager)

        if key in self.graphs:
            return self.graph_metadata[key][
                "attn_metadata"], self.graph_metadata[key]["spec_metadata"], key

        # Graph doesn't exist yet.  If on-the-fly capture is not allowed,
        # fall back to eager so the caller doesn't need a separate check.
        if not self._capture_allowed:
            return None, None, None

        if batch_size not in self.supported_batch_sizes:
            return None, None, None

        num_sequences_in_batch = batch_size * self.max_beam_width
        graph_attn_metadata = attn_metadata.create_cuda_graph_metadata(
            num_sequences_in_batch, False, key[1], self.cuda_graph_meta_buffers)
        assert graph_attn_metadata.is_cuda_graph

        if enable_spec_decode:
            graph_spec_metadata = spec_metadata.create_cuda_graph_metadata(
                num_sequences_in_batch)
            graph_spec_metadata.draft_tokens = draft_tokens_cuda
        else:
            graph_spec_metadata = None
        return graph_attn_metadata, graph_spec_metadata, key

    def needs_capture(self, key: KeyType):
        return self._capture_allowed and key not in self.graph_outputs

    @contextlib.contextmanager
    def allow_capture(self):
        """Context manager that enables CUDA graph capture.

        Capture is disabled by default.  On-the-fly captures outside this
        context are prevented because they can resize the shared
        cuda_graph_workspace tensor, invalidating addresses baked into
        previously captured graphs.
        """
        self._capture_allowed = True
        try:
            yield
        finally:
            self._capture_allowed = False

    def get_graph_pool(self):
        """Returns the CUDA memory pool used by this graph runner.

        Returns:
            The CUDA memory pool associated with captured graphs, or None if
            no graphs have been captured yet.
        """
        return self.memory_pool

    def capture(self,
                key: KeyType,
                forward_fn: Callable,
                initial_inputs: Dict[str, Any],
                enable_spec_decode: bool = False,
                postprocess_fn: Optional[Callable] = None):
        """Captures the forward pass for a given batch size."""
        batch_size = key[0]
        # [CUDA graph spec decode padding]
        # We pad input IDs/position IDs to the maximum draft length (token per request).
        # We're forced to do this because we cannot reallocate inputs over many graph runs.
        max_draft_len = key[1]
        token_per_request = max_draft_len + 1
        num_tokens_for_capture = (batch_size * self.max_beam_width *
                                  token_per_request)

        sliced_static_tensors = {
            "input_ids":
            self.shared_static_tensors["input_ids"][:num_tokens_for_capture],
            "position_ids":
            self.shared_static_tensors["position_ids"]
            [:, :num_tokens_for_capture],
        }
        if self.config.use_mrope:
            sliced_static_tensors["position_ids"] = self.shared_static_tensors[
                "position_ids"][:, :, :num_tokens_for_capture]
            sliced_static_tensors[
                "multimodal_params"] = self.shared_static_tensors[
                    "multimodal_params"][:batch_size * self.max_beam_width]

        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
            "spec_metadata": initial_inputs.get("spec_metadata", None),
        }

        def _setup_spec_decoding_and_forward(key: KeyType, forward_fn: Callable,
                                             capture_inputs: Dict[str, Any]):
            is_first_draft = key[2]
            needs_kv_cache_recompute = True if enable_spec_decode and self.config.spec_config.spec_dec_mode.needs_kv_cache_recompute(
            ) else False
            if is_first_draft and self.config.is_draft_model and needs_kv_cache_recompute:
                capture_inputs['attn_metadata'].use_spec_decoding = True
            return forward_fn(capture_inputs)

        # We have to do warm up runs to initialize PyTorch's
        # internal states according to the docs:
        # https://pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics
        # This also lets us initialize states in the attn_metadata.
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                _setup_spec_decoding_and_forward(key, forward_fn,
                                                 capture_inputs)
                if postprocess_fn is not None:
                    postprocess_fn(capture_inputs)

            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = _setup_spec_decoding_and_forward(
                    key, forward_fn, capture_inputs)
            if postprocess_fn is not None:
                postprocess_fn(capture_inputs)

        self.graphs[key] = graph
        self.graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    # ---------------------------------------------------------------- Prefill
    #
    # The methods below implement an opt-in PoC that captures a full-forward
    # CUDA graph for a 1-context-request / PREFILL_GRAPH_NUM_TOKENS-token
    # prefill iteration. They are intentionally kept separate from the decode
    # path so the existing decode graph behavior is unchanged.

    def maybe_get_prefill_cuda_graph(
        self,
        batch: ScheduledRequests,
        attn_metadata: Any,
    ) -> Tuple[Optional[Any], Optional[int]]:
        """Determine whether the current iteration matches the prefill PoC.

        Returns ``(graph_attn_metadata, key)`` if the iteration is eligible
        for prefill graph capture/replay (or already cached), else
        ``(None, None)`` so the caller falls through to the decode path /
        eager / piecewise.
        """
        if not self.prefill_enabled:
            return None, None

        # Respect transient disables (e.g. ``ModelEngine.no_cuda_graph`` is
        # active during piecewise warmup) so the prefill path doesn't
        # interfere with other capture flows that share this runner.
        if not self.enabled:
            return None, None
        # Only the TRTLLM attention backend has been validated for prefill
        # graph replay (FlashInfer / Star explicitly assert
        # not is_cuda_graph in the prefill path).
        if attn_metadata.__class__.__name__ != "TrtllmAttentionMetadata":
            return None, None
        if not batch.can_run_prefill_cuda_graph(self.prefill_num_tokens):
            return None, None

        key = self.prefill_num_tokens
        if key in self.prefill_graphs:
            graph_attn_metadata = self.prefill_graph_metadata[key][
                "attn_metadata"]
        elif self._capture_allowed:
            graph_attn_metadata = (
                attn_metadata.create_cuda_graph_metadata_prefill(
                    num_tokens=self.prefill_num_tokens,
                    buffers=self.cuda_graph_meta_buffers,
                ))
            assert graph_attn_metadata.is_cuda_graph
        else:
            # Graph not captured yet, and on-the-fly capture is disallowed
            # outside the warmup window.
            return None, None

        # The single user-facing diagnostic: prints the actual context
        # chunk size on every iteration where the prefill graph is picked.
        ctx_reqs = (batch.context_requests_last_chunk +
                    batch.context_requests_chunking)
        actual = ctx_reqs[0].context_chunk_size if ctx_reqs else None
        logger.info(f"[prefill graph] selected: "
                    f"context_chunk_size={actual} key={key}")
        return graph_attn_metadata, key

    def needs_capture_prefill(self, key: int) -> bool:
        return self._capture_allowed and key not in self.prefill_graph_outputs

    def capture_prefill(
        self,
        key: int,
        forward_fn: Callable,
        initial_inputs: Dict[str, Any],
    ) -> None:
        """Capture a prefill CUDA graph for ``key`` tokens.

        ``key`` must equal ``self.prefill_num_tokens`` (PoC supports a
        single configured shape per process).
        """
        assert self.prefill_enabled
        num_tokens = key
        assert num_tokens == self.prefill_num_tokens, (
            "Prefill CUDA graph PoC only supports "
            f"{self.prefill_num_tokens} tokens per process (got {num_tokens})")

        sliced_static_tensors = {
            "input_ids": self.prefill_static_input_ids,
            "position_ids": self.prefill_static_position_ids,
        }
        capture_inputs = initial_inputs.copy()
        capture_inputs.update(sliced_static_tensors)

        self.prefill_graph_metadata[key] = {
            "attn_metadata": initial_inputs["attn_metadata"],
        }

        # Mirror the decode capture path: warmup runs initialize PyTorch's
        # internal state (and the attn_metadata) before capture, and we
        # disable piecewise CUDA graph capture so that path doesn't fight
        # with the full-forward graph.
        graph = torch.cuda.CUDAGraph()
        with with_multi_stream(True), piecewise_cuda_graph(False):
            for _ in range(self.WARMUP_STEPS):
                forward_fn(capture_inputs)
            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = forward_fn(capture_inputs)

        self.prefill_graphs[key] = graph
        self.prefill_graph_outputs[key] = make_weak_ref(output)
        self.memory_pool = graph.pool()

    def replay_prefill(
        self,
        key: int,
        current_inputs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Replay a previously captured prefill graph for ``key`` tokens."""
        stored_meta = self.prefill_graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]

        num_tokens = key
        input_ids = current_inputs["input_ids"]
        assert input_ids.shape[0] == num_tokens, (
            f"Prefill graph expects {num_tokens} tokens, "
            f"got {input_ids.shape[0]}")
        self.prefill_static_input_ids.copy_(input_ids, non_blocking=True)

        position_ids = current_inputs["position_ids"]
        # The model engine produces position_ids with shape (1, num_tokens)
        # for the standard (non-mrope) path; mrope is intentionally not
        # supported by this PoC.
        self.prefill_static_position_ids[:, :num_tokens].copy_(
            position_ids[:, :num_tokens], non_blocking=True)

        torch.cuda.nvtx.range_push("prefill_cuda_graph_replay")
        try:
            self.prefill_graphs[key].replay()
        finally:
            torch.cuda.nvtx.range_pop()
        return self.prefill_graph_outputs[key]

    def replay(self, key: KeyType,
               current_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Replays a previously captured graph."""
        stored_meta = self.graph_metadata[key]
        assert current_inputs["attn_metadata"] is stored_meta["attn_metadata"]
        if stored_meta["spec_metadata"] is not None:
            assert current_inputs.get(
                "spec_metadata") is stored_meta["spec_metadata"]

        static_tensors = self.shared_static_tensors

        input_ids = current_inputs["input_ids"]
        seqlen = input_ids.shape[0]
        static_tensors["input_ids"][:seqlen].copy_(input_ids)

        position_ids = current_inputs["position_ids"]
        if self.config.use_mrope and current_inputs.get(
                'multimodal_params') is not None:
            static_tensors["position_ids"][:, :, :seqlen].copy_(position_ids)
            for i, multimodal_param in enumerate(
                    current_inputs['multimodal_params']):
                # NOTE: Currently, we only need 'mrope_position_deltas' on generation phase for multimodal models.
                static_tensors['multimodal_params'][i].multimodal_data[
                    'mrope_config']['mrope_position_deltas'].copy_(
                        multimodal_param.multimodal_data['mrope_config']
                        ['mrope_position_deltas'],
                        non_blocking=True)
        else:
            static_tensors["position_ids"][:, :seqlen].copy_(position_ids)

        self.graphs[key].replay()
        output_ref = self.graph_outputs[key]

        return output_ref

    def _get_padded_batch(self, batch: ScheduledRequests,
                          resource_manager: ResourceManager,
                          runtime_draft_len: int) -> int:
        kv_cache_manager = resource_manager.get_resource_manager(
            self.config.kv_cache_manager_key)
        can_run_cuda_graph = batch.can_run_cuda_graph
        batch_size = batch.batch_size
        new_batch_size = batch_size

        if self.enabled and self.config.enable_attention_dp and self.config.mapping.tp_size > 1:
            graph_batch_size = self.config.dist.tp_allgather(
                [can_run_cuda_graph, batch_size])
            all_can_graph = all(graph_batch[0]
                                for graph_batch in graph_batch_size)
            if all_can_graph:
                new_batch_size = max(gen_only_batch[1]
                                     for gen_only_batch in graph_batch_size)

        if (not self.enabled or not self.padding_enabled
                or not can_run_cuda_graph
                or new_batch_size > self.max_supported_batch_size):
            return 0

        # When dynamic draft length is enabled (one-model path), we treat the determined runtime draft length
        # as the source of truth and pad the batch size up to the nearest existing graph
        # for that draft length.
        if (self.spec_config and self.spec_config.draft_len_schedule
                and self.spec_config.spec_dec_mode.support_dynamic_draft_len()):
            padded_batch_size = self._round_up_batch_size_with_draft_len(
                new_batch_size, runtime_draft_len)
        else:
            padded_batch_size = self._round_up_batch_size(new_batch_size)

        if batch_size == padded_batch_size:
            return 0

        padding_size = padded_batch_size - batch_size
        if padding_size <= 0:
            return 0
        if padding_size + batch.batch_size > self.config.batch_size:
            return 0

        # No padding if it would create too many concurrent requests.
        # This is not strictly required, but we should probably
        # respect the requirement just in case that changes in the future.
        # Use per-draft-len dummy requests for dynamic draft length support.
        if runtime_draft_len not in self.padding_dummy_requests:

            # Get draft KV cache manager only for one-model speculative decoding.
            # In two-model mode, each model has its own KV cache manager, so
            # draft_kv_cache_manager should be None.
            draft_kv_cache_manager = get_draft_kv_cache_manager(
                self.spec_config, resource_manager)

            # Use unique dummy request ID per draft length
            dummy_request_id = CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len
            dummy_request = kv_cache_manager.add_dummy_requests(
                [dummy_request_id],
                is_gen=True,
                max_num_draft_tokens=runtime_draft_len,
                use_mrope=self.config.use_mrope,
                max_beam_width=self.config.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager)

            if dummy_request is None:
                return 0
            else:
                dummy_request = dummy_request[0]
            dummy_request.is_cuda_graph_dummy = True

            spec_res_mgr = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_res_mgr:
                spec_res_mgr.add_dummy_requests([dummy_request_id])
            self.padding_dummy_requests[runtime_draft_len] = dummy_request

        padding_dummy_request = self.padding_dummy_requests[runtime_draft_len]
        batch.generation_requests.extend([padding_dummy_request] * padding_size)
        return padding_size

    def _round_up_batch_size(self, batch_size: int) -> int:
        """Finds the smallest supported graph batch size >= the given size."""
        if not self.supported_batch_sizes:
            return 0
        idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        if idx == len(self.supported_batch_sizes):
            return 0
        return self.supported_batch_sizes[idx]

    def _round_up_batch_size_with_draft_len(self, batch_size: int,
                                            draft_len: int) -> int:
        """Finds the smallest graph batch size >= batch_size that also matches the given draft_len."""
        if not self.dynamic_draft_len_mapping:
            # Fallback to regular round up if no mapping
            return self._round_up_batch_size(batch_size)

        start_idx = bisect.bisect_left(self.supported_batch_sizes, batch_size)
        # Negate the list to make it non-decreasing for bisect
        # (draft_len decreases as batch_size increases in the schedule)
        draft_lens = [
            self.dynamic_draft_len_mapping.get(self.supported_batch_sizes[i], 0)
            for i in range(start_idx, len(self.supported_batch_sizes))
        ]
        idx = bisect.bisect_left(draft_lens, -draft_len, key=lambda x: -x)
        if idx < len(draft_lens) and draft_lens[idx] == draft_len:
            return self.supported_batch_sizes[start_idx + idx]
        # No suitable graph found
        return 0

    @contextlib.contextmanager
    def pad_batch(self,
                  scheduled_requests: ScheduledRequests,
                  resource_manager: ResourceManager,
                  runtime_draft_len: int = 0):
        """Context manager to pad a batch to a graph-compatible size."""
        padding_size = self._get_padded_batch(scheduled_requests,
                                              resource_manager,
                                              runtime_draft_len)
        try:
            yield scheduled_requests
        finally:
            if padding_size > 0:
                scheduled_requests.generation_requests = scheduled_requests.generation_requests[:
                                                                                                -padding_size]

    def clear(self):
        """Releases all captured graphs and the associated memory pool."""
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_outputs.clear()
        self.graph_metadata.clear()
        for prefill_graph in self.prefill_graphs.values():
            prefill_graph.reset()
        self.prefill_graphs.clear()
        self.prefill_graph_outputs.clear()
        self.prefill_graph_metadata.clear()
        self.padding_dummy_requests = {}
        del self.memory_pool
        self.memory_pool = None
        torch.cuda.empty_cache()
