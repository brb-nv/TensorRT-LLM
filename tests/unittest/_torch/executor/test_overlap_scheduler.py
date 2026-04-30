import json
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.llmapi.llm_args import SchedulerConfig


# A test case of mmlu_llama from lm_eval
@pytest.fixture(scope="module")
def test_case():
    with open(Path(__file__).parent / "test_overlap_scheduler_input.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


def create_llm(model_dir,
               disable_overlap_scheduler,
               sampler_type,
               scheduler_config=None,
               enable_block_reuse=False,
               stream_interval=1):
    """Create LLM with specific overlap scheduler setting"""
    if scheduler_config is None:
        scheduler_config = SchedulerConfig()
    pytorch_config = dict(disable_overlap_scheduler=disable_overlap_scheduler,
                          sampler_type=sampler_type)

    trt_kv_cache_config = TRT_KvCacheConfig(
        enable_block_reuse=enable_block_reuse)

    return LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        **pytorch_config,
        kv_cache_config=trt_kv_cache_config,
        max_num_tokens=
        128,  # Only one request longer than max_num_tokens is required to test chunked prefill
        scheduler_config=scheduler_config,
        stream_interval=stream_interval,
    )


@pytest.mark.parametrize("sampler_type", ["TorchSampler", "TRTLLMSampler"])
@pytest.mark.parametrize("use_python_scheduler", [False, True],
                         ids=["cpp_scheduler", "python_scheduler"])
@pytest.mark.parametrize("enable_block_reuse", [False, True],
                         ids=["no_reuse", "block_reuse"])
@pytest.mark.high_cuda_memory
@pytest.mark.mpi_ray_parity
def test_overlap_scheduler_consistency(model_path, test_case, sampler_type,
                                       use_python_scheduler,
                                       enable_block_reuse):
    scheduler_config = SchedulerConfig(
        use_python_scheduler=use_python_scheduler)

    # Test configuration
    prompts = test_case["prompts"]
    max_new_tokens = test_case["max_new_tokens"]
    temperature = test_case["temperature"]
    top_p = test_case["top_p"]
    stop_words = test_case["stop_words"]

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     stop=stop_words,
                                     temperature=temperature,
                                     top_p=top_p,
                                     n=1,
                                     use_beam_search=True)

    # Test with overlap scheduler enabled
    with create_llm(model_path,
                    disable_overlap_scheduler=False,
                    sampler_type=sampler_type,
                    scheduler_config=scheduler_config,
                    enable_block_reuse=enable_block_reuse) as llm:
        outputs_with_overlap = llm.generate(prompts,
                                            sampling_params=sampling_config,
                                            use_tqdm=True)
        texts_with_overlap = [[
            completion.text for completion in request_output.outputs
        ] for request_output in outputs_with_overlap]

    # Test with overlap scheduler disabled
    with create_llm(model_path,
                    disable_overlap_scheduler=True,
                    sampler_type=sampler_type,
                    scheduler_config=scheduler_config,
                    enable_block_reuse=enable_block_reuse) as llm:
        outputs_without_overlap = llm.generate(prompts,
                                               sampling_params=sampling_config,
                                               use_tqdm=True)
        texts_without_overlap = [[
            completion.text for completion in request_output.outputs
        ] for request_output in outputs_without_overlap]

    # Verify outputs are consistent
    for with_overlap, without_overlap in zip(texts_with_overlap,
                                             texts_without_overlap,
                                             strict=True):
        assert with_overlap == without_overlap


@pytest.mark.parametrize("sampler_type", ["TorchSampler", "TRTLLMSampler"])
@pytest.mark.high_cuda_memory
@pytest.mark.mpi_ray_parity
def test_overlap_scheduler_block_reuse_cache_hit(model_path, test_case,
                                                 sampler_type):
    """Verify that blocks are actually reused when sending the same prompt
    twice with the overlap scheduler enabled. Uses a single prompt to avoid
    batch-internal cache hits that could make the cold-cache check flaky."""
    prompt = test_case["prompts"][0]
    max_new_tokens = test_case["max_new_tokens"]
    temperature = test_case["temperature"]
    top_p = test_case["top_p"]
    stop_words = test_case["stop_words"]

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     stop=stop_words,
                                     temperature=temperature,
                                     top_p=top_p,
                                     n=1,
                                     use_beam_search=True)

    with create_llm(model_path,
                    disable_overlap_scheduler=False,
                    sampler_type=sampler_type,
                    enable_block_reuse=True) as llm:
        output_first = llm.generate([prompt],
                                    sampling_params=sampling_config,
                                    use_tqdm=True)[0]
        assert output_first.cached_tokens == 0, (
            "First pass should have no cached tokens (cold cache)")

        output_second = llm.generate([prompt],
                                     sampling_params=sampling_config,
                                     use_tqdm=True)[0]
        assert output_second.cached_tokens > 0, (
            "Second pass should reuse cached blocks")


def _collect_streaming_chunks(llm, prompts, sampling_params):
    results = [
        llm.generate_async(prompt,
                           sampling_params=sampling_params,
                           streaming=True) for prompt in prompts
    ]
    per_request = []
    for result in results:
        chunks = []
        for output in result:
            completion = output.outputs[0]
            chunks.append((completion.text, len(completion.token_ids),
                           completion.finish_reason))
        per_request.append(chunks)
    return per_request


@pytest.mark.parametrize("sampler_type", ["TorchSampler", "TRTLLMSampler"])
@pytest.mark.parametrize("stream_interval", [1, 4],
                         ids=["stream_each", "stream_every_4"])
@pytest.mark.high_cuda_memory
@pytest.mark.mpi_ray_parity
def test_overlap_scheduler_streaming_chunk_parity(model_path, test_case,
                                                  sampler_type,
                                                  stream_interval):
    """Streaming chunks must match between overlap on/off."""
    prompts = test_case["prompts"]
    sampling_params = SamplingParams(max_tokens=test_case["max_new_tokens"],
                                     stop=test_case["stop_words"],
                                     temperature=test_case["temperature"],
                                     top_p=test_case["top_p"],
                                     n=1)

    with create_llm(model_path,
                    disable_overlap_scheduler=False,
                    sampler_type=sampler_type,
                    stream_interval=stream_interval) as llm:
        chunks_with_overlap = _collect_streaming_chunks(llm, prompts,
                                                        sampling_params)

    with create_llm(model_path,
                    disable_overlap_scheduler=True,
                    sampler_type=sampler_type,
                    stream_interval=stream_interval) as llm:
        chunks_without_overlap = _collect_streaming_chunks(
            llm, prompts, sampling_params)

    for i, (with_o, without_o) in enumerate(
            zip(chunks_with_overlap, chunks_without_overlap, strict=True)):
        assert len(with_o) >= 1, f"Request {i}: no chunks with overlap"
        assert len(with_o) == len(without_o), (
            f"Request {i}: chunk count {len(with_o)} != {len(without_o)}")
        for j, (c_with,
                c_without) in enumerate(zip(with_o, without_o, strict=True)):
            assert c_with == c_without, (
                f"Request {i}, chunk {j}: {c_with} != {c_without}")


@pytest.mark.parametrize("sampler_type", ["TorchSampler", "TRTLLMSampler"])
@pytest.mark.high_cuda_memory
@pytest.mark.mpi_ray_parity
def test_overlap_scheduler_streaming_single_token(model_path, sampler_type):
    """First token also terminates: one chunk, finish_reason set."""
    prompts = ["Hello, my name is", "The capital of France is"]
    sampling_params = SamplingParams(max_tokens=1, n=1)

    with create_llm(model_path,
                    disable_overlap_scheduler=False,
                    sampler_type=sampler_type) as llm:
        per_request = _collect_streaming_chunks(llm, prompts, sampling_params)

    for i, chunks in enumerate(per_request):
        assert len(chunks) == 1, f"Request {i}: {len(chunks)} chunks != 1"
        _, token_len, finish_reason = chunks[0]
        assert token_len == 1, f"Request {i}: token len {token_len} != 1"
        assert finish_reason is not None, f"Request {i}: missing finish_reason"


if __name__ == "__main__":
    test_overlap_scheduler_consistency()
