"""TTFT profiling driver for the TensorRT (legacy) backend — NVBug 5615248.

Mirrors ``profile_ttft.py`` but builds/loads a TensorRT engine instead of the
PyTorch backend. Loads a YAML config (same format as
``trtllm-serve --extra_llm_api_options``), warms up the engine, then profiles
a single streaming request with the CUDA profiler API. Designed to be wrapped
in nsys so the profile contains just the measurement request, not the long
warmup / engine-build activity.

The TRT backend cannot use any of the PyTorch-only YAML keys
(``backend: pytorch``, ``cuda_graph_config``, ``torch_compile_config``).
For TRT-side CUDA graphs, set ``extended_runtime_perf_knob_config.cuda_graph_mode: true``.

Two ways to provide the engine via ``--model``:
  * HF model dir → _TrtLLM builds a TensorRT engine in-process using
    ``BuildConfig`` (slow startup, but a single self-contained command).
  * Pre-built engine dir → engine is loaded directly (recommended for
    profiling; build it once with ``llm.save(...)`` or ``trtllm-build``).

Usage (run *under* nsys):

    nsys profile \\
        --capture-range=cudaProfilerApi --capture-range-end=stop \\
        -t cuda,nvtx,osrt \\
        -o trace_trt \\
        -f true \\
        python profile_ttft_trt.py \\
            --config profile_trt.yaml \\
            --model /path/to/TinyLlama-1.1B-Chat-v1.0 \\
            --tag trt

Why nsys + cudaProfilerApi:
    The python helper calls torch.cuda.profiler.start()/stop() around the
    measurement request only. nsys's --capture-range=cudaProfilerApi limits
    the profile to that window, so the trace is small and the engine
    build / warmup activity is excluded.
"""

import argparse
import time

import torch

from tensorrt_llm import BuildConfig, SamplingParams
from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
from tensorrt_llm.llmapi.llm_args import update_llm_args_with_extra_options


def _build_llm(model: str, config_yaml: str) -> TrtLLM:
    # Do NOT pass backend="..." — _TrtLLM is the TRT-backend LLM class.
    base_args = {"model": model}
    merged = update_llm_args_with_extra_options(base_args, config_yaml)

    # If the YAML doesn't supply a build_config and `model` points at an
    # HF dir, _TrtLLM will need one to build the engine. If `model` points
    # at an already-built engine dir, this is harmless (ignored).
    if "build_config" not in merged:
        merged["build_config"] = BuildConfig(
            max_batch_size=merged.get("max_batch_size") or 1,
            max_beam_width=merged.get("max_beam_width") or 1,
            max_seq_len=merged.get("max_seq_len") or 129,
        )
    return TrtLLM(**merged)


def _build_sampling_params(max_tokens: int, max_beam_width: int) -> SamplingParams:
    use_beam_search = max_beam_width > 1
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        n=max_beam_width if use_beam_search else 1,
        best_of=max_beam_width if use_beam_search else 1,
        use_beam_search=use_beam_search,
        # We never look at decoded text, only token timing. Skipping
        # detokenization avoids an OverflowError in HF tokenizer.decode
        # on intermediate beam-search streaming chunks (placeholder
        # token IDs in not-yet-populated beams).
        detokenize=False,
    )


def _make_prompt(target_tokens: int) -> str:
    # "hello " tokenizes to 2 tokens for TinyLlama, so this gets us close
    # to ISL=107 tokens (the customer's repro length).
    return ("hello " * (target_tokens // 2)).strip()


def _run_request(llm: TrtLLM, prompt: str, sampling_params: SamplingParams):
    """Run one streaming request, return (ttft_s, total_s, n_chunks, n_tokens)."""
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    last_output = None
    for output in llm.generate_async(prompt, sampling_params, streaming=True):
        n_chunks += 1
        if ttft is None:
            ttft = time.perf_counter() - t0
        last_output = output
    total = time.perf_counter() - t0
    n_tokens = len(last_output.outputs[0].token_ids) if last_output else 0
    return ttft, total, n_chunks, n_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="YAML config (same format as trtllm-serve --extra_llm_api_options).")
    parser.add_argument("--model", required=True,
                        help="HF model dir OR pre-built TRT engine dir.")
    parser.add_argument("--tag", required=True,
                        help="Free-form tag printed in the summary, e.g. trt.")
    parser.add_argument("--isl", type=int, default=107,
                        help="Approximate input prompt length in tokens.")
    parser.add_argument("--osl", type=int, default=20,
                        help="Output sequence length (max_tokens).")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup requests before the profiled measurement.")
    parser.add_argument("--measurements", type=int, default=1,
                        help="Number of profiled measurement requests.")
    parser.add_argument("--save-engine", default=None,
                        help="If set, save the built TRT engine to this dir and exit "
                             "(skip warmup/measurement). Useful to pre-build once and "
                             "then point --model at the engine dir for subsequent runs.")
    args = parser.parse_args()

    llm = _build_llm(args.model, args.config)

    if args.save_engine is not None:
        print(f"[{args.tag}] Saving engine to {args.save_engine}", flush=True)
        llm.save(args.save_engine)
        return

    max_beam_width = getattr(llm.args, "max_beam_width", 1) or 1
    sampling_params = _build_sampling_params(args.osl, max_beam_width)
    prompt = _make_prompt(args.isl)

    print(f"[{args.tag}] === Warmup x{args.warmup} ===", flush=True)
    for i in range(args.warmup):
        ttft, total, n_chunks, n_tokens = _run_request(llm, prompt, sampling_params)
        print(f"[{args.tag}] warmup[{i}] TTFT={ttft*1000:.2f}ms "
              f"total={total*1000:.2f}ms chunks={n_chunks} tokens={n_tokens}",
              flush=True)

    torch.cuda.synchronize()
    print(f"[{args.tag}] === Measurement x{args.measurements} (profiled) ===",
          flush=True)
    torch.cuda.profiler.start()
    try:
        results = []
        for i in range(args.measurements):
            torch.cuda.nvtx.range_push(f"{args.tag}_measurement_{i}")
            ttft, total, n_chunks, n_tokens = _run_request(
                llm, prompt, sampling_params)
            torch.cuda.nvtx.range_pop()
            results.append((ttft, total, n_chunks, n_tokens))
            print(f"[{args.tag}] meas[{i}]   TTFT={ttft*1000:.2f}ms "
                  f"total={total*1000:.2f}ms chunks={n_chunks} tokens={n_tokens}",
                  flush=True)
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    if results:
        ttfts = [r[0] for r in results]
        totals = [r[1] for r in results]
        n = len(results)
        print(f"[{args.tag}] === Summary ({n} measurements) ===")
        print(f"[{args.tag}] TTFT  mean={sum(ttfts)/n*1000:.2f}ms "
              f"min={min(ttfts)*1000:.2f}ms max={max(ttfts)*1000:.2f}ms")
        print(f"[{args.tag}] Total mean={sum(totals)/n*1000:.2f}ms "
              f"min={min(totals)*1000:.2f}ms max={max(totals)*1000:.2f}ms")


if __name__ == "__main__":
    main()
