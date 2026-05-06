#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build the TinyLlama TRT engine for the NVBug 5615248 benchmark.

Mirrors the engine geometry baked into the original
``nvbugs_5615248/tinyllama_trt_engine`` (max_seq_len=129, max_batch_size=1,
max_beam_width=10, max_input_len=129, max_num_tokens=129,
paged_context_fmha=true) so the PyT-vs-TRT comparison stays apples-to-apples.

Run once inside the TRT-LLM container, from the repo root::

    python3 nvbugs_5615248/trtllm_bench/build_trt_engine.py \\
        --model /home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0 \\
        --output nvbugs_5615248/tinyllama_trt_engine

Subsequent ``run_multirun_trt.sh`` / ``run_nsys_trace.sh tensorrt`` invocations
will pick up ``--engine_dir nvbugs_5615248/tinyllama_trt_engine`` automatically.
"""

from __future__ import annotations

import argparse
import os
import sys

from tensorrt_llm import BuildConfig
from tensorrt_llm._tensorrt_engine import LLM as TrtLLM


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model dir for TinyLlama")
    p.add_argument(
        "--output",
        default="nvbugs_5615248/tinyllama_trt_engine",
        help="Where to save the built engine (default: nvbugs_5615248/tinyllama_trt_engine)",
    )
    p.add_argument("--max_batch_size", type=int, default=1)
    p.add_argument("--max_beam_width", type=int, default=10)
    p.add_argument("--max_seq_len", type=int, default=129)
    p.add_argument("--max_input_len", type=int, default=129)
    p.add_argument("--max_num_tokens", type=int, default=129)
    args = p.parse_args()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.isdir(output) and any(
        f.startswith("rank") and f.endswith(".engine") for f in os.listdir(output)
    ):
        print(f"[build_trt_engine] engine already present at {output}, skipping build.",
              file=sys.stderr)
        return 0

    bc = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_beam_width,
        max_seq_len=args.max_seq_len,
        max_input_len=args.max_input_len,
        max_num_tokens=args.max_num_tokens,
    )
    # Match the prebuilt-engine config baked in REPRO.md (paged_context_fmha=true).
    bc.plugin_config.use_paged_context_fmha = True

    print(f"[build_trt_engine] building TRT engine for {args.model} -> {output}",
          file=sys.stderr)
    llm = TrtLLM(
        model=args.model,
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_beam_width,
        max_seq_len=args.max_seq_len,
        build_config=bc,
    )
    try:
        llm.save(output)
        print(f"[build_trt_engine] engine saved at {output}", file=sys.stderr)
    finally:
        del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
