#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Isolated microbench: ``beam_logprobs_prep_triton`` vs eager 9-kernel path.

Question this script answers: at the production beam-search shape
(B=1, K=10, V=32000, bf16) and a few neighbors, is the v6 Triton fusion
faster than the eager PyTorch sequence it replaced?

Context
-------
The end-to-end multi-run (``optimized_v6`` vs ``optimized_v5``) showed v6
*regressing* TTFT by +0.164 ms and E2E by +0.127 ms (pooled n=80, p<0.02).
PREPARE_BEAM_SEARCH_FIX.md hypothesizes an executor-loop sync floor that
absorbs per-step kernel-launch savings. This microbench separates the
"isolated GPU compute + launch cost" question from the floor question:

    - If Triton wins isolated -> regression is the floor; sync removal is
      a prerequisite before keeping v6.
    - If Triton ties or loses isolated -> drop v6; the kernel was never
      buying us anything to begin with.

Eager reference reproduces the exact else-branch in
``beam_search_sampling_batch.bss.logprobs_prep`` (line ~353 of
``tensorrt_llm/_torch/pyexecutor/sampling_utils.py``):

    logprobs = torch.log_softmax(logits, dim=-1)
    logprobs = torch.where(finished_e, -inf, logprobs)
    logprobs[..., 0] = torch.where(finished_mask, 0, logprobs[..., 0])
    logprobs += cum_log_probs.unsqueeze(-1)

Run from the TRT-LLM repo root inside the dev container::

    python3 nvbugs_5615248/trtllm_bench/microbench_logprobs_prep_triton.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.pyexecutor._beam_search_kernels import (
    beam_logprobs_prep_triton,
)


def eager_path(
    logits: torch.Tensor,
    finished_mask: torch.Tensor,
    cum_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Reproduces the eager 9-kernel sequence v6 replaced (CUDA path)."""
    logprobs = torch.log_softmax(logits, dim=-1)
    finished_e = finished_mask.unsqueeze(-1).expand(-1, -1, logits.size(-1))
    logprobs = torch.where(finished_e, float("-inf"), logprobs)
    logprobs[..., 0] = torch.where(finished_mask, 0, logprobs[..., 0])
    logprobs += cum_log_probs.unsqueeze(-1)
    return logprobs


@dataclass
class TimingResult:
    label: str
    per_call_us: list[float]

    @property
    def median_us(self) -> float:
        return statistics.median(self.per_call_us)

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.per_call_us)

    @property
    def stdev_us(self) -> float:
        return statistics.pstdev(self.per_call_us) if len(self.per_call_us) > 1 else 0.0

    def percentile(self, p: float) -> float:
        s = sorted(self.per_call_us)
        idx = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
        return s[idx]


def time_path(
    fn,
    inputs: tuple,
    *,
    label: str,
    warmup_iters: int,
    measure_iters: int,
    n_repeats: int,
) -> TimingResult:
    """Time fn(*inputs) using CUDA events.

    Each repeat records (start, end) around `measure_iters` back-to-back
    calls and divides to get a per-call cost. Reduces event-record noise
    at sub-100us granularity. Returns one per-call cost per repeat.
    """
    for _ in range(warmup_iters):
        fn(*inputs)
    torch.cuda.synchronize()

    per_call_us: list[float] = []
    for _ in range(n_repeats):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(measure_iters):
            fn(*inputs)
        end_evt.record()
        torch.cuda.synchronize()
        per_call_ms = start_evt.elapsed_time(end_evt) / measure_iters
        per_call_us.append(per_call_ms * 1000.0)
    return TimingResult(label=label, per_call_us=per_call_us)


@dataclass
class ShapeBench:
    B: int
    K: int
    V: int
    dtype: torch.dtype
    label: str
    eager: TimingResult
    triton: TimingResult

    @property
    def speedup(self) -> float:
        """Triton speedup over eager: >1 means Triton is faster."""
        return self.eager.median_us / self.triton.median_us

    @property
    def verdict(self) -> str:
        s = self.speedup
        if s > 1.05:
            return "TRITON"
        if s < 0.95:
            return "EAGER"
        return "TIE"


def bench_one(
    *,
    B: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    label: str,
    warmup_iters: int,
    measure_iters: int,
    n_repeats: int,
    seed: int = 0,
) -> ShapeBench:
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(B, K, V, device="cuda", dtype=dtype, generator=g)
    cum_log_probs = torch.randn(B, K, device="cuda", dtype=torch.float32, generator=g)
    finished_mask = torch.rand(B, K, device="cuda", generator=g) < 0.3

    # Triton autotune is keyed on V, so the first call at a given V triggers
    # config search + compilation. Burn a few extra calls before timing.
    for _ in range(10):
        beam_logprobs_prep_triton(logits, finished_mask, cum_log_probs)
    torch.cuda.synchronize()

    eager = time_path(
        eager_path,
        (logits, finished_mask, cum_log_probs),
        label="eager",
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        n_repeats=n_repeats,
    )
    triton = time_path(
        beam_logprobs_prep_triton,
        (logits, finished_mask, cum_log_probs),
        label="triton",
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        n_repeats=n_repeats,
    )
    return ShapeBench(
        B=B, K=K, V=V, dtype=dtype, label=label, eager=eager, triton=triton
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--measure-iters", type=int, default=200)
    p.add_argument("--n-repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _fmt_dtype(dt: torch.dtype) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dt]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available; this microbench requires a CUDA device.")
        return 1

    device_name = torch.cuda.get_device_name()
    print(f"GPU             : {device_name}")
    print(f"PyTorch         : {torch.__version__}")
    try:
        import triton

        print(f"Triton          : {triton.__version__}")
    except Exception:
        pass
    print(f"warmup_iters    : {args.warmup_iters}")
    print(f"measure_iters   : {args.measure_iters}  (per repeat, divided to per-call)")
    print(f"n_repeats       : {args.n_repeats}")
    print()

    shapes = [
        # (B,  K,   V,         dtype,           label)
        (1, 10, 32000, torch.bfloat16, "production (TinyLlama beam=10, bf16)"),
        (1, 10, 32000, torch.float16, "production (fp16)"),
        (1, 10, 32000, torch.float32, "production (fp32)"),
        (1, 10, 128256, torch.bfloat16, "Llama-3 vocab (bf16)"),
        (1, 4, 32000, torch.bfloat16, "smaller K=4 (bf16)"),
        (4, 10, 32000, torch.bfloat16, "batched B=4 (bf16)"),
    ]

    print("=" * 100)
    hdr = f"{'shape':<44s} {'eager med':>11s} {'triton med':>11s} {'speedup':>9s} {'verdict':>9s}"
    print(hdr)
    print("-" * 100)

    rows: list[ShapeBench] = []
    for B, K, V, dtype, label in shapes:
        full = f"{label}  [B={B},K={K},V={V},{_fmt_dtype(dtype)}]"
        sys.stdout.write(f"{full[:43]:<44s} ")
        sys.stdout.flush()
        try:
            r = bench_one(
                B=B, K=K, V=V, dtype=dtype, label=full,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                n_repeats=args.n_repeats,
                seed=args.seed,
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        rows.append(r)
        print(
            f"{r.eager.median_us:>9.2f}us "
            f"{r.triton.median_us:>9.2f}us "
            f"{r.speedup:>8.2f}x "
            f"{r.verdict:>9s}"
        )

    print()
    print("Per-shape detail (median / p10 / p90 / mean+-stdev, microseconds):")
    print("-" * 100)
    for r in rows:
        print(f"\n  {r.label}")
        for tr in (r.eager, r.triton):
            print(
                f"    {tr.label:<6s}  "
                f"med={tr.median_us:7.2f}  "
                f"p10={tr.percentile(0.1):7.2f}  "
                f"p90={tr.percentile(0.9):7.2f}  "
                f"mean={tr.mean_us:7.2f} +- {tr.stdev_us:5.2f}"
            )
        print(f"    speedup (eager / triton) = {r.speedup:.3f}x  -> {r.verdict}")

    if not rows:
        print("\nNo successful benchmarks.")
        return 1

    prod = rows[0]
    print()
    print("=" * 100)
    print("VERDICT")
    print("-" * 100)
    if prod.verdict == "TRITON":
        msg = (
            f"KEEP v6 only if the executor-loop sync floor is removed first. "
            f"Triton wins isolated by {prod.speedup:.2f}x on the production shape, "
            f"but the end-to-end multi-run regressed by ~+0.13ms E2E -- "
            f"that gap is the sync floor absorbing per-step launch savings, "
            f"plus per-call autotune-cache + empty_like overhead. Removing the "
            f"sync (skill: perf-torch-sync-free) is a prerequisite to keeping v6."
        )
    elif prod.verdict == "EAGER":
        msg = (
            f"DROP v6. Even isolated, eager beats Triton by "
            f"{1 / prod.speedup:.2f}x on the production shape. The end-to-end "
            f"regression is no surprise -- the kernel was never faster to "
            f"begin with. Recommend reverting "
            f"tensorrt_llm/_torch/pyexecutor/_beam_search_kernels.py and the "
            f"call site in sampling_utils.py."
        )
    else:
        msg = (
            f"DROP v6. Isolated speedup is essentially a tie ({prod.speedup:.2f}x), "
            f"so the end-to-end regression of +0.13ms E2E is the per-call "
            f"overhead (autotune-cache lookup + empty_like + Python dispatch) "
            f"exceeding the kernel-launch savings. No isolated win to defend "
            f"this kernel."
        )
    print(msg)
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
