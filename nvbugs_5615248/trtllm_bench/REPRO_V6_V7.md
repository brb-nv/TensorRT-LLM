# NVBug 5615248 — v6 + v7 reproduction guide

This document describes how to reproduce the **per-step beam-search host-overhead**
fixes (commits v6 and v7 below) **on top of the piecewise CUDA-graph capture
fix** (`fb0acdde05`). It lives alongside the broader workload context in
[`REPRO.md`](./REPRO.md) and the post-v5 divergence map in
[`POST_V5_DIVERGENCE.md`](./POST_V5_DIVERGENCE.md). NVBug:
<https://nvbugspro.nvidia.com/bug/5615248>.

## What's being measured

| | SHA | Title |
|---|---|---|
| baseline | `fb0acdde05` | `[https://nvbugs/5615248][fix] Broader piecewise CUDA-graph capture` |
| **v6** | `b37b69ed35` (this branch) / `519c745e3a` (origin) | `[NVBUG-5615248][perf] Defer per-step beam-history D2H to terminal step` |
| **v7** | `3f0497862e` (this branch) / `5dfc2211b6` (origin) | `[NVBUG-5615248][perf] Cast seq_slots/seq_lens to CUDA exactly once per step` |

**v6** defers the heavy D2H copies inside `_prepare_beam_history`
(`cache_indirection`, `original_tokens`, optional logprob slices) into the
returned `BeamHistoryBuilder` so they only fire on the terminal decode step
when `need_history` actually flips True.

**v7** casts `seq_slots` / `seq_lens` to CUDA exactly once per step inside
`_process_requests`, sharing the result with both the per-group beam-search
metadata builder and the `sample_async` finish-reasons path. Eliminates 2
redundant H2D launches/step on the typical concurrency=1 / single-strategy
beam-search path; multi-strategy batches keep their per-group fall-back.

Both are pure-Python edits to `tensorrt_llm/_torch/pyexecutor/sampler.py`, so
**no rebuild is required when toggling refs** — `trtllm-bench` re-imports
`sampler.py` on every launch.

## Workload (TinyLlama beam=10)

Same as `REPRO.md`:

- Model: `TinyLlama-1.1B-Chat-v1.0`
- Hardware: NVIDIA L40S
- Geometry: ISL=100, OSL=20, beam_width=10, concurrency=1, max_batch_size=1
- Dataset: `dataset_isl100_osl20.jsonl` (16 measurement requests, 3 warmup)
- Driver: `run_multirun_pytorch.sh` invokes `trtllm-bench` 5× per ref
  (5 runs × 16 requests = n=80 pooled per-request samples per ref)

## How to run

Inside the TRT-LLM container, from the repo root, on a node with one L40S
visible:

```bash
bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh
```

The driver:

1. Pre-flights `trtllm-bench`, `git`, `python3`, the bench harness, and asserts
   `BASELINE_REF..FEATURE_REF` only touches `sampler.py`.
2. Resolves `BASELINE_REF` (default `HEAD~2` = piecewise) and `FEATURE_REF`
   (default `HEAD` = piecewise + v6 + v7) to immutable SHAs **before** the
   first `git checkout`.
3. Refuses to run if the working tree has staged or unstaged changes on
   tracked files (untracked `nvbugs_5615248/` artifacts are fine).
4. Checks out the baseline SHA, runs `run_multirun_pytorch.sh` into
   `v6_v7_validation/baseline_piecewise/`, stamps a `.done` marker.
5. Same for the feature SHA into `v6_v7_validation/feature_piecewise_v6_v7/`.
6. Aggregates with `aggregate_runs.py` (Welch's t-test, both per-run-mean and
   pooled-per-request views).
7. Trap restores the original branch HEAD on any exit path (success, error,
   ctrl-C). You'll never end up detached at the baseline SHA.

Useful overrides (env vars):

| Var | Default | Use |
|---|---|---|
| `NUM_RUNS` | `5` | Lower for a smoke check; raise to tighten p-values |
| `OUT_ROOT` | `nvbugs_5615248/trtllm_bench/v6_v7_validation` | Send outputs to scratch |
| `SKIP_BASELINE=1` | unset | Skip baseline if already complete |
| `SKIP_FEATURE=1` | unset | Skip feature if already complete |
| `BASELINE_REF` / `FEATURE_REF` | `HEAD~2` / `HEAD` | Override SHAs |

Each phase takes ~3-4 minutes wall on an L40S (5 runs × ~35s + warmup);
total bench time ~8-12 minutes.

## Reference results (this branch, validated 2026-05-05)

`run_v6_v7_bench.sh` against `HEAD~2` vs `HEAD` on this branch yielded the
following on a single L40S (warm node, default driver order: baseline first,
feature second):

### Pooled per-request (n=80 vs n=80) — primary view

| metric | baseline (piecewise) | feature (piecewise+v6+v7) |    Δ (ms) |  Δ (%) |       t |        p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |               11.091 |                    10.950 |    -0.141 | -1.27% |   +1.99 |    0.048 |
| **E2E** |             84.088 |                    83.699 |  **-0.389** |  **-0.46%** |   +7.44 | **2e-11** |
| ITL |                 0.367 |                     0.366 |    -0.001 | -0.34% |   +4.41 | 1.9e-05 |

### Per-run means (n=5 vs n=5) — corroborating view

| metric | baseline | feature | Δ (ms) | Δ (%) |       t |     p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   11.091 |  10.950 | -0.141 | -1.27% | +0.98 |  0.37 |
| E2E |    84.088 |  83.699 | -0.389 | -0.46% | +2.57 | 0.054 |
| ITL |     0.367 |   0.366 | -0.001 | -0.34% | +3.44 | 0.011 |

### Decomposition

Per-step decode = (E2E − TTFT) / 19:

| | baseline | feature | Δ |
|---|---:|---:|---:|
| Per-step decode (ms) | 3.842 | 3.829 | **-0.013** (-0.34%) |

```
ΔE2E   =  ΔTTFT  +  19 × Δper_step
−0.389 = −0.141  +  −0.248
         (36 %)     (64 %)
```

So **~64 % of the E2E gain comes from the steady-state decode loop** (the
design target of v6 and v7); **~36 %** is the prefill iter's share of v7's
precast (the H2D pair runs once during prefill too).

## Caveats

1. **Warm-cache ordering bias.** The default driver runs all 5 baseline
   runs first, then all 5 feature runs. The feature phase therefore sees a
   slightly warmer GPU/CPU/Python state. On an L40S at concurrency=1 this
   is typically tens of µs in favor of the feature, well below the run-to-run
   σ ≈ 0.5 ms; the E2E delta still clears p < 1e-6 pooled even after a
   conservative subtraction. If you want strict ordering neutrality, run
   `NUM_RUNS=1` twice with `SKIP_BASELINE=1` / `SKIP_FEATURE=1` in alternating
   order.
2. **Beam-search non-determinism.** At temperature=0, beam search can still
   produce different texts run-to-run because of FP tie-breaking inside
   `multi-block top-k` and timing-dependent sampler ordering. This branch's
   v6/v7 reproduce baseline's same non-determinism envelope (10-14 / 16
   request_ids stable across self-runs; 12 / 16 stable baseline ↔ feature
   cross). No new regression introduced; see the spot-check script in this
   repo's chat history.
3. **ITL is per-beam-token at beam=10.** The reported ITL number is deflated
   by ~10× vs the effective per-step decode latency (see *Reporting caveat*
   in `REPRO.md`). The "Per-step decode" row above is the right number for
   visualizing decode-loop effort per output token.
4. **Multi-strategy batches retain a fall-back path.** v7's optimisation only
   eliminates the redundant H2D pair when the strategy group covers the full
   batch (the typical concurrency=1 / single-strategy beam-search case).
   When strategies are mixed within a batch, the per-group host-gather + H2D
   still runs unchanged.

## What we are NOT measuring here

- Multi-strategy batching, multi-request concurrency, or non-beam paths —
  see [`POST_V5_DIVERGENCE.md`](./POST_V5_DIVERGENCE.md) and `REPRO.md` for
  scope.
- Long OSL (≥ 64). Both fixes scale linearly with OSL; expected gain grows
  but the variance does too. The OSL=20 setting here is the same one used
  for v3..v5 measurements and keeps the validation comparable to the prior
  series.
- TensorRT backend behaviour. `run_multirun_trt.sh` exists in this directory
  but is not part of the v6+v7 validation; v6 and v7 are PyTorch-only edits.

## Layout reference

```
nvbugs_5615248/trtllm_bench/
├── REPRO.md                       # Workload + flag-rationale (from v3..v5)
├── REPRO_V6_V7.md                 # ← this file
├── POST_V5_DIVERGENCE.md          # Post-v5 PyT-vs-TRT divergence map
├── aggregate_runs.py              # Pooled-per-request Welch's t-test (stdlib)
├── analyze_decode_loop.py         # nsys SQLite Phase-1/2 host-bound detector
├── build_trt_engine.py            # Build TinyLlama TRT engine for cross-check
├── dataset_isl100_osl20.jsonl     # 16-request synthetic dataset
├── pytorch.yaml                   # PyT extra_llm_api_options
├── trt.yaml                       # TRT runtime flags
├── run_multirun_pytorch.sh        # 5× trtllm-bench launcher (PyT)
├── run_multirun_trt.sh            # 5× trtllm-bench launcher (TRT)
├── run_nsys_trace.sh              # Single nsys trace launcher (either backend)
├── run_v6_v7_bench.sh             # ← Two-ref bench driver (this guide's entry point)
└── .gitignore                     # Keeps regenerated output trees out of git
```
