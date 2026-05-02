# Fix — `TorchSampler` prefill→decode-handoff launch storm

This document captures two related bugs in the PyTorch sampler's
prefill→decode handoff path, the diagnoses, and the measured impact of the
two-stage fix in `tensorrt_llm/_torch/pyexecutor/sampler.py`. Both fixes
follow the same playbook (replace indexed-assign + RHS-staging with a single
fused indexing primitive); the second fix additionally hoists the now-shared
`int64` index cast out of the inner callee so it's paid once per handoff.

The companion files in this directory (`REPRO.md`, multi-run baselines under
the directory root, `optimized_v3/`, `optimized_v4/`) reproduce the numbers
below.

## TL;DR

| Metric | pre-fix | v3 (`index_fill_`) | v4 (+`index_copy_` + cast hoist) | Cumulative Δ |
|---|---:|---:|---:|---:|
| `prepare_beam_search` body (avg/call, NVTX) | **503 μs** | 199 μs | **160 μs** | **−68%** |
| `setup.finish_reasons.update` (avg/call, NVTX) | ~131 μs | ~131 μs | **78 μs** | **−40%** |
| `setup_sampler_step` aggregate per handoff (NVTX) | ~700 μs | ~474 μs | **~330 μs** | **−53%** |
| GPU kernels per `prepare_beam_search` | **21** | 8 | 8 | −62% |
| Per-request **TTFT** mean (n=80 samples) | 11.280 ms | 11.013 ms | **10.896 ms** | **−0.384 ms (−3.41%), t = +5.45, p = 2×10⁻⁷** |
| Per-request **E2E** mean (n=80 samples) | 84.244 ms | 83.951 ms | **83.831 ms** | **−0.413 ms (−0.49%), t = +8.27, p = 5×10⁻¹⁴** |
| Per-request **ITL** mean (n=80 samples) | 0.367 ms | 0.367 ms | **0.367 ms** | unchanged, t = +0.56, p = 0.58 |

ITL not moving (t = +0.56 cumulatively across both fixes) is the cleanest
mechanistic signature of the change set: it touches only the one-shot
prefill→decode handoff, never the per-token decode loop. **~7.0% of the
PyT-vs-TRT TTFT gap closed, ~10.9% of the E2E gap.**

## The bugs

`TorchSampler.setup_sampler_step` is the prefill→decode handoff: when one or
more requests finish their last context chunk and transition into beam-search
generation, it (a) updates per-request state in the `_finish_reasons_handler`
store and (b) resets seven beam-search buffers via `_prepare_beam_search`.
Both of those sub-paths originally exhibited the same anti-pattern:
**indexed-assignment with a CUDA-resident RHS source tensor**, which PyTorch
lowers to multiple serialized GPU kernels, each paying ~15 μs of
`cudaLaunchKernel` overhead.

### Bug 1 — `_prepare_beam_search` per-buffer indexed-assign storm (v3)

The pre-v3 implementation reset each of the seven beam-search buffers with an
explicitly allocated CUDA-side RHS:

```python
cache_indirection[seq_slots, :, :max_prompt_len] = torch.zeros(
    (1,), dtype=cache_indirection.dtype, device=cache_indirection.device,
)
# ... and similarly for cum_log_probs, sampled_log_probs,
#     sampled_log_prob_ranks, predecessor_beams, original_tokens ...

# first_finish_reasons was even worse — pinned host alloc + H2D + unsqueeze:
first_finish_reasons[seq_slots] = (
    torch.tensor(
        FinishReason.NOT_FINISHED.value,
        pin_memory=prefer_pinned(),
        dtype=first_finish_reasons.dtype,
    )
    .to(first_finish_reasons.device, non_blocking=True)
    .unsqueeze(0)
)
```

`nvtx_kern_sum` from a one-request nsys trace showed each
`tensor[idx] = torch.zeros((1,), …)` lowering to **three** GPU kernels:

1. `vectorized_elementwise_kernel<FillFunctor>` — fill the 1-element RHS.
2. `unrolled_elementwise_kernel<direct_copy_kernel_cuda>` — stage it for the
   indexed assignment.
3. `index_elementwise_kernel<gpu_index_kernel>` — perform the scatter.

Seven buffers × three kernels = **21 GPU kernels** issued serially per
request handoff. Each does ~1 μs of actual work and pays ~15 μs of
`cudaLaunchKernel`. Total: ~503 μs of CPU on the critical path, ~92% of
`_prepare_beam_search`. Pure launch-overhead-bound.

### Bug 2 — `update_for_new_request` per-scatter staging pattern (v4)

`FinishReasonsHandler.update_for_new_request` did the same kind of
indexed-assign for the per-request `max_lengths` and `end_ids` device buffers:

```python
store.max_lengths_cuda[seq_slots_cuda] = max_lengths_cuda
store.end_ids_cuda[seq_slots_cuda] = end_ids_cuda
```

Each `tensor[idx] = source_tensor` lowers to **two** GPU kernels:

1. `direct_copy_kernel_cuda` — stage RHS for indexing.
2. `gpu_index_kernel` — perform the scatter.

Plus the per-call `seq_slots.long()` cast that lived inside
`_prepare_beam_search` after v3, which paid an additional `direct_copy_kernel`
(~25 μs) — once per handoff, never shared with `update_for_new_request`. Net:
two scatters × 2 kernels + one redundant cast ≈ 131 μs of CPU per handoff
in `setup.finish_reasons.update` alone.

## The fixes

### v3 — `Tensor.index_fill_` for the seven beam-search buffer resets

`Tensor.index_fill_(dim, index, value)` is the canonical PyTorch primitive
for "select rows along `dim`, fill with scalar `value`." It bypasses both
the RHS allocation and the broadcast-staging step, lowering directly to a
single fused `index_fill_kernel_impl` kernel. Constraint: it requires
`int64` indices.

```python
beam_search_store.cache_indirection.narrow(
    2, 0, max_prompt_len
).index_fill_(0, seq_slots_long, 0)
beam_search_store.cum_log_probs.index_fill_(0, seq_slots_long, 0)
# ... and so on for the other five buffers, with first_finish_reasons
# using FinishReason.NOT_FINISHED.value (= 0; see types.h) ...
```

`narrow(2, 0, max_prompt_len)` gives a non-owning view of `cache_indirection`
restricted to the prompt prefix along the sequence-length axis;
`index_fill_` on that view writes through to the underlying storage and
preserves the original "zero only `[..., :max_prompt_len]`" semantics.

`FinishReason.NOT_FINISHED == 0` per `cpp/include/tensorrt_llm/executor/types.h`,
so the `first_finish_reasons` reset folds cleanly into the same scalar-zero
pattern as the other six buffers (no pinned-host buffer, no H2D copy, no
`unsqueeze`).

### v4 — `Tensor.index_copy_` for the two finish-reasons scatters + cast hoist

`Tensor.index_copy_(dim, index, source)` is the canonical primitive for
"select rows along `dim`, overwrite with rows from `source`." Same shape as
`index_fill_` but with a tensor RHS instead of a scalar. Same `int64` index
constraint.

```python
# In FinishReasonsHandler.update_for_new_request:
store.max_lengths_cuda.index_copy_(0, seq_slots_cuda_long, max_lengths_cuda)
store.end_ids_cuda.index_copy_(0, seq_slots_cuda_long, end_ids_cuda)
```

That collapses each scatter from 2 kernels to 1 fused `index_copy_kernel_impl`.
Two scatters × (2 → 1) = 2 launches saved.

The `int64` cast was already needed once per handoff for v3's `index_fill_`
calls; v4 hoists it to `setup_sampler_step` itself so the *same* int64 view
feeds both `update_for_new_request` (`index_copy_`) and `_prepare_beam_search`
(`index_fill_`):

```python
# In setup_sampler_step, right after the H2D copy:
with nvtx_range("setup.seq_slots_to_long"):
    seq_slots_tensor_cuda_long = seq_slots_tensor_cuda.long()

with nvtx_range("setup.finish_reasons.update"):
    self._finish_reasons_handler.update_for_new_request(
        seq_slots_cuda_long=seq_slots_tensor_cuda_long,
        ...,
    )

if self._use_beam_search:
    with nvtx_range("setup.prepare_beam_search"):
        self._prepare_beam_search(
            beam_search_store, log_probs_store,
            seq_slots_long=seq_slots_tensor_cuda_long,
            max_prompt_len=max_prompt_len,
        )
```

Net: the redundant 25 μs cast inside `_prepare_beam_search` is amortized
across both consumers, and partially overlaps with `setup.h2d` instead of
being serialized inside the prepare-beam-search range.

## NVTX instrumentation (added alongside the fixes)

Both fixes added nested NVTX ranges so future profiling can attribute time
without re-instrumenting:

- `setup_sampler_step` is decomposed into `setup.collect_loop`,
  `setup.h2d`, **`setup.seq_slots_to_long`** (added in v4),
  `setup.finish_reasons.update`, and `setup.prepare_beam_search`.
- `_prepare_beam_search` carries a top-level `prepare_beam_search` decorator
  plus seven `pbs.<buffer>_zero` sub-ranges. The `pbs.seq_slots_to_long`
  range that v3 introduced was promoted to `setup.seq_slots_to_long` in v4.

These are zero-cost when nsys isn't tracing (`nvtx.annotate` is a no-op
outside a profiling session).

## Diagnosis path (for future readers)

1. Multi-run aggregation (5× PyT, 6× TRT) at beam=10 showed PyT excess of
   ~3 ms E2E / ~5 ms TTFT vs TRT (see `REPRO.md`).
2. NVTX-instrumented one-request nsys trace
   (`nvbugs_5615248/trtllm_bench/nsys_handoff/`) attributed the largest
   *one-shot* chunk to `setup_sampler_step` (~700 μs/handoff), of which
   `_prepare_beam_search` was ~503 μs (71%) and
   `setup.finish_reasons.update` was ~131 μs (19%).
3. `nvtx_kern_sum` per `pbs.*_zero` revealed the 3-kernels-per-buffer
   launch-storm in v1 (Bug 1).
4. The `index_fill_` rewrite (v3) was validated end-to-end with a 5× PyT
   multi-run (see `optimized_v3/`) and Welch's t-test.
5. After v3 landed, the next-largest `setup_sampler_step` consumer was
   `setup.finish_reasons.update`. `nvtx_kern_sum` confirmed the same
   2-kernels-per-scatter pattern (Bug 2).
6. The `index_copy_` rewrite + cast hoist (v4) was validated with another
   5× PyT multi-run (see `optimized_v4/`).

## Measurement protocol

- Hardware: NVIDIA L40S (single GPU, single-stream).
- Workload: TinyLlama-1.1B-Chat-v1.0, beam=10, ISL=100, OSL=20,
  `max_seq_len=129`, `max_batch_size=1`, `--concurrency 1`, `--streaming`,
  16 requests + 3 warmup per run.
- Configs: `pytorch.yaml` (PyT side, with `enable_piecewise_cuda_graph: true`),
  `trt.yaml` + `nvbugs_5615248/tinyllama_trt_engine` (TRT side).
- Sampling: 5 independent `trtllm-bench` invocations per condition, launched
  via `run_multirun_pytorch.sh`. Each invocation cold-starts a fresh
  `PyExecutor` so the runs are statistically independent.
- NVTX/kernel attribution from a separate one-request nsys-profiled run
  (`--trace=nvtx,cuda,osrt --sample=none --cpuctxsw=none --backtrace=none`).
  See `nsys_handoff/` (v1), `nsys_handoff_v3/` (v3), `nsys_handoff_v4/` (v4).
- Statistics: per-run means (n=5 vs n=5) and pooled per-request samples
  (n=80 vs n=80), aggregated by `aggregate_runs.py`. Two-sided Welch's t-test
  on both. Pooled per-request is the definitive test (~16× more statistical
  power than n=5).
- Sign convention in tables: Δ (ms) = experiment − baseline (negative =
  improvement); t = raw Welch's t of (baseline, experiment) (positive =
  improvement).

## Per-call NVTX deltas (one-request nsys trace per condition)

| Range | v1 (n=7) | v3 (n=7) | v4 (n=9) | Cumulative Δ |
|---|---:|---:|---:|---:|
| `setup.prepare_beam_search` | 502.9 μs | 198.6 μs | **163.4 μs** | −68% |
| `prepare_beam_search` (body) | 500.0 μs | 195.6 μs | **159.9 μs** | −68% |
| `setup.finish_reasons.update` | ~131 μs | ~131 μs | **77.8 μs** | −40% |
| `setup.seq_slots_to_long` | — | (25.5 μs as `pbs.seq_slots_to_long`) | **25.6 μs** (hoisted) | (relocated, 1 cast/handoff) |
| `setup.h2d` | ~65 μs | ~65 μs | 62.7 μs | flat |
| `setup.collect_loop` | ~10 μs | ~10 μs | 10.0 μs | flat |
| `pbs.cache_indirection_zero` | 80.5 μs | 49.5 μs | 45.1 μs | −44% |
| `pbs.cum_log_probs_zero` | 62.0 μs | 17.2 μs | 16.8 μs | −73% |
| `pbs.sampled_log_probs_zero` | 60.6 μs | 13.7 μs | 14.4 μs | −76% |
| `pbs.sampled_log_prob_ranks_zero` | 56.3 μs | 14.2 μs | 13.2 μs | −76% |
| `pbs.predecessor_beams_zero` | 55.9 μs | 12.7 μs | 13.0 μs | −77% |
| `pbs.first_finish_reasons_zero` | 82.4 μs | 14.6 μs | 15.1 μs | −82% |
| `pbs.original_tokens_zero` | 61.2 μs | 12.6 μs | 12.9 μs | −79% |

`pbs.cache_indirection_zero` retains the largest residual cost in the
beam-search buffer-reset group because of the `narrow(2, 0, max_prompt_len)`
view path; the other six are at the ~13–17 μs floor, dominated by
`cudaLaunchKernel` for a single fused fill.

`setup.finish_reasons.update` (v4) has notably high variance in a single
trace (min 56.7 μs, max 201.8 μs, mean 77.8 μs) — the max is the first
heavy invocation eating cold-cache and first-pinned-alloc cost. Multi-run
amortization brings the per-request impact to a stable signal.

## Per-request multi-run validation (n=80 vs n=80, pooled)

### v3 vs pre-fix (the v3 fix in isolation)

| Metric | pre-fix mean | v3 mean | Δ (ms) | Δ (%) | t | p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 11.280 | 11.013 | −0.267 | −2.37% | +3.56 | 0.000487 |
| E2E | 84.244 | 83.951 | −0.293 | −0.35% | +5.07 | 1.18e-06 |
| ITL | 0.367 | 0.367 | −0.000 | −0.03% | +0.49 | 0.624 |

### v4 vs v3 (incremental gain from the v4 fix only)

| Metric | v3 mean | v4 mean | Δ (ms) | Δ (%) | t | p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 11.013 | 10.896 | −0.117 | −1.06% | +1.70 | 0.0913 |
| E2E | 83.951 | 83.831 | −0.121 | −0.14% | +2.08 | 0.0393 |
| ITL | 0.367 | 0.367 | −0.000 | −0.01% | +0.07 | 0.942 |

### v4 vs pre-fix (cumulative gain from both fixes)

| Metric | pre-fix mean | v4 mean | Δ (ms) | Δ (%) | t | p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT | 11.280 | 10.896 | **−0.384** | **−3.41%** | **+5.45** | **2.01e-07** |
| E2E | 84.244 | 83.831 | **−0.413** | **−0.49%** | **+8.27** | **5.19e-14** |
| ITL | 0.367 | 0.367 | −0.000 | −0.04% | +0.56 | 0.579 |

The cumulative TTFT/E2E signals are overwhelmingly significant. The 117 μs
TTFT and 121 μs E2E observed in v4-vs-v3 sit between the two natural NVTX
predictions: 65 μs of direct savings in `setup.finish_reasons.update`, and
138 μs if you also credit the cast hoist (which now overlaps with `setup.h2d`
rather than serializing inside `prepare_beam_search`). Numbers reconcile.

## What these fixes do *not* close

The remaining ~3.4 ms / ~5.0 ms PyT-vs-TRT gap on E2E / TTFT is dominated
by **per-decode-step Python overhead**: 19 generation iterations per request
× ~150 μs of Python tax each ≈ 2.85 ms. That residual lives in the
`_sample_async` / `_process_requests` / `_prepare_inputs` chain and would
require either extending the piecewise CUDA-graph capture across the
sampling step or porting the per-iteration coordination to a tighter C++
path. Out of scope for these fixes.

Within `setup_sampler_step` itself, the largest remaining levers are:

| Range | v4 cost | Floor target | Effort |
|---|---:|---:|---|
| `setup.h2d` | 62.7 μs | ~15 μs | Medium-hard. Pre-allocate a pinned scratch buffer and copy slices into it instead of paying `cudaHostAlloc` per call. |
| `pbs.cache_indirection_zero` | 45.1 μs | ~15 μs | Hard. Would need to replace `narrow + index_fill_` with a strided-fill custom op. |
| `setup.seq_slots_to_long` | 25.6 μs | hard floor | One unavoidable `direct_copy_kernel` per handoff. |

Neither is obviously worth the complexity vs. the per-decode-step lever.

## Files

- `tensorrt_llm/_torch/pyexecutor/sampler.py` — both fixes (v3: `index_fill_`
  in `_prepare_beam_search`; v4: `index_copy_` in `update_for_new_request`
  + `int64` cast hoisted to `setup_sampler_step`).
- `nvbugs_5615248/trtllm_bench/run_multirun_pytorch.sh` — multi-run launcher
  (5 independent `trtllm-bench` invocations into a CLI-supplied output dir).
- `nvbugs_5615248/trtllm_bench/aggregate_runs.py` — per-run-mean and pooled
  per-request aggregation with two-sided Welch's t-test (pure stdlib).
- `nvbugs_5615248/trtllm_bench/nsys_handoff/` — v1 one-request nsys trace.
- `nvbugs_5615248/trtllm_bench/nsys_handoff_v3/` — v3 one-request nsys trace.
- `nvbugs_5615248/trtllm_bench/nsys_handoff_v4/` — v4 one-request nsys trace.
- `nvbugs_5615248/trtllm_bench/request_pytorch{,2..5}.json` — pre-fix
  multi-run baseline.
- `nvbugs_5615248/trtllm_bench/optimized_v3/request_pytorch{,2..5}.json` —
  post-v3 multi-run.
- `nvbugs_5615248/trtllm_bench/optimized_v4/request_pytorch{,2..5}.json` —
  post-v4 multi-run.
- `nvbugs_5615248/trtllm_bench/REPRO.md` — multi-run protocol.
