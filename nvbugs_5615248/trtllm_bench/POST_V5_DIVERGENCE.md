# NVBug 5615248 — PyT vs TRT divergence map after v3 + v4 + v5

This document captures the post-v5 PyT-vs-TRT comparison and identifies
remaining divergences for the TinyLlama beam=10 workload behind NVBug
5615248. It pairs with `REPRO.md` (which defines the bench protocol) and
records the residual TTFT / decode-loop gap once the four upstream fixes
on this branch are in place:

- `fb0acdde0` — broader piecewise-CUDA-graph capture
- `2f46e362c` — v3: `_prepare_beam_search` `index_fill_` rewrite
- `efa15c506` — v4: finish-reasons handler `index_copy_` rewrite
- `e75cbaa70` — v5: cached `BeamSearchStore.seq_offsets` / `beam_idx_arange`

Hardware: NVIDIA L40S, TinyLlama-1.1B-Chat-v1.0, ISL=100 / OSL=20,
beam=10, concurrency=1, 16 measurement requests per run, 3 warmup
requests per run.

## TL;DR

**The whole remaining E2E gap lives in TTFT (prefill→first-decode handoff).
The decode loop itself is already a structural PyT win.** PyTorch's host
loop is launch-bound (`M2 = 0.135` cudaLaunchKernel ratio in the
steady-state window vs `0.049` for TRT); TRT's apparent host idle is
80 %  `cudaEventSynchronize` (the host sleeping on the GPU under
concurrency=1), not host bookkeeping.

The single largest remaining waste in the PyT decode loop is
`_prepare_beam_history` firing every step and unconditionally enqueuing
2–5 D2H copies of growing `cache_indirection` / `original_tokens` slices
that get discarded on every step except the very last one.

## Headline numbers (5+5 fresh runs, n=80 per backend)

| Metric                | PyT v3+v4+v5     | TRT              | PyT − TRT  | PyT − TRT % | Winner |
|-----------------------|-----------------:|-----------------:|-----------:|------------:|:------:|
| **TTFT** (ms)         | 10.822 ± 0.087   |  5.894 ± 0.090   | **+4.93**  | +83.6 %     | TRT    |
| **E2E** (ms)          | 83.781 ± 0.110   | 80.446 ± 0.108   | **+3.34**  |  +4.1 %     | TRT    |
| **Per-step decode**   |  3.840 ± 0.0015  |  3.924 ± 0.006   |   −0.08    |  −2.1 %     | **PyT**|
| ITL (per-beam-token)  |  0.367           |  0.375           |   −0.008   | —           | PyT    |

`±` is run-to-run stddev across 5 PyT and 5 TRT runs. ITL is reported
as-is from `streaming_metrics`; it is per-beam-token and deflated by
~10× at beam=10 (see *Reporting caveat* in `REPRO.md`). "Per-step
decode" is `(E2E − TTFT) / 19`.

E2E gap decomposes as:

```
ΔE2E = ΔTTFT + 19 × Δper-step
3.34 ≈ 4.93   − 1.60      (PyT loses TTFT, wins per-step)
```

So **the entire ΔE2E lives in TTFT**.

## Effect of v3+v4+v5 on PyT (vs piecewise-only baseline)

Pooled per-run means, baseline_piecewise_run1+run2 (n=10) vs the fresh
optimized_v5_pyt_fresh (n=5):

| Metric           | piecewise-only | v3+v4+v5 |   Δ            |
|------------------|---------------:|---------:|---------------:|
| TTFT (ms)        |        11.182  |  10.822  | −0.36 (−3.2 %) |
| E2E (ms)         |        84.253  |  83.781  | −0.47 (−0.56 %)|
| per-step (ms)    |         3.846  |   3.840  | −0.006 (−0.15 %)|

The v3+v4 (handoff) and v5 (per-step arange cache) wins are real and
stable across the new run, matching what each commit advertised in its
description.

## Steady-state window divergence (nsys, ~1.5–1.9 s)

Window picked from `analyze_decode_loop.py` semantics: skip the first 5
prefill iterations (warmup), end at the start of the last prefill so the
final request's tail is inside the window.

| Window metric                | PyT                    | TRT                       | Interpretation                                                |
|------------------------------|-----------------------:|--------------------------:|---------------------------------------------------------------|
| Window length                | 1939 ms (21 reqs)      | 1478 ms (17 reqs)         | normalised to ≈ 88 ms / req                                   |
| Kernels in window            |             **37 299** |                **12 289** | PyT launches ~3× more per request                             |
| GPU active                   | 82.8 ms (4.3 %)        |     151.2 ms (10.2 %)     | TRT does more on-GPU work / req (`insertUnfinishedPathKernel`)|
| `cudaLaunchKernel` total     |     **209 ms (10.8 %)**| 50 ms (3.4 %)             | PyT's launch tax is real                                      |
| `cudaGraphLaunch` total      | 52 ms / 859 replays    | 22 ms / 342 replays       | both backends graph-replay decode                             |
| `cudaEventSynchronize`       | 15.3 ms (0.79 %)       |     **1182 ms (80 %)**    | TRT is genuinely *idle-waiting*, PyT is *launching*           |
| `cudaMemcpyAsync`            | 75.7 ms (3.9 %, n=9 633)| 38.1 ms (2.6 %, n=9 486)  | PyT moves 2× the bytes                                        |
| **M1** GPU-idle ratio        | **0.957**              | 0.898                     | both nominally "host-bound"                                   |
| **M2** launch ratio          | **0.135**              | 0.049                     | only PyT crosses the 0.10 launch-bound threshold              |
| **M4** GPU util              | 0.043                  | 0.102                     | TRT keeps the GPU busier per request                          |

Important nuance: TRT's 80 %  `cudaEventSynchronize` ratio means the
host is *sleeping* on event-wait — the correct concurrency=1 behavior
(per-step kernels are fast enough that there is nothing else to
schedule). PyT's "idle" is fragmented host bookkeeping between
micro-launches. Same M1, very different shape.

## Per-prefill iter cost (drives TTFT)

|                        | PyT `_forward_step` (ctx) | TRT `enqueueV3` |
|------------------------|--------------------------:|----------------:|
| mean                   |                **5.91 ms**|         2.31 ms |
| median                 |                   5.81 ms |         2.32 ms |
| n in window            |                        22 |              17 |

PyT prefill costs **3.6 ms more than TRT per request** at the engine-step
level. Add ~1.3 ms of per-request lifecycle outside `_forward_step` and
you recover the 4.93 ms TTFT gap exactly.

## Per-iter NVTX budget on PyT (steady-state, 420 forward-steps)

| NVTX range                     | per-iter (µs)   | n   |
|--------------------------------|----------------:|----:|
| `_sample_async`                |       **2 691** | 420 |
| `_process_requests`            |           1 859 | 420 |
| `sample_batched_by_strategy`   |           1 491 | 420 |
| `_prepare_inputs`              |             581 | 419 |
| `_write_finish_reasons`        |             299 | 420 |
| `maybe_create_beam_histories`  |             228 | 420 |
| `prepare_resources`            |             176 | 420 |
| `_fetch_new_requests`          |             142 | 441 |
| `_update_requests`             |             117 | 420 |

## Decode-step kernel mix (PyT vs TRT)

PyT top kernels by total time (in window):

| kernel                               |     n  | total (ms) | mean (µs) |
|--------------------------------------|-------:|-----------:|----------:|
| `index_elementwise_kernel` (variant 1) | 7 098  |    15.260 |      2.15 |
| `mbtopk::computeDigitCumSum`          | 1 680  |     8.617 |      5.13 |
| `index_elementwise_kernel` (variant 2) | 3 360  |     6.977 |      2.08 |
| GEMV (gemvx::kernel)                  |    21  |     3.917 |    186.53 |
| `mbtopk::computeBlockwiseWithinKCounts`| 1 680 |     3.792 |      2.26 |
| softmax                               |   420  |     3.588 |      8.54 |
| `mbtopk::computeBlockDigitCounts`     | 1 680  |     3.480 |      2.07 |

TRT top kernels:

| kernel                                   |     n  | total (ms) | mean (µs) |
|------------------------------------------|-------:|-----------:|----------:|
| `insertUnfinishedPathKernel`             |   360  |    45.139 |    125.39 |
| GEMM tilesize 64×96×64                   |   383  |    27.599 |     72.06 |
| GEMM tilesize 32×32×64 (nn)              |   766  |    19.285 |     25.18 |
| GEMM tilesize 32×32×64 (tn)              |   383  |     8.577 |     22.40 |
| `addBiasSoftMax`                         |   360  |     8.566 |     23.79 |
| `air_topk_stable::radix_k`               | 1 080  |     5.227 |      4.84 |
| `batchApplyPenalty`                      |   360  |     4.611 |     12.81 |

Read-out:

- The **10 458** `index_elementwise_kernel` calls are the fingerprint of
  `index_fill_` / `index_copy_` / scatter ops in the per-step beam-search
  book-keeping (`beam_search_sampling_batch`,
  `cache_indirection_swap`, `finished_beams_update`,
  `_handle_first_finish_reasons`, and the per-step
  `_prepare_beam_history` D2H copies). v3+v4+v5 cut the worst offenders
  but the long tail remains.
- TRT's kernel mix is dominated by a single
  **`insertUnfinishedPathKernel`** at 125 µs / call ~3.0 ms / request —
  the TRT-side beam-search book-keeping that has no PyT analogue.

## Primary candidates for v6+

Rank-ordered by **(impact × confidence) / effort**.

### 1. **v6** — Defer per-step D2H copies in `_prepare_beam_history`

`@nvtx_range("maybe_create_beam_histories")` fires every step (228 µs /
iter × 20 iter / req ≈ **4.6 ms / request** of host budget), but its
output (`BeamHistoryBuilder`) is discarded on every step except the
very last one when `need_history` flips True (driven by stop-words or
`should_stop`).

Each call enqueues **2–5 async D2H copies** of growing slices —
`cache_indirection[..., prompt_length:num_tokens]`,
`current_path`, optionally `sampled_log_probs`,
`sampled_logprobs_indices`, `cum_log_probs` — *unconditionally*. These
copies are wasted work on every intermediate step.

Approach: keep the cheap async `_copy_to_host(need_history)` (it's the
only thing `update_requests` needs to decide whether to materialize),
but defer the heavy slice copies until `need_history` is *known* to be
True on the host. The simplest deterministic shortcut: if the request
is one decode step away from its `max_tokens`, then we know the next
`update_requests` will need a history — kick off the heavy copies on
*that* step only.

Estimated gain: removes ≈ 4 ms / req of host budget. On a launch-bound
loop (PyT M2 = 0.135) this projects to roughly:

- ~0.2 ms / step shaved → ~3–4 ms E2E reduction at OSL=20.
- Larger gains at higher OSL (savings scale with the number of
  intermediate steps where the copies are wasted).
- TTFT untouched.

### 2. Hoist sampler-thread `seq_slots`/`seq_lens` casts (low effort)

Inside `_compute_grouped_metadata` (sampler.py around line 3216), each
beam-search iteration re-runs:

```python
seq_slots[indices].to(device="cuda", dtype=torch.int64, non_blocking=True)
seq_lens[indices].to(device="cuda", non_blocking=True)
```

even though v4 already produced an int64 `seq_slots_long` upstream in
`setup_sampler_step`. Hoist the casts into `setup_sampler_step` and
thread the result through `BeamSearchMetadata`. Removes 2 H2D launches
per group per step (~0.05 ms / req).

### 3. CUDA-graph capture of the sampler step (item 3 in chat summary)

Higher effort, but converts the 25 launches/iter of
`index_elementwise_kernel` etc. into a single `cudaGraphLaunch`. Drops
PyT M2 launch ratio from 0.135 toward TRT's 0.049 (~1.5 ms / req of host
budget recovered). Risk: beam-search sampler has data-dependent control
flow that needs to be made explicit at capture boundaries.

### 4. Fuse residual `index_fill_`/`index_copy_` scatter ops in the sampler

v3/v4 fused the worst seven; remaining 10 000+ tiny scatter launches in
the steady-state window are mostly inside `beam_search_sampling_batch`
and `_handle_first_finish_reasons`. A small custom op or
`torch.compile` boundary around `beam_search_sampling_batch` would
collapse them.

### 5. Re-evaluate critical-path placement of `_write_finish_reasons` D2H

`_write_finish_reasons` already uses v4's `index_copy_` fusion but its
final D2H copy is enqueued every step (299 µs / iter). If the consumer
(`_check_beam_search_stop_criteria`) doesn't strictly need the result on
the same step, deferring by one step would let the GPU pipeline forward.

## Suggested order of execution

If we want to land another perf-PR series on top of v3+v4+v5:

- **v6** = item 1 (`_prepare_beam_history` deferral). Pure refactor, big
  win at higher OSL, no graph-capture risk.
- **v7** = item 2 (sampler-thread cast hoist). Tiny patch, follows from
  v4's `seq_slots_long` contract.
- **v8** = item 3 scoped design (sampler graph capture). Needs an own
  design discussion before code.

## Reproducing this analysis

```bash
# (per the helper scripts already on this branch)
bash nvbugs_5615248/trtllm_bench/run_multirun_pytorch.sh \
     nvbugs_5615248/trtllm_bench/optimized_v5_pyt_fresh
bash nvbugs_5615248/trtllm_bench/run_multirun_trt.sh \
     nvbugs_5615248/trtllm_bench/optimized_v5_trt_fresh
bash nvbugs_5615248/trtllm_bench/run_nsys_trace.sh pytorch \
     nvbugs_5615248/trtllm_bench/nsys_optimized_v5_pyt
bash nvbugs_5615248/trtllm_bench/run_nsys_trace.sh tensorrt \
     nvbugs_5615248/trtllm_bench/nsys_optimized_v5_trt

python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \
    --backend pytorch --experiment optimized_v5_pyt_fresh \
    --experiment-label v5-pyt-fresh
python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \
    --backend trt --experiment optimized_v5_trt_fresh \
    --experiment-label v5-trt-fresh

python3 nvbugs_5615248/trtllm_bench/analyze_decode_loop.py \
    --sqlite nsys_optimized_v5_pyt/pyt_beam10.sqlite --skip-prefills 5
# (TRT trace lacks the [Executor] _forward_step NVTX markers; window
# pick uses the ExecutionContext::enqueueV3 events instead — see the
# inline analysis in this file's chat history.)
```
