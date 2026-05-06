# NVBug 5615248 — Post-v7 plan

This document is the current source of truth for the **next** round of
PyTorch-vs-TensorRT optimisation work on the TinyLlama beam=10 workload behind
NVBug 5615248. It supersedes the priority list in
[`POST_V5_DIVERGENCE.md`](./POST_V5_DIVERGENCE.md), which was written before
v6 and v7 landed and is now historical.

Workload context: see [`REPRO.md`](./REPRO.md). Bench reproduction protocol
for v6+v7 (the most recent landed work): see [`REPRO_V6_V7.md`] on the
companion validation branch (`user/brb/nvbug-5615248-beam-host-overhead`).

## Where we stand (post-v7, measured)

5 runs × 16 requests, ISL=100 / OSL=20 / beam=10 / concurrency=1, L40S, TinyLlama-1.1B.
Pooled per-request, n=80 each side.

|                        | TTFT (ms) | E2E (ms) | per-step decode (ms)¹ |
|------------------------|----------:|---------:|----------------------:|
| PyT (v3..v7)           |    10.640 |   83.287 |                 3.823 |
| TRT                    |     5.894 |   80.446 |                 3.927 |
| **PyT − TRT**          | **+4.746**| **+2.840** |       **−0.104** (PyT win) |

¹ Per-step decode = (E2E − TTFT) / 19, derived per `REPRO.md` § "Reporting caveat".

E2E decomposition vs TRT:

```
ΔE2E   =  ΔTTFT  +  19 × Δper_step
+2.840  =  +4.746  +    -1.976
            ────         ────
           167% of      −70% of
            E2E gap     E2E gap
```

**The decode loop already structurally beats TRT by ~2 ms over 19 steps.
100% of the remaining E2E deficit lives in TTFT (prefill).** Every µs we
now spend on per-step decode work pushes the PyT lead further ahead but
cannot close the E2E gap alone.

What got us here (cumulative, vs piecewise-only baseline; pooled n=80 per side):

| stage                  | TTFT (ms) | E2E (ms) |
|------------------------|----------:|---------:|
| baseline (piecewise)   |    11.182 |   84.253 |
| v3+v4+v5               |    10.822 |   83.781 |
| v3+v4+v5+v6            |    10.744 |   83.410 |
| **v3+v4+v5+v6+v7**     | **10.640**|**83.287**|

v3..v7 = **−0.967 ms E2E (−1.15 %)** from baseline at p ≪ 1e-10.

## Key insight

The per-step decode loop is no longer the bottleneck. Further sampler-side
work yields diminishing returns on E2E vs TRT, even though it would widen
the per-step lead. **The only lever that can change the E2E sign vs TRT is
TTFT** (prefill `_forward_step` and per-request lifecycle around it).

## Ranked next steps

Rank order is **(impact on PyT−TRT E2E gap) × (confidence) / (effort)**.

### 1. TTFT / prefill investigation — HIGHEST priority

Per `POST_V5_DIVERGENCE.md` § "Per-prefill iter cost":

|                | PyT `_forward_step` (ctx) | TRT `enqueueV3` |
|----------------|--------------------------:|----------------:|
| mean / req     |                **5.91 ms**|         2.31 ms |
| n in window    |                        22 |              17 |

**3.6 ms / req is the entire `_forward_step` gap.** Add ~1.3 ms of
per-request lifecycle outside `_forward_step` (request fetch / lifecycle /
response dispatch) and you recover the full 4.93 ms TTFT delta exactly.

We have **not yet decomposed** what's inside that 5.91 ms. That's the next
investigation.

#### Sub-ranges to instrument inside `_forward_step` (prefill iter):

- KV-cache allocation / metadata setup
- Attention metadata prep (`AttentionMetadata` build + RoPE precompute)
- Per-layer transformer blocks (attention + MLP)
- Output projection / logits computation

#### Concrete starting action:

```bash
# 1. Add per-layer + per-prefill-stage NVTX inside _forward_step
#    (one-day spike on a feature branch). Suggested ranges:
#       prefill_setup    -> KV alloc + attn metadata + RoPE
#       prefill_layer_<i>-> per transformer block
#       prefill_logits   -> output projection
# 2. Run one nsys trace on the existing workload:
#    bash nvbugs_5615248/trtllm_bench/run_nsys_trace.sh pytorch \
#         nvbugs_5615248/trtllm_bench/nsys_ttft_phase1
# 3. Use analyze_decode_loop.py's Phase-2 logic adapted to *prefill* iters
#    (filter by NVTX category=ctx-only). The output is a per-sub-range
#    breakdown of the 5.91 ms.
# 4. Cross-reference with TRT's `enqueueV3` internals: TRT prefill at
#    ISL=100 typically batches per-layer kernel launches via the engine's
#    persistent execution plan. PyT prefill emits ~10x more launches per
#    layer at the same compute volume.
```

#### Likely outcomes & follow-ups:

| Decomposition outcome | Highest-leverage follow-up |
|---|---|
| Launch overhead dominates per-layer | Broader CUDA-graph capture for prefill (extend piecewise capture to cover all bs/seq combinations actually hit) |
| KV-cache init + attention metadata dominate | Hoist setup to `setup_sampler_step` / pre-warm pool, similar pattern to v4 |
| Per-layer kernel-level inefficiency dominates | Custom fused kernels for RoPE+attn+MLP (kernel-cuda or kernel-cute specialist territory) |
| Output projection dominates | Consider TRT-style fused logit selection for last-token-only prefill |

We don't know which row applies until step 3 above. Estimated effort to
get the decomposition: 1-2 days. Estimated effort for the follow-up itself:
varies wildly by row (1 day for KV-init hoist, 1-2 weeks for graph-capture
extension, 2-4 weeks for custom kernels).

#### Why this is #1:

- 100% of the remaining E2E deficit lives in TTFT. No other lever can move
  PyT past TRT on this workload.
- High confidence the gap is real (3.6 ms/req at the engine-step level is
  reproducible across all the nsys traces we have).
- Effort to decompose is 1-2 days; the *fix* effort is unknown but at least
  scoped after the decomposition.

### 2. v8 — sampler-step CUDA-graph capture (decode loop)

Per-step decode is launch-bound on PyT (M2 cudaLaunchKernel ratio = 0.108
post-v7 vs TRT's 0.049). A captured graph for the sampler step replaces
~25 launches/iter (`index_elementwise_kernel` × N + `mbtopk` chain +
softmax + scatter ops) with a single `cudaGraphLaunch`.

#### Estimated impact:

- ~1.5 ms / req of host budget recovered ⇒ ~0.08 ms / step ⇒ ~1.5 ms E2E.
- **Does not close the TTFT gap; widens an already-positive per-step lead.**

#### Effort:

- Medium-high. Beam-search sampler has data-dependent control flow:
  - The `if requests:` guard at the top of `sample_async`.
  - Fast-greedy fallback path inside `_process_requests`.
  - Multi-strategy iteration in `_sample_batched_by_strategy`.
- Requires explicit capture boundaries; needs a torch.compile-or-manual
  graph design discussion before code.

#### When to do it:

After step 1 has produced the TTFT decomposition. If TTFT turns out to
need broader graph capture anyway (most likely outcome), v8 may share
infrastructure with the prefill capture work and become cheaper.

### 3. Residual scatter-op fusion in `beam_search_sampling_batch` (decode)

10 458 `index_elementwise_kernel` launches in the steady-state window
(post-v5 baseline; v6 reduced this slightly). v3+v4 fused the worst seven
scatter sites; the long tail lives inside `beam_search_sampling_batch`
and `_handle_first_finish_reasons`.

#### Estimated impact:

- 200–500 µs / req ⇒ ≤ 0.5 ms E2E.
- Same caveat as v8: widens per-step lead, doesn't help E2E vs TRT.

#### Effort:

- Medium. Requires either a `torch.compile` boundary around
  `beam_search_sampling_batch` (with the data-dependent shapes locked
  via specialisation) or a small custom op.

### 4. `_write_finish_reasons` D2H deferral (decode)

`_write_finish_reasons` is 305 µs / iter post-v7, with a per-step async
D2H of the finish-reasons tensor. The consumer
(`_check_beam_search_stop_criteria`) might tolerate this being deferred
by one step — same structural pattern as v6.

#### Estimated impact:

- 50–100 µs / step ⇒ 1–2 ms E2E.
- Doesn't help TTFT.

#### Effort:

- Low. v6 established the deferral pattern; this is a copy-paste
  application to a different per-step D2H.

#### Risk:

- Need to verify deferring by one step doesn't change *when* a stop fires
  (i.e., we don't over-generate by one token in some edge case). The
  beam-search test suite has coverage for stop-word handling that should
  catch this.

#### Why this is #4 not #2:

- It *could* be a quick win if you want incremental progress while the
  TTFT investigation runs (low effort, well-trodden pattern).
- But it's the same shape of fix as v6/v7 — eats into already-positive
  per-step decode lead and doesn't move the E2E gap vs TRT.

## Recommended order

1. **Pivot to TTFT instrumentation (#1).** Spend 1-2 days on the
   per-sub-range NVTX decomposition of `_forward_step` (prefill). This is
   the only path that can flip the E2E sign vs TRT on this workload. The
   decomposition itself is low-risk; the *fix* it points at is unknown until
   we have the data.
2. **In parallel**, if you want to keep landing PRs while the investigation
   runs, do **#4 (`_write_finish_reasons` deferral)**. It reuses v6's
   pattern, has a clean correctness story, and is small enough to land
   without blocking on the larger TTFT work. Treat it as a stocking-stuffer.
3. **Defer v8 (#2)** until step 1 produces the TTFT decomposition. If
   TTFT turns out to need broader graph capture, v8 can ride on shared
   infrastructure. If not, v8 stands on its own merit but is a smaller win
   than the TTFT lever.
4. **Defer #3 (residual scatter fusion)** indefinitely unless step 1
   produces an unexpected null result and we need to keep harvesting from
   the decode loop.

## Open questions / risks

- **TRT's `_forward_step` advantage may be partly engine pre-baking** that
  PyT cannot fully replicate without static-shape compilation. If the
  decomposition reveals this is the dominant component, the realistic
  ceiling for PyT TTFT is "TRT + small fixed launch overhead", not
  "match TRT". This is fine — the goal is to close the gap as much as
  practical, not necessarily eliminate it.
- **Beam-search non-determinism** (FP tie-breaking in `multi-block top-k`,
  timing-dependent sampler ordering) means token-equivalence checks need
  to use the envelope-comparison approach we used for v6 and v7
  (10–14 / 16 stable across self-runs is the baseline noise floor). Any
  TTFT-side fix must preserve this envelope, not tighten or loosen it.
- **CUDA-graph extensions for prefill** would interact with the existing
  piecewise-capture filter (`_filter_piecewise_capture_num_tokens` from
  commit `fb0acdde05`). Any change to capture shape needs to validate the
  existing piecewise infrastructure still holds for both prefill and
  decode iters.

## Out of scope

- Multi-strategy batching, multi-request concurrency, or non-beam paths.
  v6 and v7's gains apply to those workloads if and only if the per-group
  / per-request fall-back paths are hit, which is the same as before.
- TensorRT backend changes. v3..v7 are PyT-only; v8 and the TTFT work
  proposed here are also PyT-only.
- Long-OSL workloads (≥ 64). v3..v7 measurements all use OSL=20 to keep
  the validation comparable; gains scale linearly with OSL but variance
  also grows.
