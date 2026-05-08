# NVBug 5615248 — Post-v7 plan

> **Last updated: 2026-05-07 (second revision).** Two iterations of analysis
> on the same day produced two different stories. The current truth is at
> the bottom of the day — see "Apples-to-apples graphnode comparison
> (2026-05-07 second revision)". GPU work is essentially tied between PyT
> and TRT prefill; **all** of the TTFT gap is host-side, with the bulk
> (~3 ms / 4.75 ms) in the *first* `_sample_async` after prefill, not in
> the prefill iter itself. Earlier sections in this doc record the path
> we took to get here — they are kept as a learning trail and **the
> recommendations there are superseded**.

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

## Prefill iter — GPU vs host decomposition (2026-05-07 update)

Per-iter median for the post-warmup **prefill** iter, derived from
`nvbugs_5615248/trtllm_bench/nsys_optimized_v7_pyt/pyt_beam10.sqlite` and
`…/nsys_optimized_v5_trt/trt_beam10.sqlite`. The PyT row now includes
piecewise-CUDA-graph contents via `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`; without
that table the on-GPU time of the captured graphs (LayerNorm + QKV proj +
MLP + residuals) is invisible because nsys records each piecewise graph as a
single opaque `cudaGraphLaunch` event by default.

| Component                                          | PyT v7   | TRT      | Δ (PyT − TRT) |
|----------------------------------------------------|---------:|---------:|--------------:|
| Wall (NVTX `_forward_step` ctx / `enqueueV3`)      | 5.30 ms  | 2.32 ms  | **+2.98 ms**  |
| Piecewise-graph spans (`GRAPH_TRACE`, n=23 in PyT) | 3.68 ms  | 0.00 ms  |   +3.68 ms    |
| Discrete kernels (`CUPTI_KERNEL`)                  | 0.42 ms  | 2.08 ms  |   −1.66 ms    |
| **Effective GPU active** (graph + discrete)        | **4.10 ms** | **2.08 ms** | **+2.02 ms** |
| **Host-only** (wall − effective GPU active)        | **1.20 ms** | **0.24 ms** | **+0.96 ms** |
| `cudaGraphLaunch` API time                         | 0.34 ms  | 0.00 ms  |   +0.34 ms    |
| `cudaLaunchKernel` API time                        | 0.49 ms  | 0.78 ms  |   −0.29 ms    |
| Discrete kernel count                              |       74 |      128 | −             |
| Graph-trace count                                  |       23 |        0 | (= 1/layer in PyT) |

The 23 PyT piecewise spans match an expected layout: roughly one captured
graph per transformer layer (input_norm + QKV proj or post-attn norm + MLP +
residual), plus one for `final_norm + lm_head`. TRT prefill is **not**
graph-captured (`trt.yaml` only enables `cuda_graph_mode` for the generation
phase per the comment), so all 128 of TRT's prefill kernels are visible
directly.

**The 2.98 ms PyT − TRT prefill gap splits roughly 2:1 GPU vs host:**

```
ΔPrefill_iter = +2.98 ms
              = +2.02 ms more on-GPU work     (~68 %)
              + +0.96 ms more host-side time  (~32 %)
```

This is a material correction to the pre-2026-05-07 framing in this doc and
in `POST_V5_DIVERGENCE.md`, both of which assumed the prefill gap was
overwhelmingly host-launch-bound (driven by `M2 = 0.135` cudaLaunchKernel
ratio in the **decode-loop** steady-state window). That M2 number is real
but it characterises the *decode loop*, not the prefill iter; the prefill
iter is dominated by ~3.7 ms of graph-captured GPU work that the existing
analyser couldn't see.

### Likely sources of the +2 ms GPU-side gap (need confirmation)

The `--cuda-graph-trace=node` re-run proposed below will turn each piecewise
graph into per-kernel records and let us bucket the difference. The leading
hypotheses, ordered by expected contribution:

1. **Kernel fusion gap inside the captured graph.** TRT prefill shows fused
   kernels like `__myl_AddCastMulMeanAddSqrtDivMulCastMul_*` (RMSNorm chain
   collapsed: mean + rsqrt + cast + mul + residual into one kernel) and
   `__myl_SiluMul_*` (SiLU + multiply fused). PyT's piecewise graph likely
   captures discrete `pow / mean / rsqrt / mul / cast / silu / mul` kernels
   per layer. At ISL=100 each is launch- and memory-bound rather than
   math-bound, so the per-layer kernel count multiplies the elapsed time.
2. **Tile-size mismatch on small GEMMs.** TRT engine builder picks
   `tilesize64x96x64` and `tilesize32x32x64` for these prefill GEMMs at M=100;
   PyT uses cuBLAS heuristic picks (`s16816gemm_64x64`, `s1688gemm_128x128`,
   etc.) without per-shape tuning. At M=100 the heuristic often wastes rows.
3. **Piecewise carve-out boundary cost.** If the residual add at the end of
   layer N and the input_norm at the start of layer N+1 are in *different*
   piecewise graphs, every layer pays a "exit graph → re-enter graph"
   boundary (cudaGraphLaunch + memory round-trip on the residual tensor).
   23 graphs × ~14 µs/launch from CUPTI ≈ 0.32 ms of pure boundary cost.

### Likely sources of the +0.96 ms host-side gap

- `_prepare_inputs` (Python): 0.51 ms / prefill iter on PyT, ~0 on TRT.
- Per-prefill PyExecutor scheduling: KV-cache alloc + attention metadata
  build + per-layer Python dispatch around each `cudaGraphLaunch`.
- `cudaGraphLaunch` API itself: 0.34 ms / iter (vs 0.00 on TRT, but counted
  in the 0.96 above only as the wall-time complement, not as a separate
  charge).

These are still real but smaller than the GPU-side delta, so they drop in
priority compared to the original "instrument _forward_step and chase host
overhead" plan.

> **Superseded.** The +2.02 ms GPU-side delta above is a windowing
> artefact. See "Apples-to-apples graphnode comparison (2026-05-07
> second revision)" below for the corrected numbers — GPU work is
> essentially tied between the two backends.

## Apples-to-apples graphnode comparison (2026-05-07 second revision)

After re-running both backends with `nsys profile --cuda-graph-trace=node`
under identical flags
(`nsys_v7_graphnode_pyt/pyt_beam10.sqlite` and
 `nsys_v5_graphnode_trt/trt_beam10.sqlite`),
and using a request-level prefill window on **both** sides — PyT uses NVTX
`[Executor] _forward_step` (ctx-only) which is a host-blocking span; TRT
uses `[enqueueV3_start, first_decode_graph_kernel_start)` because
`enqueueV3` is async-submit and the NVTX span only covers the host-side
submit (this is what the previous revision got wrong).

### Prefill iter, fair window

|                           | PyT v7 graphnode | TRT graphnode | Δ (PyT − TRT) |
|---------------------------|-----------------:|--------------:|--------------:|
| Wall                      |       **5.70 ms**|   **4.66 ms** |   **+1.05 ms**|
| GPU work                  |        4 063 µs  |    4 107 µs   |     **−44 µs**|
| Kernel count              |          261     |      254      |        +7     |
| Host-only (wall − GPU)    |        ~1.64 ms  |   ~0.55 ms    |   **+1.09 ms**|

GPU work in prefill is essentially identical (PyT actually 44 µs faster).
The previous "TRT does 2 ms less GPU work" finding from the first
revision came from comparing the PyT NVTX `_forward_step` span (which
includes the host wait, ~5.30 ms) against the TRT NVTX `enqueueV3` span
(which only covers the host-side submit, ~2.32 ms). The async tail of
TRT's prefill (the rest of the GPU work after `enqueueV3` returns) was
invisible to that window. Once we fix the windowing, the two backends do
the same amount of GPU work.

### Per-category prefill kernel breakdown (fair window)

| Category               | PyT n×us           | TRT n×us           | Δ µs       |
|------------------------|--------------------|--------------------|-----------:|
| gemm (model body)      | 88 × 3 430.4       | 89 × 3 481.8       |     −51    |
| gemv / lm_head         |  1 ×   183.8 (gemvx)|  1 × 194.4 (xmma_gemm) |  −11    |
| FusedAddRMSNorm / `__myl_AddCastMul…` | 44 × 116.7 | 26 × 78.1     |     +39    |
| `silu_and_mul` / `__myl_SiluMul`      | 22 ×  38.2 | 12 × 40.8 (TRT splits) | −3 |
| attention (fmha_v2)    | 22 × 106.4         | 22 × 101.9         |      +4    |
| RoPE + KV update       | 22 ×  61.8         | 22 ×  60.1         |      +2    |
| attn metadata          | 22 ×  54.5         | 22 ×  56.1         |      −2    |
| splitK reduce          | 22 ×  43.3         |  0 ×   0.0         |     +43    |
| TRT-only beam-search C++ kernels (insertUnfinishedPath, batchApplyPenalty, …) | 0 | several | −200 |
| **TOTAL GPU**          | 4 063 µs           | 4 107 µs           |    **−44** |

GEMM tile / shape choices differ slightly (PyT down_proj is 22 ×
`s1688gemm_128x128` at 79.2 µs each = 1 742 µs; TRT down_proj is 22 ×
`xmma_gemm_64x96x64` at 73.1 µs each = 1 609 µs). That's a ~6 µs / GEMM
edge for TRT on the largest GEMM, but PyT recovers it on the smaller
GEMMs. **Fusion is matched**: PyT already has `FusedAddRMSNormKernel` and
`silu_and_mul_kernel`; the previous revision's "fuse RMSNorm/SiluMul"
hypothesis was wrong. **Tile choice is roughly matched**: there is no
~2 ms GEMM gap to capture.

### TTFT decomposition (clean reference, no nsys overhead)

|                                          |    PyT     |    TRT    | Δ (PyT − TRT) |
|------------------------------------------|-----------:|----------:|--------------:|
| Prefill iter wall                        |   5.70 ms  |  4.66 ms  |   **+1.05 ms**|
| Post-prefill (ctx_end → first token end) |   4.28 ms  |  ~1.23 ms |   **+3.05 ms**|
| Pre-prefill / scheduling lifecycle       |   ~0.7 ms  |  ~0.0 ms  |    ~+0.65 ms  |
| **TTFT total** (clean reference)         | **10.64 ms** | **5.89 ms** | **+4.75 ms** |

**~64 % of the TTFT gap (~3 ms / 4.75 ms) lives in the post-prefill
window — not in the prefill iter.** That window is the time between the
prefill iter ending and the first generated token being ready to send to
the streaming consumer. Only 467 µs of that 4 275 µs PyT window is GPU
work (10.9 %); the rest is pure host execution.

### Post-prefill NVTX breakdown (PyT, 4.28 ms median window)

```
_sample_async                  2 768 µs   ← first beam-search sample after prefill
  └─ _process_requests         1 826 µs
     └─ sample_batched_by_strategy  1 400 µs
_prepare_inputs                  576 µs   (prep for the first gen iter)
_write_finish_reasons            301 µs
setup_sampler_step               253 µs
maybe_create_beam_histories      129 µs
update_original_tokens            79 µs
prepare_resources                 77 µs
_fetch_new_requests               45 µs
... (all others < 30 µs)
TOTAL window wall              4 275 µs
GPU work in window               467 µs   (10.9 %)
```

The single dominant cost is the **first invocation** of `_sample_async`
(2 768 µs), specifically the `sample_batched_by_strategy` path inside
`_process_requests`. Steady-state `_sample_async` is ~2 555 µs / iter
(per `analyze_decode_loop.py` § Phase-2 on the same trace), so the
first invocation is only ~10 % more expensive than steady-state. **The
gap vs TRT is not that the first sample is bigger than steady-state on
PyT — it's that TRT does this work as part of the `enqueueV3` GPU stream
(beam-search book-keeping like `insertUnfinishedPathKernel`,
`batchApplyPenalty`, `radix_topk_one_block_kernel`, `addCumLogProbs`,
`copyBeamHypotheses`, `finalizeKernel` runs *inside the prefill GPU
work*, all visible in the TRT prefill kernel breakdown above), while PyT
runs the equivalent as a separate Python-driven sampler step after
prefill.**

This is structural: TRT's beam-search C++ runtime invokes the sampler
inline at the end of the engine pass; PyT's sampler is a separate Python
function called by the PyExecutor *after* `_forward_step` returns.

## Key insight (2026-05-07 second revision)

The per-step decode loop is no longer the bottleneck (PyT structurally
beats TRT by ~0.10 ms per gen iter). **100 % of the remaining E2E deficit
lives in TTFT, and 100 % of the TTFT delta is host-side.** Splitting the
host delta:

- ~1 ms / 4.75 ms inside the prefill `_forward_step` (Python overhead
  in `_prepare_inputs`, the 23 per-layer `cudaGraphLaunch` API calls,
  attention metadata build).
- **~3 ms / 4.75 ms in the *first* `_sample_async` after prefill**, where
  PyT runs the post-prefill beam-search sampler as a separate Python
  iteration while TRT runs the equivalent inline as part of the
  prefill GPU stream.

The single highest-leverage lever is therefore **collapsing the
prefill→first-decode handoff**: either (a) make the first sample run
inline as part of the prefill iter (no separate Python iteration), or
(b) capture both prefill and the first sample in a single CUDA graph so
they share the same launch budget. Neither was on the previous radar;
both supersede the GPU-fusion / tile-tuning hypothesis from the first
revision.

## Ranked next steps (2026-05-07 second revision)

Rank order is **(impact on PyT−TRT TTFT gap) × (confidence) / (effort)**.
The previous revision's items 1-4 (per-kernel graphnode decomposition,
RMSNorm/SiluMul fusion, piecewise carve-out tightening, cuBLAS tile
tuning) are **dropped** — the apples-to-apples comparison shows GPU
work is tied between the two backends, so all of those items target
non-existent gaps. The new top items target the host-side gap directly.

### 1. Decompose the first `_sample_async` after prefill — HIGHEST priority

~3 ms / 4.75 ms TTFT (≈ 64 %) of the gap lives here. PyT runs the
post-prefill beam-search sampler as a separate Python iteration
(`_sample_async` / `_process_requests` / `sample_batched_by_strategy`,
2 768 µs total wall on the median first-iter post-prefill window).
TRT runs the equivalent work inline as part of `enqueueV3` (e.g.
`insertUnfinishedPathKernel`, `batchApplyPenalty`,
`radix_topk_one_block_kernel`, `addCumLogProbs`, `copyBeamHypotheses`,
`finalizeKernel`, `gatherId`, `last_filter_kernel` all visible in the
TRT prefill kernel breakdown).

#### Sub-stages to instrument inside the first `_sample_async`:

- `setup_sampler_step` (first call materialises beam-search state for 10
  beams from 1 prefill output)
- `_prepare_beam_search` for the *first* gen iter (different code path
  from steady-state)
- Beam-search fast path vs multi-strategy path (the multi-strategy
  iteration inside `_sample_batched_by_strategy` is hit on every
  iteration including the first, even at single-strategy beam=10)
- Per-beam `cache_indirection_swap` first invocation (sets up cache
  indirection layout for beam=10)
- `_handle_first_finish_reasons` first call (different code path from
  steady-state)
- D2H copies whose first call materialises slices that subsequent calls
  only update incrementally

#### Concrete starting action:

```bash
# 1. Add NVTX ranges around the sub-stages above inside _sample_async
#    and _process_requests. ~1 day of NVTX patching.
# 2. Re-run the existing graphnode trace setup — see
#    "Diagnostic recipe (graphnode trace command of record)" below.
# 3. Bucket NVTX ranges by "first iter after ctx" vs "steady-state gen iter"
#    using the analyze_decode_loop.py pattern adapted to use the
#    "ctx_end -> first_gen_end" window from this revision's analysis.
#    Productise the ad-hoc Python from the chat trail into
#    nvbugs_5615248/trtllm_bench/analyze_prefill_handoff.py.
# 4. The sub-stage that has the largest first-vs-steady-state delta on
#    PyT is the lever.
```

#### Likely outcomes & follow-ups:

| Outcome                                      | Highest-leverage follow-up                                             |
|----------------------------------------------|------------------------------------------------------------------------|
| Beam-search state init dominates first call  | Hoist setup into a one-time pool warmup at PyExecutor startup          |
| Multi-strategy dispatch overhead dominates   | Fast-path single-strategy at beam=10 to skip the multi-strategy machinery |
| First-call-only D2H copies dominate          | Pre-allocate the D2H staging buffers; same pattern as v6               |
| The whole sampler iter is structurally large | Move the first sample inline to the prefill iter (item 2 below)        |

#### Why this is #1:

- Single largest source of TTFT delta (3 ms / 4.75 ms = 64 %).
- High confidence the gap is in this window — directly measured, not
  inferred (4.275 ms wall, 467 µs GPU, 11 % GPU active is unambiguously
  host-bound).
- Decomposition is ~1 day; the fix effort depends on which sub-stage
  dominates (1 day to weeks).

### 2. Make the first sample inline to the prefill iter (LARGEST possible win, medium-high effort)

The structural reason TRT's TTFT is lower is that it runs beam-search
sampling *inside* the prefill GPU stream as part of `enqueueV3`, while
PyT runs it as a *separate* Python iteration after `_forward_step`
returns. Closing this structurally requires either:

- **(a) Inline the first sample into `_forward_step` ctx-only.** When the
  `_forward_step` is processing a context-only request, after the model
  forward, run the sampler-graph for the first token before returning
  to the PyExecutor scheduler. Saves the Python re-entry cost
  (~1 ms / 4.28 ms) and lets the GPU pipeline the sampler kernels
  behind the model forward without a host sync.

- **(b) Single CUDA-graph capture covering prefill + first sample.**
  Capture the model forward (already piecewise-captured) and the first
  sampler iteration as a single graph. Replaces the post-prefill
  `_sample_async` with one `cudaGraphLaunch`. This is also v8
  (sampler-step CUDA graph capture) but applied to the *first*
  invocation specifically, which is in TTFT critical path.

#### Estimated impact:

- (a) recovers ~1.5–2 ms / req (most of the Python re-entry cost in
  the post-prefill window); ~30–40 % of TTFT delta.
- (b) recovers ~2.5–3 ms / req (most of the host-side cost in the
  4.28 ms post-prefill window); ~50–60 % of TTFT delta.

#### Effort:

- (a) Medium. Touches PyExecutor scheduling state machine; the sampler
  has a beam-search dependency on the prefill output that needs
  threading through the inline path.
- (b) Medium-high. Beam-search sampler has data-dependent control flow
  (multi-strategy dispatch, fast-greedy fallback); explicit capture
  boundaries are needed. Same engineering as item 6 of the previous
  revision (deferred v8 sampler-graph capture), but applied to the
  first invocation only — narrower scope.

#### When to do it:

After item 1 produces the sub-stage decomposition. If the bulk of the
2.77 ms `_sample_async` is in beam-search book-keeping that doesn't
fit a graph capture cleanly, fall back to (a). If most of the cost is
in the steady-state-style sampler kernels (top-k, softmax, scatter),
(b) is cheaper.

### 3. Trim the +1.05 ms host overhead inside `_forward_step` ctx iter

This is the smaller of the two host-side gaps but cleanly addressable.
Within the prefill iter:

- `_prepare_inputs` (Python): 0.51 ms / iter on PyT, ~0 on TRT.
  Lift the int64 casts and `arange` index tensors out of the prefill
  critical path (cache them with a shape-keyed dict; the prefill shape
  set at ISL=100 is small).
- 23 × cudaGraphLaunch dispatch overhead: ~0.34 ms / iter (from the
  PyT prefill kernel breakdown). One graph per layer non-attn slice
  plus one for `final_norm + lm_head`. If the residual + next-layer
  norm boundary is forcing the split, merging adjacent slices halves
  the launch count to ~12.
- Residual ~0.2 ms in attention metadata build / KV-cache alloc.
  Hoist into a pre-warmed pool at PyExecutor startup, similar to v4's
  `seq_slots_long`.

#### Estimated impact:

- ~0.5–1.0 ms / req TTFT (all of the +1.05 ms intra-prefill-iter host
  delta). Smaller than items 1 and 2 but cleaner correctness story.

#### Effort:

- Low–medium. Each sub-fix is 1–2 days; can be landed independently.

### 4. Add NVTX inside `_forward_step` for prefill phases (small effort, instrumentation only)

Instrumentation only — doesn't fix anything by itself. Adds the
sub-ranges `prefill_setup` / `prefill_layer_<i>` / `prefill_logits` so
future regressions in the intra-prefill-iter host budget are caught
automatically by `analyze_decode_loop.py`. Pairs naturally with item 1's
NVTX additions to `_sample_async` (single PR can cover both).

- Impact: 0 directly; enables item 3's sub-fix selection and item 1's
  prefill-side decomposition.
- Effort: half-day patch.

### 5. v8 — sampler-step CUDA-graph capture (steady-state decode)  *[deferred]*

Per-step decode is launch-bound on PyT (M2 cudaLaunchKernel ratio = 0.108
post-v7 vs TRT's 0.049). Captured graph for the steady-state sampler
step replaces ~25 launches/iter with one `cudaGraphLaunch`.

- Impact: ~1.5 ms / req of host budget recovered ⇒ ~0.08 ms / step ⇒
  ~1.5 ms E2E. **Does not close the TTFT gap; widens an already-positive
  per-step lead.**
- Effort: medium-high.
- When: after items 1 and 2. If item 2(b) is done first, it provides
  most of v8's infrastructure for free.

### 6. Residual scatter-op fusion in `beam_search_sampling_batch` (decode)  *[deferred]*

10 458 `index_elementwise_kernel` launches in the steady-state window
(post-v5 baseline; v6/v7 reduced this somewhat). Long tail in
`beam_search_sampling_batch` and `_handle_first_finish_reasons`.

- Impact: 200–500 µs / req ⇒ ≤ 0.5 ms E2E. Doesn't help TTFT.
- Effort: medium.

### 7. `_write_finish_reasons` D2H deferral (decode)  *[deferred]*

305 µs / iter post-v7, per-step async D2H of finish-reasons. Same
structural deferral pattern as v6.

- Impact: 50–100 µs / step ⇒ 1–2 ms E2E. Doesn't help TTFT.
- Effort: low.
- Risk: stop-word edge case; covered by existing tests.

## Recommended order (2026-05-07 second revision)

1. **Item 1 first.** Add NVTX inside `_sample_async` /
   `sample_batched_by_strategy` for the sub-stages listed in item 1.
   Re-run the existing graphnode trace, then bucket "first iter after
   ctx" vs "steady-state". Estimated ~1 day. This single decomposition
   decides between item 2(a) and item 2(b).
2. **Item 4 in parallel.** Same NVTX patch covers prefill phases too
   (`prefill_setup` / `prefill_layer_<i>` / `prefill_logits`). Half-day
   add-on; enables item 3.
3. **Item 2(a) or 2(b) based on item 1's outcome** (most likely 2(a) —
   inlining the first sample into `_forward_step`). 1–2 weeks.
4. **Item 3 in parallel** with item 2 if engineering bandwidth allows.
   ~1–2 days per sub-fix.
5. **Items 5–7 deferred.** All decode-side work, doesn't help TTFT.
   Pick up only if items 1–3 produce an unexpected null result on TTFT.

## Diagnostic recipe (graphnode trace command of record)

The single profiling command that produced the second-revision analysis.
Use this anytime the prefill iter or post-prefill window needs to be
re-measured. Drop the `--capture-range=cudaProfilerApi` flag from the
first-revision recipe — `trtllm-bench` does not call
`cudaProfilerStart()`, so that flag would have produced empty traces (it
is what bricked the May 6 `nsys_ttft_phase1` attempt).

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM
MODEL=/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
WORKDIR=nvbugs_5615248/trtllm_bench
ENGINE_DIR=nvbugs_5615248/tinyllama_trt_engine
DATASET=$WORKDIR/dataset_isl100_osl20.jsonl

# (a) PyTorch trace with per-graph-node tracing
OUT=$WORKDIR/nsys_v7_graphnode_pyt
mkdir -p "$OUT"
nsys profile \
    --output "$OUT/pyt_beam10" \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --force-overwrite=true --sample=none --cpuctxsw=none \
    trtllm-bench \
        --model "$MODEL" --model_path "$MODEL" --workspace "$WORKDIR" \
        throughput --backend pytorch --config "$WORKDIR/pytorch.yaml" \
        --dataset "$DATASET" --concurrency 1 --warmup 3 --num_requests 16 \
        --beam_width 10 --max_batch_size 1 --streaming \
        --report_json "$OUT/report_pytorch.json" \
        --output_json "$OUT/output_pytorch.json" \
        --request_json "$OUT/request_pytorch.json" \
        2>&1 | tee "$OUT/run_pytorch.log"
nsys export --type sqlite --force-overwrite=true \
    --output "$OUT/pyt_beam10.sqlite" "$OUT/pyt_beam10.nsys-rep"

# (b) TRT trace with the same flags
OUT=$WORKDIR/nsys_v5_graphnode_trt
mkdir -p "$OUT"
nsys profile \
    --output "$OUT/trt_beam10" \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --force-overwrite=true --sample=none --cpuctxsw=none \
    trtllm-bench \
        --model "$MODEL" --model_path "$MODEL" --workspace "$WORKDIR" \
        throughput --backend tensorrt --engine_dir "$ENGINE_DIR" \
        --config "$WORKDIR/trt.yaml" \
        --dataset "$DATASET" --concurrency 1 --warmup 3 --num_requests 16 \
        --beam_width 10 --max_batch_size 1 --streaming \
        --report_json "$OUT/report_trt.json" \
        --output_json "$OUT/output_trt.json" \
        --request_json "$OUT/request_trt.json" \
        2>&1 | tee "$OUT/run_trt.log"
nsys export --type sqlite --force-overwrite=true \
    --output "$OUT/trt_beam10.sqlite" "$OUT/trt_beam10.nsys-rep"

# (c) Decode-loop analyser still works as-is
python3 nvbugs_5615248/trtllm_bench/analyze_decode_loop.py \
    --sqlite $WORKDIR/nsys_v7_graphnode_pyt/pyt_beam10.sqlite \
    --skip-prefills 5 --top-n 30
```

For the prefill / post-prefill analysis specifically, the decode-loop
analyser is not enough — see the inline ad-hoc Python in
`nsys_v5_graphnode_trt`/`nsys_v7_graphnode_pyt` (chat history) for the
request-level windowing. Productising that into
`analyze_prefill_handoff.py` is part of item 1 above.

## nsys-overhead caveat

The graphnode trace inflates wall-clock per-iter timings above the clean
reference numbers, with the inflation itself being asymmetric between
the two backends:

| metric              | clean ref | graphnode trace | nsys overhead |
|---------------------|----------:|----------------:|--------------:|
| PyT TTFT            |  10.64 ms |        14.43 ms |     +3.79 ms  |
| PyT per-step decode |   3.83 ms |         4.15 ms |     +0.32 ms  |
| TRT TTFT            |   5.89 ms |         6.36 ms |     +0.47 ms  |
| TRT per-step decode |   3.93 ms |         4.04 ms |     +0.11 ms  |

PyT's nsys overhead is ~8× TRT's. This is itself evidence the gap is
host-bound — nsys instruments host-side activity (Python frames,
runtime API calls), so a backend with more host-side activity per iter
gets more inflated. The kernel-by-kernel breakdown still holds because
nsys overhead is mostly between-kernel, not inside-kernel.

Use clean reference numbers (the `nvbugs_5615248/trtllm_bench/optimized_v7_pyt_fresh`
and `…/optimized_v5_trt_fresh` directories) for any number that goes into
a PR description or commit message; use graphnode numbers only for
relative kernel-mix analysis.

## Open questions / risks

- **TRT's TTFT advantage is structural, not algorithmic.** TRT's
  beam-search C++ runtime invokes the post-prefill sampler inline as
  part of the engine pass (`insertUnfinishedPathKernel`,
  `batchApplyPenalty`, `radix_topk_one_block_kernel`, `addCumLogProbs`,
  `copyBeamHypotheses`, `finalizeKernel`, `gatherId`,
  `last_filter_kernel` all visible in the TRT prefill kernel
  breakdown). PyT runs the same logical work as a Python iteration
  *after* `_forward_step` returns. Closing this gap requires changing
  the PyExecutor → sampler control flow, not optimising kernels.
  Realistic ceiling for PyT TTFT is "TRT + small fixed Python re-entry
  overhead", not "match TRT bit-for-bit". Goal remains to close the gap
  as much as practical.
- **First-call sampler costs match steady-state, not the gap.** The
  first `_sample_async` after prefill is 2.77 ms; steady-state
  `_sample_async` is 2.55 ms. The first call is only ~10 % more
  expensive than steady-state on PyT. The gap vs TRT is **not** that
  the first call has unusual cost — it's that TRT amortises the same
  work into the prefill GPU stream where PyT doesn't. Item 1's NVTX
  decomposition needs to verify this; if it turns out the first call
  *is* unusually expensive, the lever shifts toward "warm up the
  beam-search-state pool at PyExecutor startup" rather than item 2's
  graph-capture work.
- **Graphnode nsys overhead asymmetry could distort the ratio.** PyT
  graphnode trace adds 3.79 ms TTFT overhead, TRT adds 0.47 ms — i.e.
  the *graphnode-trace gap* is +8.07 ms and the *clean-reference gap*
  is +4.75 ms. The kernel-by-kernel decomposition is still valid (most
  nsys overhead is in between-kernel host instrumentation, not inside
  CUPTI kernel records), but absolute timings should be cross-checked
  against the clean-reference runs before committing to a fix's
  expected-impact estimate.
- **Beam-search non-determinism** (FP tie-breaking in `multi-block top-k`,
  timing-dependent sampler ordering) means token-equivalence checks need
  to use the envelope-comparison approach we used for v6 and v7
  (10–14 / 16 stable across self-runs is the baseline noise floor). Any
  TTFT-side fix must preserve this envelope, not tighten or loosen it.
- **Item 2(b)'s graph capture for prefill+sample** would interact with
  the existing piecewise-capture filter
  (`_filter_piecewise_capture_num_tokens` from commit `fb0acdde05`).
  Any change to capture shape needs to validate the existing piecewise
  infrastructure still holds for both prefill and decode iters. Item
  2(a) (inline-into-`_forward_step`) does not have this constraint and
  is the safer first attempt.

## Out of scope

- Multi-strategy batching, multi-request concurrency, or non-beam paths.
  v6 and v7's gains apply to those workloads if and only if the per-group
  / per-request fall-back paths are hit, which is the same as before.
- TensorRT backend changes. v3..v7 are PyT-only; v8 and the TTFT work
  proposed here are also PyT-only.
- Long-OSL workloads (≥ 64). v3..v7 measurements all use OSL=20 to keep
  the validation comparable; gains scale linearly with OSL but variance
  also grows.
