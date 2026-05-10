# NVBug 5615248 — `pyt-workflow-fixes` branch validation

Reproduction guide for the **PyTorch-workflow beam-search host-overhead
fixes** on the `user/brb/pyt-workflow-fixes` branch. Compares three
incremental PyTorch refs against each other and against a TensorRT
backend baseline on the customer's TinyLlama beam=10 / ISL=100 / OSL=20 /
concurrency=1 / streaming workload.

NVBug: <https://nvbugspro.nvidia.com/bug/5615248>

For older context, see:

- [`REPRO_V6_V7.md`](./REPRO_V6_V7.md) — analogous v6+v7 experiment on the
  source branch (`user/brb/nvbug-5615248-beam-host-overhead`); same workload
  shape but the commits there were squashed/reordered into the three
  upstream PRs benched here.
- [`REPRO_HISTORICAL.md`](./REPRO_HISTORICAL.md) — the original
  PyTorch-vs-TRT TTFT comparison doc carried over from the source branch.
- [`POST_V5_DIVERGENCE.md`](./POST_V5_DIVERGENCE.md) — divergence map
  identifying which gaps remained after the v3-v5 fixes; the branch under
  test addresses the v6/v7 buckets.

## What's being measured

Three PyTorch refs (incremental) and one TensorRT baseline:

| Label | SHA | Title |
|---|---|---|
| `baseline_piecewise` (PyT) | `435580404c` | `[https://nvbugs/5615248][fix] Broader capture of piecewise cudagraph` |
| `plus_handoff` (PyT) | `0115bd2430` | `[https://nvbugs/5615248][fix] Reduce beam-search prefill->decode handoff cost` |
| `plus_handoff_beamhist` (PyT) | `00e8b46370` | `[https://nvbugs/5615248][fix] Beam history copies only on terminal steps` |
| `baseline_trt` (TRT) | n/a (prebuilt engine) | TensorRT runtime against `nvbugs_5615248/tinyllama_trt_engine` |

The two PyTorch fixes (`0115bd2430` and `00e8b46370`) target disjoint
phases:

- **`0115bd2430` "Reduce handoff cost"** — single one-shot prefix→decode
  handoff fix; expected to move TTFT only.
- **`00e8b46370` "Beam history terminal-step copies"** — defers per-step
  D2H copies into the terminal step of the decode loop; expected to move
  per-token ITL (and E2E by extension).

Both edits are pure-Python (`tensorrt_llm/_torch/pyexecutor/{sampler,sampling_utils}.py`),
so toggling refs requires **no rebuild** — `trtllm-bench` re-imports the
modules on every launch.

## Workload (TinyLlama beam=10)

- Model: `TinyLlama-1.1B-Chat-v1.0`
  (`/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0`)
- Hardware: NVIDIA L40S (single GPU)
- Geometry: ISL=100, OSL=20, beam_width=10, concurrency=1, max_batch_size=1, streaming
- Dataset: [`dataset_isl100_osl20.jsonl`](./dataset_isl100_osl20.jsonl) — 16 measurement requests, 3 warmup
- Driver: [`run_multirun_pytorch.sh`](./run_multirun_pytorch.sh) /
  [`run_multirun_trt.sh`](./run_multirun_trt.sh) invokes `trtllm-bench` 5×
  per ref (5 × 16 = n=80 pooled per-request samples per ref)
- PyT YAML ([`pytorch.yaml`](./pytorch.yaml)): max_seq_len=129, piecewise
  CUDA graph enabled, dense `capture_num_tokens=[1,2,3,4,8,12,16,...,128]`
  buckets so every `num_tokens` value this workload reaches has a dedicated
  captured graph
- TRT YAML ([`trt.yaml`](./trt.yaml)): `cuda_graph_mode=true`, `multi_block_mode=true`;
  geometry baked into the prebuilt engine at
  `nvbugs_5615248/tinyllama_trt_engine` (max_seq_len=129, max_batch_size=1,
  max_beam_width=10)

## Files in this directory

| File | Role |
|---|---|
| [`pytorch.yaml`](./pytorch.yaml) | PyT `--config` |
| [`trt.yaml`](./trt.yaml) | TRT `--config` |
| [`dataset_isl100_osl20.jsonl`](./dataset_isl100_osl20.jsonl) | 16-request synthetic dataset |
| [`run_multirun_pytorch.sh`](./run_multirun_pytorch.sh) | Per-ref launcher: 5×`trtllm-bench --backend pytorch` |
| [`run_multirun_trt.sh`](./run_multirun_trt.sh) | Per-ref launcher: 5×`trtllm-bench --backend tensorrt` |
| [`run_3ref_bench.sh`](./run_3ref_bench.sh) | 3-ref incremental driver (toggles PyT refs via narrow `git checkout`, no rebuild) |
| [`run_trt_baseline_compare.sh`](./run_trt_baseline_compare.sh) | TRT baseline + cross-backend aggregations against the three PyT refs |
| [`aggregate_runs.py`](./aggregate_runs.py) | Robust stats aggregator (median + bootstrap CI + Mann-Whitney + Hodges-Lehmann; mean + Welch's t informational) |
| `three_ref_validation/` | Output dir (one subdir per ref + the TRT baseline) |

## How to reproduce

Inside the TRT-LLM container, from the repo root, on a node with one L40S
visible. The current branch HEAD must be `user/brb/pyt-workflow-fixes`
(otherwise the default `HEAD~2` / `HEAD~1` / `HEAD` refs won't resolve to
the right commits).

### Step 1 — Three-ref PyT bench (baseline + two incrementals)

```bash
bash nvbugs_5615248/trtllm_bench/run_3ref_bench.sh
```

The driver:

1. Pre-flights `trtllm-bench`, `git`, `python3`, the harness, and asserts
   `BASELINE_REF..PLUS_V2_REF` only touches the expected sampler files.
2. Resolves all three refs to immutable SHAs **before** the first checkout.
3. Refuses to start with a dirty tracked working tree (untracked
   `nvbugs_5615248/` is fine).
4. For each ref: narrow-checkout
   `tensorrt_llm/_torch/pyexecutor/{sampler,sampling_utils}.py` at that
   SHA, run 5×`trtllm-bench`, stamp `.done`. (Branch HEAD never moves.)
5. Trap restores both files from the original branch HEAD on any exit
   path (success, error, ctrl-C).
6. Aggregates three pairwise comparisons:
   - `plus_handoff` vs `baseline_piecewise` — incremental 1
   - `plus_handoff_beamhist` vs `plus_handoff` — incremental 2
   - `plus_handoff_beamhist` vs `baseline_piecewise` — cumulative

Each phase ≈ 3-4 minutes wall on an L40S; total ≈ 12-15 minutes.

### Step 2 — TRT baseline + cross-backend comparison

```bash
bash nvbugs_5615248/trtllm_bench/run_trt_baseline_compare.sh
```

The driver:

1. Pre-flights everything plus the engine dir at
   `nvbugs_5615248/tinyllama_trt_engine`. Refuses to start unless the
   three PyT ref dirs from Step 1 each have a `.done` marker.
2. Runs 5× `trtllm-bench --backend tensorrt` into
   `three_ref_validation/baseline_trt/`. (No narrow-checkout: TRT path
   uses the C++ runtime + prebuilt engine, so toggling Python sampler.py
   is irrelevant.)
3. For each PyT ref, builds an ephemeral symlink dir under
   `three_ref_validation/.cross_backend_shim/` that renames
   `request_pytorch*.json` → `request_trt*.json`, and aggregates against
   the TRT dir using `--backend trt` so a single backend tag drives the
   pairing logic in `aggregate_runs.py`. Tears the shim dir down on exit.

TRT phase ≈ 3-4 minutes; aggregations are instant. Total ≈ 4-5 minutes.

### Useful overrides

| Var | Default | Use |
|---|---|---|
| `NUM_RUNS` | `5` | Lower for a smoke check; raise to tighten p-values |
| `OUT_ROOT` | `nvbugs_5615248/trtllm_bench/three_ref_validation` | Send outputs to scratch |
| `MODEL` | TinyLlama path | Override model path |
| `ENGINE_DIR` | `nvbugs_5615248/tinyllama_trt_engine` | Override TRT engine |
| `BASELINE_REF` / `PLUS_V1_REF` / `PLUS_V2_REF` | `HEAD~2` / `HEAD~1` / `HEAD` | Override PyT SHAs |
| `SKIP_BASELINE=1` / `SKIP_PLUS_V1=1` / `SKIP_PLUS_V2=1` | unset | Skip a phase already done |
| `SKIP_TRT=1` | unset | Re-aggregate against existing TRT dir without re-running it |

### Standalone aggregator invocations

For ad-hoc pairwise comparisons (e.g., re-aggregating a different pair of
already-completed dirs):

```bash
python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \
    --backend pytorch \
    --baseline   nvbugs_5615248/trtllm_bench/three_ref_validation/baseline_piecewise \
    --baseline-label baseline_piecewise \
    --experiment nvbugs_5615248/trtllm_bench/three_ref_validation/plus_handoff_beamhist \
    --experiment-label plus_handoff_beamhist
```

For PyT-vs-TRT, pass `--backend trt` and use a symlink-renamed copy of the
PyT dir as the experiment side (or just call `run_trt_baseline_compare.sh`
with `SKIP_TRT=1`).

## Reference results (this branch, validated 2026-05-09)

Single L40S, warm node. All numbers below are pooled per-request medians
(n = 5 runs × 16 requests = 80 samples per ref). The full distribution
plus per-run-medians, bootstrap CI, Mann-Whitney p, Hodges-Lehmann shift,
and Welch's t are in the captured run logs.

### PyT incrementals — Δmedian vs the immediately-prior ref

| Step | Comparison | TTFT (ms) | E2E (ms) | ITL (ms) |
|---|---|---:|---:|---:|
| **#1** | `plus_handoff` vs `baseline_piecewise` | **−0.328 (−2.97%)** | **−0.309 (−0.37%)** | −0.0001 (−0.02%) |
| **#2** | `plus_handoff_beamhist` vs `plus_handoff` | −0.103 (−0.96%) | **−0.547 (−0.65%)** | **−0.0021 (−0.57%)** |
| **Σ** | `plus_handoff_beamhist` vs `baseline_piecewise` | **−0.431 (−3.90%)** | **−0.855 (−1.02%)** | **−0.0021 (−0.58%)** |

(**bold** = Mann-Whitney p < 1e−3 *and* 95% bootstrap CI excludes 0.)

Per-fix attribution lines up with intent:

- **`0115bd2430` "Reduce handoff cost"** — TTFT/prefill fix.
  - TTFT: −0.328 ms (−2.97%), p = 1.2e−9.
  - E2E: −0.309 ms ≈ pure TTFT spillover.
  - ITL: 0 (CI [−0.0004, +0.0004], p = 0.55) — does not touch the
    decode loop.
- **`00e8b46370` "Beam history terminal-step copies"** — decode-loop fix.
  - ITL: −0.0021 ms/step (−0.57%), p = 6e−18.
  - E2E: −0.547 ms — larger than the per-step prediction (19 × 0.0021 ≈
    0.04 ms) because deferring D2H to the terminal step also collapses
    one big stall on top of the per-step savings.
  - TTFT: −0.103 ms — small but real (CI [−0.241, −0.032]); secondary
    spillover from removing per-step beam-history launches.

**Additivity check** (the increments stack with no observed interaction):

| Metric | Δ#1 | + Δ#2 | predicted Σ | observed Σ | residual |
|---|---:|---:|---:|---:|---:|
| TTFT | −0.328 | −0.103 | −0.431 | −0.431 | **0.000** |
| E2E | −0.309 | −0.547 | −0.856 | −0.855 | +0.001 |
| ITL | −0.0001 | −0.0021 | −0.0022 | −0.0021 | +0.0001 |

### Cross-backend — Δmedian vs TRT (TRT as baseline)

TRT pooled medians: TTFT = 5.856 ms, E2E = 80.400 ms, ITL = 0.374 ms.

| PyT ref | TTFT | Δ vs TRT | E2E | Δ vs TRT | ITL | Δ vs TRT |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_piecewise` | 11.060 | **+5.204 (+88.9%)** | 83.970 | +3.570 (+4.44%) | 0.366 | **−0.008 (−2.09%)** |
| `plus_handoff` | 10.731 | +4.876 (+83.3%) | 83.662 | +3.262 (+4.06%) | 0.366 | −0.008 (−2.11%) |
| `plus_handoff_beamhist` | 10.628 | **+4.773 (+81.5%)** | 83.115 | **+2.715 (+3.38%)** | 0.364 | **−0.010 (−2.66%)** |

(Δ > 0 ⇒ PyT slower than TRT; Δ < 0 ⇒ PyT faster than TRT. All
Mann-Whitney p ≤ 9.2e−28 — the pooled distributions don't overlap.)

**Gap closure** — what the two fixes recover against the TRT yardstick:

| metric | PyT baseline gap | After both fixes | Total gap closed | % of original gap closed |
|---|---:|---:|---:|---:|
| TTFT | +5.204 ms (+88.9%) | +4.773 ms (+81.5%) | 0.431 ms / 7.4 pp | **8.3%** |
| E2E | +3.570 ms (+4.44%) | +2.715 ms (+3.38%) | 0.855 ms / 1.06 pp | **23.9%** |
| ITL | −0.008 ms (PyT-favored) | −0.010 ms (PyT-favored) | +0.002 ms (PyT lead grows) | n/a |

### Three-line takeaway

1. **TRT still wins TTFT by ~2×** (residual +4.77 ms / +81.5% after both
   fixes). The fixes close 0.43 ms / ~8% of that gap — the bulk of the
   residual is *not* beam-search host overhead, it's prefill /
   scheduling / chunked-prefill overhead that this branch doesn't touch.
2. **TRT still wins E2E by ~3-4%**, but the fixes close 24% of that gap
   (+4.44% → +3.38%).
3. **PyT beats TRT on per-token ITL** (0.364 vs 0.374 ms/step, −2.66%
   after both fixes). Piecewise-CUDA-graph capture genuinely
   out-performs TRT's `cuda_graph_mode` once you're past prefill, and
   the beam-history fix widens the lead. So with longer OSL the
   PyT-vs-TRT comparison would tip in PyT's favor on E2E too.

## Pre-flight guards (recap)

`run_3ref_bench.sh` and `run_trt_baseline_compare.sh` both:

- Refuse to run with a dirty tracked working tree.
- Resolve refs to immutable SHAs before the first checkout.
- Stamp `.done` markers per phase so re-runs auto-skip.
- (3-ref only) Restore the toggled Python files from the original branch
  HEAD on any exit path via an `EXIT` trap.
- (TRT-compare only) Refuse to start unless all three PyT ref dirs from
  the 3-ref run have `.done` markers.

This means a Ctrl-C during any phase always leaves the working tree on
the original branch HEAD with the original files intact — re-running
just resumes where it stopped.
