<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Repro: TRT (baseline) vs PyTorch + two feature flags (NVBug 5615248)

This branch ships a self-contained set of `*_repro.*` artifacts so you can
reproduce the TRT-vs-PyTorch latency comparison we observed on TinyLlama
beam-10. All `_repro` files live next to this README under
`nvbugs_5615248/trtllm_bench/`.

## What is being compared

Two configurations on the same TinyLlama-1.1B beam-10 workload (16 requests,
concurrency 1, streaming, ISL=100, OSL=20):

| Leg            | Backend  | Notes                                                                                             |
|----------------|----------|---------------------------------------------------------------------------------------------------|
| `baseline_trt` | TRT      | Prebuilt TinyLlama engine at `nvbugs_5615248/tinyllama_trt_engine`; runtime config `trt_repro.yaml`. |
| `feature_pyt`  | PyTorch  | Config `pytorch_repro.yaml` with both opt-ins ON: `enable_early_first_token_response=true` and `enable_speculative_beam_history_d2h=true`. |

Sign convention used by the aggregated report:

```
Δmedian = median(feature_pyt) - median(baseline_trt)
  > 0 -> PyT slower than TRT (residual gap)
  < 0 -> PyT faster than TRT (feature win)
```

## Files in this directory

| File                                  | Role                                                                                                   |
|---------------------------------------|--------------------------------------------------------------------------------------------------------|
| `pytorch_repro.yaml`                  | PyTorch backend YAML; pins engine geometry (`max_seq_len=129`, beam=10) + the two feature flags.       |
| `trt_repro.yaml`                      | TRT runtime YAML (geometry baked into the engine).                                                     |
| `dataset_isl100_osl20_repro.jsonl`    | 32 tokenized prompts, ISL=100 / OSL=20, tokenizer = TinyLlama-1.1B-Chat-v1.0.                          |
| `aggregate_runs_repro.py`             | Median + bootstrap 95% CI + Mann-Whitney + Hodges-Lehmann + Welch's t aggregator.                      |
| `build_trt_engine_repro.sh`           | One-shot HF -> TRT-LLM checkpoint -> TRT engine builder for the baseline leg.                          |
| `run_multirun_trt_repro.sh`           | Single-leg TRT launcher (5 runs by default).                                                           |
| `run_multirun_pytorch_repro.sh`       | Single-leg PyTorch launcher (5 runs by default).                                                       |
| `run_trt_vs_pyt_compare_repro.sh`     | Top-level driver: runs both legs and aggregates into one Markdown report.                              |
| `repro.md`                            | This file.                                                                                             |

## Prerequisites

* NVIDIA GPU with the same SM family used for your reference numbers
  (the headline numbers in **Expected numbers** below were collected on
  an L40S; behavior on other archs is qualitatively similar but the
  absolute deltas will shift).
* The TensorRT-LLM **release container** matching this branch (or a
  development container built from this branch's source). `trtllm-bench`
  and `trtllm-build` must be on `PATH`.
* A **local directory** containing the TinyLlama-1.1B-Chat-v1.0 HuggingFace
  model (config.json + safetensors). HF repo ids are NOT accepted by
  `trtllm-bench` -- click parses `--model_path` as `pathlib.Path`, which
  HF Hub's validator rejects with `HFValidationError: Repo id must be a
  string, not <class 'pathlib.PosixPath'>`. To get a local copy:

  ```bash
  huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --local-dir $PWD/nvbugs_5615248/tinyllama_hf
  export MODEL=$PWD/nvbugs_5615248/tinyllama_hf
  ```

  (`MODEL` is also read by `build_trt_engine_repro.sh` via `HF_MODEL`,
  so the same local dir works for both steps; see Step 2.)

## Step 1 - Check out this branch

```bash
git fetch origin user/brb/trt-vs-pyt-repro-nvbug-5615248
git checkout user/brb/trt-vs-pyt-repro-nvbug-5615248
```

All scripts below assume you run them from the repo root.

## Step 2 - Build the TensorRT engine for the TRT leg

The prebuilt engine is too large to commit, so re-build it locally with
the geometry that matches `pytorch_repro.yaml`:

```bash
# Recommended: re-use the same local HF dir for both the build and the bench
HF_MODEL=$MODEL bash nvbugs_5615248/trtllm_bench/build_trt_engine_repro.sh
```

`HF_MODEL` accepts either an HF repo id (triggers an HF download into
`~/.cache/huggingface/`) or a local directory. Using a local dir is
strongly recommended because the bench step (Step 3) requires one anyway.

This converts to a TRT-LLM checkpoint under
`nvbugs_5615248/tinyllama_trt_ckpt/`, then builds the engine under
`nvbugs_5615248/tinyllama_trt_engine/`. Both intermediate dirs are
skipped on re-runs if the relevant output files exist.

## Step 3 - Run the comparison

```bash
bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh
```

This produces a fresh, timestamped output dir under
`nvbugs_5615248/trtllm_bench/trt_vs_pyt_repro_<ts>/` containing:

```
trt_vs_pyt_repro_<ts>/
  driver.log
  baseline_trt/
    env.txt
    report_trt.json,    report_trt{2..5}.json
    request_trt.json,   request_trt{2..5}.json
    output_trt.json,    output_trt{2..5}.json
    run_trt.log,        run_trt{2..5}.log
    .done
    .engine_dir
  feature_pyt/
    env.txt
    report_pytorch.json,  ...
    request_pytorch.json, ...
    output_pytorch.json,  ...
    run_pytorch.log,      ...
    .done
  cmp_feature_pyt_vs_baseline_trt.md   <-- headline report
```

Useful overrides:

```bash
# Higher-confidence run (more samples, same protocol)
NUM_RUNS=10 bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh

# Pin a specific output dir (handy for repeated re-aggregation)
bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh \
    nvbugs_5615248/trtllm_bench/trt_vs_pyt_repro_pinned

# Re-aggregate only (skip both legs; their .done markers must exist)
SKIP_TRT=1 SKIP_PYT=1 \
    bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh \
        nvbugs_5615248/trtllm_bench/trt_vs_pyt_repro_pinned

# Override MODEL (must be a local HF dir; see Prerequisites)
MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0 \
    bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh
```

## Expected numbers (reference)

Reference numbers collected on this branch (HEAD = `f278c4f170`) on a
single NVIDIA L40S, inside the matching TRT-LLM container, with the
default protocol (`NUM_RUNS=5`, 16 requests per run, `--concurrency 1`,
`--streaming`, beam=10, ISL=100, OSL=20; n=80 pooled per leg).

Sign convention: **Δ = feature_pyt - baseline_trt** (negative ⇒ PyT
faster than TRT).

### Pooled per-request - robust (n=80 vs n=80)

| metric | TRT median | PyT median | Δmedian    | Δ %      | 95 % CI (bootstrap)  | Mann-Whitney p | Hodges-Lehmann |
|--------|------------|------------|------------|----------|----------------------|----------------|----------------|
| TTFT   | 6.463 ms   | 8.711 ms   | +2.249 ms  | +34.80 % | [+2.127, +2.401]     | 1.36e-15       | +2.240 ms      |
| E2E    | 81.021 ms  | 82.544 ms  | +1.522 ms  | +1.88 %  | [+1.390, +1.677]     | 2.16e-14       | +1.492 ms      |
| ITL    | 0.375 ms   | 0.371 ms   | -0.0037 ms | -0.99 %  | [-0.0039, -0.0035]   | 2.32e-26       | -0.0037 ms     |

What this says on this workload:

* **TRT wins on TTFT by ~2.2 ms (+34.8 %)**: even with
  `enable_early_first_token_response=true`, the PyTorch backend does not
  close the prefill-side gap to TRT on beam=10 TinyLlama.
* **TRT wins on E2E by ~1.5 ms (+1.88 %)**: the per-request gap is real
  and tight (95 % CI [+1.39, +1.68] ms, p = 2.2e-14).
* **PyT wins on ITL by ~3.7 us / token (-0.99 %)**:
  `enable_speculative_beam_history_d2h=true` does what it claims;
  per-token cost drops with very high confidence (p = 2.3e-26). At
  OSL=20 that's only ~70 us / request, so it does not flip E2E.

Expect deviations of roughly ±10-20 % on a different host or GPU model;
the **sign and ordering** of all three metrics should be stable
(`TTFT` and `E2E` favor TRT; `ITL` favors PyT) on this workload.

If your numbers do not look like the table above, sanity-check:

1. You ran inside the matching TRT-LLM container.
2. `pytorch_repro.yaml` still has both feature flags set to `true`
   (`grep enable_ nvbugs_5615248/trtllm_bench/pytorch_repro.yaml`).
3. The TRT engine geometry matches `max_seq_len=129`, `max_batch_size=1`,
   `max_beam_width=10`
   (`python3 -c "import json; d=json.load(open('nvbugs_5615248/tinyllama_trt_engine/config.json'))['build_config']; print(d['max_seq_len'], d['max_batch_size'], d['max_beam_width'])"`).
4. Nothing else is using the GPU (`nvidia-smi`).
5. The per-run statistics for `baseline_trt` are tight (per-run TTFT
   stdev < ~2 ms). A noisy run (e.g. stdev ~4-5 ms with a single ~17+ ms
   TTFT median) is usually a transient co-tenant or thermal-throttle
   event; re-run that leg with `rm -rf <out>/baseline_trt && SKIP_PYT=1
   bash run_trt_vs_pyt_compare_repro.sh <out>`.

## Step 4 - Read the report

Open `cmp_feature_pyt_vs_baseline_trt.md` in the output dir. The headline
table to look at is **"Pooled per-request - robust"**, which reports
Δmedian, a bootstrap 95% CI on the difference of medians, the
Mann-Whitney U two-sided p-value, and the Hodges-Lehmann shift estimator
for each of `TTFT`, `E2E`, and `ITL` (all in ms).

Reminder on the sign:

* **Δmedian < 0**: `feature_pyt` is faster than `baseline_trt` on that
  metric -> the PyTorch path with both flags ON beats TRT.
* **Δmedian > 0**: `feature_pyt` is slower than `baseline_trt` -> there's
  still a residual gap.
* **Mann-Whitney p**: a small value (e.g. < 0.05) means the shift is
  unlikely to be noise; a large value means the two distributions overlap.

The supplementary "Per-run medians" and "mean / Welch's t" tables are
intentionally less robust (mean-based, n=5); use them only as cross-checks.

## Manual re-aggregation

If you tweak `aggregate_runs_repro.py` or want to re-aggregate existing
artifacts without re-running the legs:

```bash
python3 nvbugs_5615248/trtllm_bench/aggregate_runs_repro.py \
    --backend trt \
    --baseline   nvbugs_5615248/trtllm_bench/trt_vs_pyt_repro_<ts>/baseline_trt \
    --baseline-label baseline_trt \
    --experiment nvbugs_5615248/trtllm_bench/trt_vs_pyt_repro_<ts>/feature_pyt \
    --experiment-label feature_pyt
```

Note: the aggregator auto-detects the backend tag from filenames, so a
direct cross-backend run would fail (`request_trt*.json` vs
`request_pytorch*.json`). The driver works around this by symlinking the
PyT artifacts into `request_trt*.json` names inside an ephemeral shim
dir, then aggregating with `--backend trt`. If you re-aggregate manually,
replicate that symlink trick or modify the aggregator to take an
explicit per-side `--backend`.

## What the feature flags do

* `enable_early_first_token_response` (PyTorch only): under the overlap
  scheduler, the first-token response is emitted ahead of the next step,
  cutting TTFT by roughly one decode iteration on streaming workloads.
* `enable_speculative_beam_history_d2h` (PyTorch only, beam search):
  skips per-step beam-history D2H copies on likely-non-terminal steps
  (the full copy happens on terminal steps), trimming per-token ITL on
  beam-search workloads. Incompatible with
  `sampler_force_async_worker=True`.

Both fields live on `TorchLlmArgs` (see `tensorrt_llm/llmapi/llm_args.py`),
so they apply only to the PyTorch backend. The TRT leg is unaffected.

## Troubleshooting

* `ERROR: MODEL='...' is not a local directory` -- you set (or defaulted)
  `MODEL` to an HF repo id or a non-existent path. See **Prerequisites**
  for the `huggingface-cli download` recipe.
* `HFValidationError: Repo id must be a string, not <class 'pathlib.PosixPath'>`
  -- same root cause: `MODEL` was an HF repo id. The script's pre-flight
  check normally catches this before `trtllm-bench` runs; if you see it
  raw, you may have bypassed the launcher and called `trtllm-bench`
  directly.
* `ERROR: TRT engine dir not found at ...` -- run Step 2 first.
* `ERROR: trtllm-bench not on PATH` -- you're outside the TRT-LLM
  container; enter the container (or activate the env) and re-run.
* Re-running with the same output dir reuses any leg whose `.done` marker
  is present. Delete the leg's directory (or its `.done` file) to force a
  re-run of just that leg.
