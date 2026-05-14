# Benchmark: beam-history speculative D2H opt-in

Bench infra for the beam-history speculative-D2H opt-in introduced on this
branch. Three configurations are compared on the existing TinyLlama
beam-10 workload (16 requests, concurrency 1, streaming, ISL=100, OSL=20):

| Leg              | Commit                                       | Env var setting                                      |
|------------------|----------------------------------------------|------------------------------------------------------|
| baseline         | `f03cb1ce6b327171be8a0ed9ceed64a078294aff`   | n/a (env var did not exist yet)                      |
| feature_off      | `5f55cce559518f0e43da53f9d01a29a4fed5fcc8`   | `TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H` unset    |
| feature_on       | `5f55cce559518f0e43da53f9d01a29a4fed5fcc8`   | `TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=1`        |

Comparisons:

* `feature_off vs baseline` is a regression sanity check; with the env var
  off, the new code path should behave like the pre-PR code.
* `feature_on vs baseline` and `feature_on vs feature_off` are the headline
  numbers for the speculative path.

## Files added by this PR's bench infra

| File                              | Role                                                                   |
|-----------------------------------|------------------------------------------------------------------------|
| `run_multirun_beam_d2h_ab.sh`     | A/B launcher; toggles `TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H`.     |
| `run_beam_d2h_compare.sh`         | Driver that swaps commits, runs all three legs, and aggregates.        |
| `README_BEAM_D2H.md`              | This file.                                                             |

Existing infra reused as-is (local scratch under `nvbugs_5615248/trtllm_bench/`,
intentionally not committed to the repo since it predates this PR):

* `run_multirun_pytorch.sh` (used for the baseline leg).
* `pytorch.yaml` (TinyLlama, beam=10, max_seq_len=129, piecewise CUDA graph).
* `dataset_isl100_osl20.jsonl`.
* `aggregate_runs.py` (median + bootstrap 95% CI + Mann-Whitney + Welch).

## Running

Inside the TRT-LLM container, from the repo root:

```bash
bash nvbugs_5615248/trtllm_bench/run_beam_d2h_compare.sh
```

This writes per-leg artifacts and three Markdown comparison reports under
`nvbugs_5615248/trtllm_bench/beam_d2h_compare_<timestamp>/`.

Optional overrides:

```bash
NUM_RUNS=10 \
BASELINE_SHA=<sha> FEATURE_SHA=<sha> \
bash nvbugs_5615248/trtllm_bench/run_beam_d2h_compare.sh \
    nvbugs_5615248/trtllm_bench/beam_d2h_compare_custom
```

## Pre-flight

The driver refuses to run if:

* `trtllm-bench` is not on `PATH` (i.e., outside the container).
* The working tree has modified tracked files (untracked files are fine).
* Either SHA is unreachable in the local repo.
* Any required input under `nvbugs_5615248/trtllm_bench/` is missing.

The original branch ref is restored on exit (success or failure) via an
EXIT trap.

## Output layout

```
beam_d2h_compare_<ts>/
  driver.log
  baseline_f03cb1ce6b/
    env.txt
    report_pytorch.json,  report_pytorch{2..5}.json
    request_pytorch.json, request_pytorch{2..5}.json
    output_pytorch.json,  output_pytorch{2..5}.json
    run_pytorch.log,      run_pytorch{2..5}.log
  feature_off_5f55cce559/
    ...
  feature_on_5f55cce559/
    ...
  cmp_feature_off_vs_baseline.md     # regression sanity
  cmp_feature_on_vs_baseline.md      # headline
  cmp_feature_on_vs_feature_off.md   # headline (same SHA)
```

## Manual aggregation

To re-run the aggregator on existing artifacts (e.g., after editing
`aggregate_runs.py`):

```bash
python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \
    --baseline   nvbugs_5615248/trtllm_bench/beam_d2h_compare_<ts>/baseline_f03cb1ce6b   --baseline-label   baseline \
    --experiment nvbugs_5615248/trtllm_bench/beam_d2h_compare_<ts>/feature_on_5f55cce559 --experiment-label feature_on
```
