# Beam-history speculative D2H benchmark summary

Run dir: `nvbugs_5615248/trtllm_bench/beam_d2h_compare_20260514_050556/`.
Workload: TinyLlama-1.1B, beam=10, ISL=100/OSL=20, concurrency=1, streaming,
5 runs x 16 requests = 80 pooled per-request samples per leg.

Three configurations:

| Leg          | Commit                                       | Env var setting                                    |
|--------------|----------------------------------------------|----------------------------------------------------|
| baseline     | `f03cb1ce6b327171be8a0ed9ceed64a078294aff`   | n/a (env var did not exist yet)                    |
| feature_off  | `5f55cce559518f0e43da53f9d01a29a4fed5fcc8`   | `TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H` unset  |
| feature_on   | `5f55cce559518f0e43da53f9d01a29a4fed5fcc8`   | `TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=1`      |

## 1. `feature_off vs baseline` (regression sanity check)

PASS. The refactor + gating, with the env var off, behaves like pre-PR code.

| metric | base median | exp median | Delta median | Delta (%) | Mann-Whitney p |
|--------|------------:|-----------:|-------------:|----------:|---------------:|
| TTFT   | 10.583 ms   | 10.550 ms  | -0.033 ms    | -0.31%    | 0.986          |
| E2E    | 82.743 ms   | 82.625 ms  | -0.118 ms    | -0.14%    | 0.108          |
| ITL    | 0.362 ms    | 0.362 ms   | -0.0002 ms   | -0.05%    | 0.0952         |

All three deltas are within run-to-run noise; Mann-Whitney p-values are
non-significant and the 95% bootstrap CIs straddle (or hug) zero. The
default code path is preserved.

## 2. `feature_on vs baseline` (headline, pre-PR comparison)

WIN. The speculative path is faster than the pre-PR baseline on every
metric, with strong significance on E2E and ITL.

| metric | base median | exp median | Delta median | Delta (%) | 95% CI             | Mann-Whitney p |
|--------|------------:|-----------:|-------------:|----------:|--------------------|---------------:|
| TTFT   | 10.583 ms   | 10.491 ms  | -0.092 ms    | -0.87%    | [-0.230, -0.004]   | 0.0192         |
| E2E    | 82.743 ms   | 82.257 ms  | -0.486 ms    | -0.59%    | [-0.569, -0.378]   | 4.31e-15       |
| ITL    | 0.362 ms    | 0.361 ms   | -0.0014 ms   | -0.40%    | [-0.0019, -0.0011] | 1.61e-15       |

E2E moves down ~0.49 ms per request (median); ITL moves down ~1.4 us per
emitted token. Hodges-Lehmann shifts agree (-0.43 ms E2E, -0.0016 ms ITL).

## 3. `feature_on vs feature_off` (headline, same SHA, isolates env-var effect)

WIN. Confirms the gain comes from the speculative D2H path itself, not
from anything else in the refactor.

| metric | base median | exp median | Delta median | Delta (%) | 95% CI             | Mann-Whitney p |
|--------|------------:|-----------:|-------------:|----------:|--------------------|---------------:|
| TTFT   | 10.550 ms   | 10.491 ms  | -0.060 ms    | -0.56%    | [-0.173, +0.003]   | 0.047          |
| E2E    | 82.625 ms   | 82.257 ms  | -0.369 ms    | -0.45%    | [-0.438, -0.272]   | 2.7e-12        |
| ITL    | 0.362 ms    | 0.361 ms   | -0.0013 ms   | -0.35%    | [-0.0018, -0.0008] | 9.35e-11       |

The E2E and ITL deltas are nearly identical to the `feature_on vs
baseline` deltas, which is the expected pattern: feature_off is
indistinguishable from baseline, and feature_on lifts by the same amount
above either reference.

## Takeaways

* The opt-in path is doing what it should: off behaves like pre-PR; on is
  consistently faster.
* TinyLlama steady-state ITL drops ~0.35-0.40% (~1.3-1.4 us/token); E2E
  drops ~0.45-0.59% (~0.37-0.49 ms/request).
* TTFT moves marginally; it is prefill-dominated, while beam-history D2H
  is a decode-step cost.
* ITL is the cleanest signal: its stdev is ~1 us while the shift is ~1.4
  us, hence p-values in the 1e-11 / 1e-15 range.
* The gating correctly preserves the default code path: `feature_off vs
  baseline` shows no significant difference on any metric.

See the per-pair reports in this directory for full robust + informational
statistics (medians, bootstrap CI, Mann-Whitney, Hodges-Lehmann, Welch's t):

* `cmp_feature_off_vs_baseline.md`
* `cmp_feature_on_vs_baseline.md`
* `cmp_feature_on_vs_feature_off.md`
