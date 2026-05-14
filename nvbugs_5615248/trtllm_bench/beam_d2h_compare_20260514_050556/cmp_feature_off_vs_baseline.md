
### feature_off_5f55cce559 — pytorch backend
(5 runs × 16 requests)

Per-run statistics (one mean per file, then aggregated across runs):

| label | metric | median (ms) | mean (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|
| feature_off_5f55cce559 (pytorch, n=5) | TTFT |   10.681 |   10.731 |    0.285 |   10.525 |   11.222 |
| feature_off_5f55cce559 (pytorch, n=5) | E2E |   82.727 |   82.788 |    0.292 |   82.514 |   83.254 |
| feature_off_5f55cce559 (pytorch, n=5) | ITL |    0.362 |    0.362 |    0.000 |    0.362 |    0.363 |

Pooled per-request:

| label | metric | median (ms) | mean (ms) | p10 (ms) | p90 (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| feature_off_5f55cce559 (pytorch, n=80) | TTFT |   10.550 |   10.731 |   10.290 |   11.518 |    0.523 |    9.944 |   12.418 |
| feature_off_5f55cce559 (pytorch, n=80) | E2E |   82.625 |   82.788 |   82.364 |   83.621 |    0.499 |   82.138 |   84.539 |
| feature_off_5f55cce559 (pytorch, n=80) | ITL |    0.362 |    0.362 |    0.361 |    0.363 |    0.001 |    0.356 |    0.367 |

### baseline_f03cb1ce6b — pytorch backend
(5 runs × 16 requests)

Per-run statistics (one mean per file, then aggregated across runs):

| label | metric | median (ms) | mean (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|
| baseline_f03cb1ce6b (pytorch, n=5) | TTFT |   10.553 |   10.639 |    0.193 |   10.496 |   10.966 |
| baseline_f03cb1ce6b (pytorch, n=5) | E2E |   82.767 |   82.773 |    0.160 |   82.618 |   83.010 |
| baseline_f03cb1ce6b (pytorch, n=5) | ITL |    0.362 |    0.362 |    0.000 |    0.362 |    0.363 |

Pooled per-request:

| label | metric | median (ms) | mean (ms) | p10 (ms) | p90 (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_f03cb1ce6b (pytorch, n=80) | TTFT |   10.583 |   10.639 |   10.266 |   11.018 |    0.306 |   10.156 |   11.878 |
| baseline_f03cb1ce6b (pytorch, n=80) | E2E |   82.743 |   82.773 |   82.432 |   83.050 |    0.314 |   82.354 |   84.226 |
| baseline_f03cb1ce6b (pytorch, n=80) | ITL |    0.362 |    0.362 |    0.361 |    0.364 |    0.001 |    0.360 |    0.368 |

## feature_off_5f55cce559 vs baseline_f03cb1ce6b  (backend=pytorch)

### Pooled per-request — robust (n=80 vs n=80)

| metric | base median | exp median | Δmedian (ms) | Δ (%) | 95% CI (bootstrap) | Mann-Whitney p | Hodges-Lehmann (ms) |
|---|---:|---:|---:|---:|---|---:|---:|
| TTFT |   10.583 |   10.550 |  -0.0329 |  -0.311% | [-0.1426, +0.0775] |    0.986 |  +0.0014 |
| E2E |   82.743 |   82.625 |  -0.1177 |  -0.142% | [-0.2036, -0.0231] |    0.108 |  -0.0749 |
| ITL |    0.362 |    0.362 |  -0.0002 |  -0.051% | [-0.0007, +0.0002] |   0.0952 |  -0.0003 |

### Per-run medians  (n=5 vs n=5)

| metric | base median | exp median | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| TTFT |   10.553 |   10.681 |  +0.1282 |  +1.215% |
| E2E |   82.767 |   82.727 |  -0.0394 |  -0.048% |
| ITL |    0.362 |    0.362 |  -0.0003 |  -0.089% |

### Pooled per-request — mean / Welch's t  (informational; sensitive to outliers)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.639 |   10.731 |  +0.0916 |  +0.861% |  -1.35 |    0.178 |
| E2E |   82.773 |   82.788 |  +0.0150 |  +0.018% |  -0.23 |     0.82 |
| ITL |    0.362 |    0.362 |  -0.0004 |  -0.106% |  +1.83 |   0.0686 |

### Per-run means — mean / Welch's t  (informational)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.639 |   10.731 |  +0.0916 |  +0.861% |  -0.60 |     0.57 |
| E2E |   82.773 |   82.788 |  +0.0150 |  +0.018% |  -0.10 |    0.923 |
| ITL |    0.362 |    0.362 |  -0.0004 |  -0.106% |  +1.49 |    0.175 |
