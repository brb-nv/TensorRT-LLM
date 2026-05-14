
### feature_on_5f55cce559 — pytorch backend
(5 runs × 16 requests)

Per-run statistics (one mean per file, then aggregated across runs):

| label | metric | median (ms) | mean (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|
| feature_on_5f55cce559 (pytorch, n=5) | TTFT |   10.538 |   10.539 |    0.069 |   10.443 |   10.632 |
| feature_on_5f55cce559 (pytorch, n=5) | E2E |   82.321 |   82.350 |    0.089 |   82.237 |   82.469 |
| feature_on_5f55cce559 (pytorch, n=5) | ITL |    0.361 |    0.361 |    0.000 |    0.360 |    0.361 |

Pooled per-request:

| label | metric | median (ms) | mean (ms) | p10 (ms) | p90 (ms) | stdev (ms) | min (ms) | max (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| feature_on_5f55cce559 (pytorch, n=80) | TTFT |   10.491 |   10.539 |   10.185 |   10.997 |    0.289 |   10.082 |   11.269 |
| feature_on_5f55cce559 (pytorch, n=80) | E2E |   82.257 |   82.350 |   82.064 |   82.742 |    0.297 |   81.950 |   83.232 |
| feature_on_5f55cce559 (pytorch, n=80) | ITL |    0.361 |    0.361 |    0.359 |    0.362 |    0.001 |    0.357 |    0.367 |

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

## feature_on_5f55cce559 vs baseline_f03cb1ce6b  (backend=pytorch)

### Pooled per-request — robust (n=80 vs n=80)

| metric | base median | exp median | Δmedian (ms) | Δ (%) | 95% CI (bootstrap) | Mann-Whitney p | Hodges-Lehmann (ms) |
|---|---:|---:|---:|---:|---|---:|---:|
| TTFT |   10.583 |   10.491 |  -0.0924 |  -0.873% | [-0.2301, -0.0035] |   0.0192 |  -0.0980 |
| E2E |   82.743 |   82.257 |  -0.4862 |  -0.588% | [-0.5694, -0.3784] | 4.31e-15 |  -0.4279 |
| ITL |    0.362 |    0.361 |  -0.0014 |  -0.398% | [-0.0019, -0.0011] | 1.61e-15 |  -0.0016 |

### Per-run medians  (n=5 vs n=5)

| metric | base median | exp median | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| TTFT |   10.553 |   10.538 |  -0.0149 |  -0.141% |
| E2E |   82.767 |   82.321 |  -0.4456 |  -0.538% |
| ITL |    0.362 |    0.361 |  -0.0015 |  -0.422% |

### Pooled per-request — mean / Welch's t  (informational; sensitive to outliers)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.639 |   10.539 |  -0.1003 |  -0.943% |  +2.13 |   0.0345 |
| E2E |   82.773 |   82.350 |  -0.4223 |  -0.510% |  +8.75 | 3.16e-15 |
| ITL |    0.362 |    0.361 |  -0.0016 |  -0.446% |  +7.63 | 2.46e-12 |

### Per-run means — mean / Welch's t  (informational)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.639 |   10.539 |  -0.1003 |  -0.943% |  +1.10 |    0.323 |
| E2E |   82.773 |   82.350 |  -0.4223 |  -0.510% |  +5.16 |  0.00182 |
| ITL |    0.362 |    0.361 |  -0.0016 |  -0.446% |  +5.81 | 0.000402 |
