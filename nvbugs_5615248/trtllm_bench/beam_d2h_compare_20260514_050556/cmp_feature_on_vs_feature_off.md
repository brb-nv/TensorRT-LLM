
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

## feature_on_5f55cce559 vs feature_off_5f55cce559  (backend=pytorch)

### Pooled per-request — robust (n=80 vs n=80)

| metric | base median | exp median | Δmedian (ms) | Δ (%) | 95% CI (bootstrap) | Mann-Whitney p | Hodges-Lehmann (ms) |
|---|---:|---:|---:|---:|---|---:|---:|
| TTFT |   10.550 |   10.491 |  -0.0595 |  -0.564% | [-0.1727, +0.0027] |    0.047 |  -0.0987 |
| E2E |   82.625 |   82.257 |  -0.3685 |  -0.446% | [-0.4377, -0.2723] |  2.7e-12 |  -0.3603 |
| ITL |    0.362 |    0.361 |  -0.0013 |  -0.347% | [-0.0018, -0.0008] | 9.35e-11 |  -0.0013 |

### Per-run medians  (n=5 vs n=5)

| metric | base median | exp median | Δ (ms) | Δ (%) |
|---|---:|---:|---:|---:|
| TTFT |   10.681 |   10.538 |  -0.1431 |  -1.340% |
| E2E |   82.727 |   82.321 |  -0.4062 |  -0.491% |
| ITL |    0.362 |    0.361 |  -0.0012 |  -0.333% |

### Pooled per-request — mean / Welch's t  (informational; sensitive to outliers)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.731 |   10.539 |  -0.1919 |  -1.789% |  +2.88 |  0.00476 |
| E2E |   82.788 |   82.350 |  -0.4373 |  -0.528% |  +6.74 |  4.9e-10 |
| ITL |    0.362 |    0.361 |  -0.0012 |  -0.341% |  +5.34 | 3.23e-07 |

### Per-run means — mean / Welch's t  (informational)

| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |   10.731 |   10.539 |  -0.1919 |  -1.789% |  +1.46 |     0.21 |
| E2E |   82.788 |   82.350 |  -0.4373 |  -0.528% |  +3.21 |   0.0257 |
| ITL |    0.362 |    0.361 |  -0.0012 |  -0.341% |  +4.80 |  0.00146 |
