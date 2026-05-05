# NVBug 5615248 — `trtllm-bench` PyTorch-vs-TRT TTFT comparison

Reproduces the post-fix PyTorch vs TRT TTFT / E2E / TPOT comparison for the
TinyLlama scenario behind NVBug 5615248. Uses `trtllm-bench throughput`
with `--concurrency 1` so each request is a single in-flight, single-stream
beam-10 run — the same shape as the customer's TTFT-sensitive workload.

The piecewise-CUDA-graph fix (`56a1f5e1c1` + follow-up `cb516b8223`) is
assumed to be in the build under test. Both backends use `max_seq_len=129`
so engine geometry is identical between them.

## Files in this directory

| File | Role |
|---|---|
| `pytorch.yaml`              | PyTorch backend `--config` (max_seq_len=129, beam=10, batch=1, piecewise CUDA graph). |
| `trt.yaml`                  | TRT backend `--config` (runtime knobs only — engine geometry comes from the engine dir). |
| `dataset_isl100_osl20.jsonl` | Synthetic ISL=100 / OSL=20 dataset, generated via `prepare-dataset` below. |
| `report_pytorch.json`, `report_pytorch{2..5}.json` | Aggregate reports from 5 independent PyTorch runs. |
| `report_trt.json`, `report_trt{1..5}.json`         | Aggregate reports from 6 independent TRT runs. |
| `request_pytorch*.json` / `request_trt*.json`      | Per-request timestamps + latencies (ns), one file per run. |
| `output_pytorch*.json` / `output_trt*.json`        | Per-request generated tokens, one file per run. |
| `run_pytorch*.log` / `run_trt*.log`                | Full stdout/stderr from each `trtllm-bench` run. |
| `beamwidth1/`               | Companion beam=1 experiment (separate engine, separate YAMLs, multi-run results — see `beamwidth1/REPRO.md`). |

The repeated runs are used in the *Results* section below to derive run-to-run
statistics rather than relying on a single 16-request sample.

The TRT engine itself is not in this directory; it is the prebuilt one at
`nvbugs_5615248/tinyllama_trt_engine` (`max_seq_len=129, max_batch_size=1,
max_beam_width=10, max_input_len=129, max_num_tokens=129,
paged_context_fmha=true`).

## Prerequisites

- A working TRT-LLM Python env on a GPU node (this run was on an NVIDIA L40S).
- `trtllm-bench` on `$PATH` (ships with the TRT-LLM wheel).
- TinyLlama weights at the path below — adjust `$MODEL` if yours is elsewhere.
- Prebuilt TRT engine at `nvbugs_5615248/tinyllama_trt_engine` (see
  `nvbugs_5615248/profile_ttft_trt.py` for how that engine was built).

## Steps

### Step 0 — Setup (run once per shell)

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM

MODEL=/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
WORKDIR=nvbugs_5615248/trtllm_bench
ENGINE_DIR=nvbugs_5615248/tinyllama_trt_engine
DATASET=$WORKDIR/dataset_isl100_osl20.jsonl
```

### Step 1 — Generate the dataset (ISL=100, OSL=20)

ISL=100 is well above the historical 64-token piecewise-CUDA-graph cutoff,
so it exercises the bug-relevant code path. ISL+OSL=120 fits in both
PyTorch's `max_seq_len=129` config and the TRT engine's `max_seq_len=129`.

> The Click subcommand name is `token-norm-dist` (Click converts the
> `token_norm_dist` function name's underscores to dashes). Using
> `token_norm_dist` will fail with "No such command".

```bash
trtllm-bench \
    --model "$MODEL" \
    --model_path "$MODEL" \
    --workspace "$WORKDIR" \
    prepare-dataset \
    --output "$DATASET" \
    token-norm-dist \
    --num-requests 32 \
    --input-mean 100 --input-stdev 0 \
    --output-mean 20  --output-stdev 0
```

`--num-requests 32` gives headroom over the 16-request bench run + 3 warmup
requests. `--model_path "$MODEL"` is required so `trtllm-bench` skips
`huggingface_hub.snapshot_download` (which would error on a local path).

### Step 2 — PyTorch backend (throughput, single in-flight request), repeated 5 times

The first run writes to the unsuffixed `report_pytorch.json`; runs 2..5 write
to `report_pytorch{2..5}.json`. The PyExecutor is cold-started fresh inside
`trtllm-bench` for each run (each invocation builds and warms its own LLM
instance), so runs are independent samples.

`--streaming` is required for `request_*.json` to contain
`time_to_first_token` and per-request TPOT (`intertoken_latency`).

```bash
# Run 1 (unsuffixed)
trtllm-bench \
    --model "$MODEL" \
    --model_path "$MODEL" \
    --workspace "$WORKDIR" \
    throughput \
    --backend pytorch \
    --config "$WORKDIR/pytorch.yaml" \
    --dataset "$DATASET" \
    --concurrency 1 \
    --warmup 3 \
    --num_requests 16 \
    --beam_width 10 \
    --max_batch_size 1 \
    --streaming \
    --report_json "$WORKDIR/report_pytorch.json" \
    --output_json "$WORKDIR/output_pytorch.json" \
    --request_json "$WORKDIR/request_pytorch.json" \
    2>&1 | tee "$WORKDIR/run_pytorch.log"

# Runs 2..5 (suffixed)
for i in 2 3 4 5; do
    trtllm-bench \
        --model "$MODEL" \
        --model_path "$MODEL" \
        --workspace "$WORKDIR" \
        throughput \
        --backend pytorch \
        --config "$WORKDIR/pytorch.yaml" \
        --dataset "$DATASET" \
        --concurrency 1 \
        --warmup 3 \
        --num_requests 16 \
        --beam_width 10 \
        --max_batch_size 1 \
        --streaming \
        --report_json  "$WORKDIR/report_pytorch${i}.json" \
        --output_json  "$WORKDIR/output_pytorch${i}.json" \
        --request_json "$WORKDIR/request_pytorch${i}.json" \
        2>&1 | tee "$WORKDIR/run_pytorch${i}.log"
done
```

### Step 3 — TRT backend (throughput, same workload, prebuilt engine), repeated 6 times

For the C++/TRT path, `--max_seq_len` cannot be passed at runtime
(`throughput.py` asserts it must be `None`), and `--engine_dir` supplies
the geometry via `<engine_dir>/config.json`. The existing
`nvbugs_5615248/tinyllama_trt_engine` is reused as-is across all runs.

```bash
# Run 1 (unsuffixed)
trtllm-bench \
    --model "$MODEL" \
    --model_path "$MODEL" \
    --workspace "$WORKDIR" \
    throughput \
    --backend tensorrt \
    --engine_dir "$ENGINE_DIR" \
    --config "$WORKDIR/trt.yaml" \
    --dataset "$DATASET" \
    --concurrency 1 \
    --warmup 3 \
    --num_requests 16 \
    --beam_width 10 \
    --max_batch_size 1 \
    --streaming \
    --report_json "$WORKDIR/report_trt.json" \
    --output_json "$WORKDIR/output_trt.json" \
    --request_json "$WORKDIR/request_trt.json" \
    2>&1 | tee "$WORKDIR/run_trt.log"

# Runs 1..5 (suffixed)
for i in 1 2 3 4 5; do
    trtllm-bench \
        --model "$MODEL" \
        --model_path "$MODEL" \
        --workspace "$WORKDIR" \
        throughput \
        --backend tensorrt \
        --engine_dir "$ENGINE_DIR" \
        --config "$WORKDIR/trt.yaml" \
        --dataset "$DATASET" \
        --concurrency 1 \
        --warmup 3 \
        --num_requests 16 \
        --beam_width 10 \
        --max_batch_size 1 \
        --streaming \
        --report_json  "$WORKDIR/report_trt${i}.json" \
        --output_json  "$WORKDIR/output_trt${i}.json" \
        --request_json "$WORKDIR/request_trt${i}.json" \
        2>&1 | tee "$WORKDIR/run_trt${i}.log"
done
```

### Step 4 — Compare per-request metrics

The `request_*.json` files have one entry per request with these fields
(times in **nanoseconds**, divide by 1e6 for ms):

- `time_to_first_token` → TTFT
- `end_to_end_latency`  → total request latency
- `intertoken_latency`  → TPOT (per output token, after the first)

Aggregates (avg / p50 / p90 / p95) are pre-computed in `report_*.json`
under `streaming_metrics` and `performance.request_latency_percentiles_ms`.

The aggregation script below pools over all 5 PyTorch and 6 TRT runs and
applies a Welch t-test on the per-run averages. The numerical results are
reported in the *Results* section.

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM/nvbugs_5615248/trtllm_bench
python3 - <<'PY'
import json, glob, math, statistics as st

def load(p):
    with open(p) as f: return json.load(f)

# Filter out the early_emit experimental reports already in this dir
pyt_reports = sorted([f for f in glob.glob('report_pytorch*.json')
                      if 'early_emit' not in f])
trt_reports = sorted(glob.glob('report_trt*.json'))

def pull(rp):
    r = load(rp)
    perf = r['performance']; sm = r['streaming_metrics']; e = r['energy']
    # OSL=20 -> 19 inter-token gaps. (E2E - TTFT) / 19 = real per-step decode.
    # NOTE: do NOT use sm['avg_tpot_ms'] for cross-beam comparisons — it's
    # generation_time / num_generated_tokens, and num_generated_tokens is
    # summed across beams (so beam=10 deflates by ~10).
    return {
        'file':    rp,
        'ttft':    sm['avg_ttft_ms'],
        'ttft_p95': sm['ttft_percentiles']['p95'],
        'e2e':     perf['avg_request_latency_ms'],
        'e2e_p95': perf['request_latency_percentiles_ms']['p95'],
        'per_step': (perf['avg_request_latency_ms'] - sm['avg_ttft_ms']) / 19.0,
        'tot_run': perf['total_latency_ms'],
        'rps':     perf['request_throughput_req_s'],
        'avg_w':   e['average_gpu_power'],
        'energy_j':e['total_energy_j'],
    }

pyt = [pull(p) for p in pyt_reports]
trt = [pull(p) for p in trt_reports]

def welch(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a)/na, sum(b)/nb
    va = st.variance(a) if na>1 else 0; vb = st.variance(b) if nb>1 else 0
    se = math.sqrt(va/na + vb/nb)
    return ma-mb, (ma-mb)/se if se>0 else float('inf')

print(f'PyT runs: {len(pyt)}, TRT runs: {len(trt)}')
for m in ('ttft','ttft_p95','per_step','e2e','e2e_p95','tot_run','rps','avg_w','energy_j'):
    pv = [r[m] for r in pyt]; tv = [r[m] for r in trt]
    pm, ps = sum(pv)/len(pv), st.stdev(pv) if len(pv)>1 else 0
    tm, ts = sum(tv)/len(tv), st.stdev(tv) if len(tv)>1 else 0
    diff, t = welch(pv, tv)
    print(f'{m:<10} PyT={pm:9.3f}±{ps:6.3f}  TRT={tm:9.3f}±{ts:6.3f}  PyT-TRT={diff:+8.3f}  |t|={abs(t):6.2f}')
PY
```

## Why these flag choices

- **`throughput --concurrency 1` instead of `latency`** — the `latency`
  subcommand forces `chunking = False` and forces `max_batch_size = 1`
  inside `low_latency.py`. This bug is specifically about chunked-prefill
  + piecewise CUDA graph, so we need chunking on. `throughput` keeps
  chunking on (and `enable_chunked_prefill: true` in the YAMLs re-asserts
  it via `extra_llm_api_options`); `--concurrency 1` keeps the workload
  single-stream so per-request TTFT is not contaminated by queuing.
- **`max_seq_len=129` on PyTorch** — matches the TRT engine's baked
  `max_seq_len=129`, so engine geometry is identical across the two
  backends. With the piecewise-CUDA-graph fix in place, the 128-token
  capture slot is recorded for both 128 and 129, so this is a fair
  apples-to-apples comparison rather than a bug-vs-fix comparison.
- **`--model "$MODEL" --model_path "$MODEL"`** — `--model_path` set
  causes `bench_env.checkpoint_path` to be non-`None`, which makes the
  throughput path skip `snapshot_download(model)` (which would otherwise
  treat the local path as an HF repo name and error out).
- **`token-norm-dist` (dashes), not `token_norm_dist` (underscores)** —
  Click 8.x renames function-name underscores to dashes when registering
  commands.

## Results

NVIDIA L40S, TinyLlama-1.1B-Chat-v1.0, ISL=100 / OSL=20, beam=10, concurrency=1,
16 requests per run, 3 warmup requests per run. **5 PyTorch runs + 6 TRT runs**
(80 PyT + 96 TRT pooled per-request samples).

> **Reporting caveat — `avg_tpot_ms` is per-beam-token, not per-decode-iteration.**
> `trtllm-bench` reports `avg_tpot_ms = generation_time / num_generated_tokens`,
> where `num_generated_tokens` is summed across beams. At beam=10 the raw
> `avg_tpot_ms` deflates by ~10×. The fair "time per decode iteration" is
> `(avg_request_latency_ms - avg_ttft_ms) / (OSL - 1)`, reported below as
> `per_step`. Use that for any cross-beam comparison against `beamwidth1/REPRO.md`.

### TL;DR — PyT vs TRT comparison at beam=10

| Metric | PyTorch (5 runs) | TRT (6 runs) | PyT − TRT | Δ % | Winner | Welch \|t\| |
|---|---:|---:|---:|---:|:---:|---:|
| TTFT avg (ms)        |  11.280 ± 0.314 |   5.861 ± 0.089 |  +5.419 | +92.5 % | TRT     |  37.4 |
| TTFT p95 (ms)        |  12.049 ± 0.648 |   6.302 ± 0.285 |  +5.747 | +91.2 % | TRT     |  18.4 |
| Per-step decode (ms) |   3.840 ± 0.008 |   3.926 ± 0.002 |  −0.086 |  −2.2 % | **PyT** |  23.0 |
| E2E avg (ms)         |  84.244 ± 0.199 |  80.458 ± 0.114 |  +3.786 |  +4.7 % | TRT     |  37.7 |
| E2E p95 (ms)         |  84.668 ± 0.333 |  81.176 ± 0.517 |  +3.493 |  +4.3 % | TRT     |  13.5 |
| Total run (ms)       | 1348.49 ± 3.18  | 1287.90 ± 1.83  | +60.588 |  +4.7 % | TRT     |  37.8 |
| Throughput (req/s)   |  11.865 ± 0.028 |  12.423 ± 0.018 |  −0.558 |  −4.5 % | TRT     |  38.7 |
| Avg GPU power (W)    | 233.51  ± 9.47  | 235.51  ± 7.32  |  −2.006 |  −0.9 % | (n.s.)  |   0.4 |
| Energy / run (J)     | 314.88  ± 12.73 | 303.32  ± 9.32  | +11.565 |  +3.8 % | (n.s.)  |   1.7 |

`±` is run-to-run stddev across the 5 PyT / 6 TRT independent runs. Δ % is
`(PyT − TRT) / TRT`. With these sample sizes, Welch `|t| ≥ 4` corresponds to
p < 0.001; every latency / throughput row clears that bar comfortably.

Notable: **avg power and energy are not statistically distinguishable at
beam=10** (`|t| = 0.4` and `1.7`), unlike beam=1 where TRT was ~9 % more
energy-efficient. PyT's beam=10 average power (234 W) is materially lower
than its beam=1 average (262 W) — beam=10's wider decode work seems to hit a
lower-power kernel mix in PyTorch.

**One-line summary:** TRT decisively wins TTFT, E2E, and throughput;
PyTorch keeps a small per-step decode advantage; energy/power are a wash.

### Per-run table

PyTorch (5 runs):

| file | TTFT_avg | TTFT_p95 | PerStep | E2E_avg | E2E_p95 | TotRun | req/s | avgW | E_J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `report_pytorch.json`  | 11.359 | 12.894 | 3.832 | 84.174 | 84.476 | 1347.40 | 11.87 | 223.1 | 300.6 |
| `report_pytorch2.json` | 11.420 | 12.100 | 3.837 | 84.318 | 84.865 | 1349.65 | 11.85 | 223.3 | 301.3 |
| `report_pytorch3.json` | 11.034 | 11.253 | 3.850 | 84.185 | 84.554 | 1347.54 | 11.87 | 240.6 | 324.2 |
| `report_pytorch4.json` | 11.687 | 12.399 | 3.834 | 84.539 | 85.141 | 1353.19 | 11.82 | 239.3 | 323.8 |
| `report_pytorch5.json` | 10.902 | 11.597 | 3.847 | 84.004 | 84.305 | 1344.64 | 11.90 | 241.4 | 324.5 |

TRT (6 runs):

| file | TTFT_avg | TTFT_p95 | PerStep | E2E_avg | E2E_p95 | TotRun | req/s | avgW | E_J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `report_trt.json`  | 5.890 | 6.146 | 3.928 | 80.525 | 80.861 | 1288.97 | 12.41 | 237.1 | 305.6 |
| `report_trt1.json` | 5.848 | 6.486 | 3.923 | 80.378 | 80.919 | 1286.63 | 12.44 | 239.2 | 307.7 |
| `report_trt2.json` | 6.003 | 6.610 | 3.928 | 80.641 | 82.094 | 1290.82 | 12.40 | 236.5 | 305.2 |
| `report_trt3.json` | 5.891 | 6.547 | 3.926 | 80.476 | 81.377 | 1288.19 | 12.42 | 220.9 | 284.5 |
| `report_trt4.json` | 5.773 | 6.119 | 3.924 | 80.326 | 80.638 | 1285.78 | 12.44 | 240.1 | 308.7 |
| `report_trt5.json` | 5.764 | 5.904 | 3.928 | 80.401 | 81.164 | 1287.01 | 12.43 | 239.5 | 308.2 |

### Pooled per-request distribution

```
PyTorch pooled (n=80):
  TTFT  mean=11.280 sd=0.485  p50=11.172 p90=12.106 p95=12.278 p99=12.678 max=12.894
  E2E   mean=84.244 sd=0.314  p50=84.201 p90=84.586 p95=84.984 p99=85.120 max=85.141

TRT pooled (n=96):
  TTFT  mean= 5.861 sd=0.220  p50= 5.839 p90= 6.108 p95= 6.198 p99= 6.551 max= 6.610
  E2E   mean=80.458 sd=0.307  p50=80.412 p90=80.791 p95=80.955 p99=81.413 max=82.094
```

PyTorch's TTFT distribution is wider than TRT's at beam=10 (sd 0.485 vs
0.220) and the tail extends much further (p99 = 12.68 ms vs 6.55 ms).
E2E variance is comparable across backends (sd ~0.31 ms).

## Takeaways

- **TRT wins beam=10 E2E by 4.7 % (3.79 ms)** — the original parent-experiment
  headline holds up under multi-run statistics, with much higher confidence
  (`|t| = 37.7`).
- **TTFT is ~92 % faster on TRT** (5.86 ms vs 11.28 ms, ~5.4 ms gap).
  This is *not* the eager-fallback regression from the bug — the
  piecewise capture set now includes the 128 slot, so the 100-token chunk
  pads to 128 and runs on a CUDA graph. About half of this TTFT gap is
  beam-search-attributable setup overhead in the PyExecutor (see the
  decomposition in `beamwidth1/REPRO.md`); the structural prefill scheduling
  difference accounts for ~2.3 ms of TTFT (the residual that survives at
  beam=1).
- **Per-step decode is essentially on par** — PyTorch is 2.2 % faster per
  decode iteration (3.840 vs 3.926 ms/step). The PyTorch CUDA-graph +
  piecewise capture is doing its job in the decode phase.
- **Energy / power are not statistically distinguishable at beam=10**
  (`|t| < 2`), in contrast to beam=1 where TRT is ~9 % more
  energy-efficient. PyTorch's average power *drops* from 262 W at beam=1
  to 234 W at beam=10.
- **PyTorch's TTFT is noisier** (run-to-run sd 0.31 ms vs TRT 0.09 ms;
  per-request sd 0.49 ms vs 0.22 ms), and its TTFT tail (p99 = 12.68 ms)
  extends significantly further than TRT's (p99 = 6.55 ms). For p99-sensitive
  SLAs, the gap is worse than the mean implies.

### Cross-reference: beam=10 vs beam=1

The companion experiment in `beamwidth1/` shows that turning beam search
*off* (beam=1) flips the E2E winner to PyTorch. With the multi-run averages
on both sides:

| beam | PyT E2E (ms) | TRT E2E (ms) | E2E winner |
|---:|---:|---:|---|
|  10 |  84.244 (5 runs) |  80.458 (6 runs) | TRT  +4.7 % |
|   1 |  72.309 (5 runs) |  73.418 (6 runs) | PyT  +1.5 % |

Cost of "turning beam search on" decomposes as (using all multi-run data):

|  | PyTorch | TRT | PyT excess |
|---|---:|---:|---:|
| ΔTTFT (one-time prefill setup) | +3.446 ms | +0.353 ms | **+3.09 ms** |
| ΔPer-step × 19 (decode loop)    | +8.49 ms  | +6.69 ms  | **+1.80 ms** |
| ΔE2E (sum)                      | +11.94 ms | +7.04 ms  | **+4.90 ms** |

The 4.90 ms PyTorch-excess beam-search cost exceeds the entire 3.79 ms
beam=10 E2E gap that TRT wins by — i.e. **beam-search overhead in the
PyExecutor (heaviest in the prefill→decode handoff) is the dominant
explanation for the beam=10 PyT-vs-TRT loss**, not a structural prefill
scheduling difference. See `beamwidth1/REPRO.md` for the full derivation
and the OSL crossover model.
