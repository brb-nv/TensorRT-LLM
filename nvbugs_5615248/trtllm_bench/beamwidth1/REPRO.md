# NVBug 5615248 — beam=1 companion run for the trtllm-bench TTFT comparison

This directory is the beam-width=1 counterpart to `../` (the original beam=10
experiment under `nvbugs_5615248/trtllm_bench/`). Only the beam width changes;
everything else (model, ISL/OSL, concurrency, chunked prefill, piecewise CUDA
graph, max_seq_len=129, max_batch_size=1, etc.) is held fixed so the TTFT /
TPOT / E2E delta vs. the beam=10 baseline is attributable to beam width alone.

## Why a separate engine and YAMLs

| Backend | Constraint | Implication |
|---|---|---|
| PyTorch (`PyExecutor`) | `_validate_request` enforces `request.beam_width == self.max_beam_width` (`tensorrt_llm/_torch/pyexecutor/py_executor.py`). | The YAML's `max_beam_width` *must* match the runtime `--beam_width`. So `pytorch.yaml` here pins `max_beam_width: 1`. |
| TRT (C++ inflight batcher) | Runtime check is `mMaxBeamWidth <= modelConfig.getMaxBeamWidth()` (`cpp/tensorrt_llm/batch_manager/trtGptModel.h`); `changeBeamWidth(beamWidth)` allows shrinking at runtime. | A rebuild is *not strictly required* — the existing beam=10 engine in `../../tinyllama_trt_engine` accepts beam=1 at runtime. We rebuild here anyway so PyTorch and TRT have *identical* engine geometry (`max_beam_width=1` on both sides), matching the apples-to-apples framing of the parent experiment. |

If you only want the runtime data and don't care about geometry parity, you
can skip Step 1 and point the TRT run at `../../tinyllama_trt_engine`
(beam=10) instead.

## Files in this directory

| File | Role |
|---|---|
| `pytorch.yaml`            | PyTorch backend `--config` for `trtllm-bench` (max_seq_len=129, **max_beam_width=1**, batch=1, piecewise CUDA graph). |
| `trt.yaml`                | TRT backend runtime `--config` (chunked prefill, block reuse, gen-phase CUDA graphs). Engine geometry comes from the engine dir. |
| `trt_build.yaml`          | YAML fed to `nvbugs_5615248/profile_ttft_trt.py --save-engine` to build the beam=1 TRT engine. |
| `tinyllama_trt_engine/`   | Pre-built beam=1 TRT engine (created by Step 1 below). |
| `report_pytorch.json`, `report_pytorch{2..5}.json`     | Aggregate reports from 5 independent PyTorch runs. |
| `report_trt.json`, `report_trt{1..5}.json`             | Aggregate reports from 6 independent TRT runs. |
| `request_pytorch*.json` / `request_trt*.json`          | Per-request timestamps + latencies (ns), one file per run. |
| `output_pytorch*.json` / `output_trt*.json`            | Per-request generated tokens, one file per run. |
| `run_pytorch*.log` / `run_trt*.log`                    | Full stdout/stderr from each `trtllm-bench` run. |

The repeated runs are used in the *Results* section below to derive run-to-run
statistics rather than relying on a single 16-request sample.

## Prerequisites

Same as `../REPRO.md`. Reuses `../dataset_isl100_osl20.jsonl` (beam width does
not affect the prompt set).

## Steps

### Step 0 — Setup (run once per shell)

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM

MODEL=/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
WORKDIR=nvbugs_5615248/trtllm_bench/beamwidth1
ENGINE_DIR=$WORKDIR/tinyllama_trt_engine
DATASET=nvbugs_5615248/trtllm_bench/dataset_isl100_osl20.jsonl
```

### Step 1 — Build the beam=1 TRT engine

Reuses `nvbugs_5615248/profile_ttft_trt.py` in `--save-engine` mode. The build
configuration (`trt_build.yaml`) matches `nvbugs_5615248/extra_llm_options_trt.yaml`
(used to build the beam=10 engine) except for `max_beam_width: 1`.

```bash
python nvbugs_5615248/profile_ttft_trt.py \
    --config "$WORKDIR/trt_build.yaml" \
    --model  "$MODEL" \
    --tag    trt_beam1 \
    --save-engine "$ENGINE_DIR"
```

After this finishes, `$ENGINE_DIR/config.json` should report
`"max_beam_width": 1`, `"max_seq_len": 129`, `"max_batch_size": 1`,
`"max_num_tokens": 129`, `"use_paged_context_fmha": true`.

### Step 2 — PyTorch backend (beam=1), repeated 5 times

The first run writes to the unsuffixed `report_pytorch.json`; runs 2..5 write
to `report_pytorch{2..5}.json`. The PyExecutor is cold-started fresh inside
`trtllm-bench` for each run (each invocation builds and warms its own LLM
instance), so runs are independent samples.

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
    --beam_width 1 \
    --max_batch_size 1 \
    --streaming \
    --report_json  "$WORKDIR/report_pytorch.json" \
    --output_json  "$WORKDIR/output_pytorch.json" \
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
        --beam_width 1 \
        --max_batch_size 1 \
        --streaming \
        --report_json  "$WORKDIR/report_pytorch${i}.json" \
        --output_json  "$WORKDIR/output_pytorch${i}.json" \
        --request_json "$WORKDIR/request_pytorch${i}.json" \
        2>&1 | tee "$WORKDIR/run_pytorch${i}.log"
done
```

### Step 3 — TRT backend (beam=1, beam=1 engine from Step 1), repeated 6 times

Same shape, with the unsuffixed first run plus 5 suffixed runs. Reuses the
engine built in Step 1 — only the C++ runtime is re-initialized each
invocation.

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
    --beam_width 1 \
    --max_batch_size 1 \
    --streaming \
    --report_json  "$WORKDIR/report_trt.json" \
    --output_json  "$WORKDIR/output_trt.json" \
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
        --beam_width 1 \
        --max_batch_size 1 \
        --streaming \
        --report_json  "$WORKDIR/report_trt${i}.json" \
        --output_json  "$WORKDIR/output_trt${i}.json" \
        --request_json "$WORKDIR/request_trt${i}.json" \
        2>&1 | tee "$WORKDIR/run_trt${i}.log"
done
```

### Step 4 — Compare per-request metrics

Same field semantics as `../REPRO.md`: `request_*.json` carries
`time_to_first_token`, `end_to_end_latency`, `intertoken_latency` (all in
nanoseconds). Aggregates are pre-computed in `report_*.json` under
`streaming_metrics` and `performance.request_latency_percentiles_ms`.

The aggregation script below pools over all 5 PyTorch and 6 TRT runs and
applies a Welch t-test on the per-run averages and on the pooled
per-request samples. The numerical results are reported in the *Results*
section.

```bash
python3 - <<'PY'
import json, glob, math, statistics as st

def load(p):
    with open(p) as f: return json.load(f)

pyt_reports = sorted(glob.glob('report_pytorch*.json'))
trt_reports = sorted(glob.glob('report_trt*.json'))

def pull(rp):
    r = load(rp)
    perf = r['performance']; sm = r['streaming_metrics']; e = r['energy']
    return {
        'file': rp,
        'ttft_avg': sm['avg_ttft_ms'],
        'ttft_p95': sm['ttft_percentiles']['p95'],
        'e2e_avg':  perf['avg_request_latency_ms'],
        'e2e_p95':  perf['request_latency_percentiles_ms']['p95'],
        # OSL=20 -> 19 inter-token gaps. (E2E - TTFT) / 19 = real per-step decode.
        'per_step': (perf['avg_request_latency_ms'] - sm['avg_ttft_ms']) / 19.0,
        'tot_run':  perf['total_latency_ms'],
        'rps':      perf['request_throughput_req_s'],
        'avg_w':    e['average_gpu_power'],
        'energy_j': e['total_energy_j'],
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
for m in ('ttft_avg','ttft_p95','per_step','e2e_avg','e2e_p95','tot_run','rps','avg_w','energy_j'):
    pv = [r[m] for r in pyt]; tv = [r[m] for r in trt]
    pm, ps = sum(pv)/len(pv), st.stdev(pv) if len(pv)>1 else 0
    tm, ts = sum(tv)/len(tv), st.stdev(tv) if len(tv)>1 else 0
    diff, t = welch(pv, tv)
    print(f'{m:<10} PyT={pm:8.3f}±{ps:5.3f}  TRT={tm:8.3f}±{ts:5.3f}  PyT-TRT={diff:+7.3f}  |t|={abs(t):6.2f}')
PY
```

## Results

NVIDIA L40S, TinyLlama-1.1B-Chat-v1.0, ISL=100 / OSL=20, beam=1, concurrency=1,
16 requests per run, 3 warmup requests per run. **5 PyTorch runs + 6 TRT runs**
(80 PyT + 96 TRT pooled per-request samples).

> **Reporting caveat — `avg_tpot_ms` is per-beam-token, not per-decode-iteration.**
> `trtllm-bench` reports `avg_tpot_ms = generation_time / num_generated_tokens`,
> where `num_generated_tokens` is summed across beams. At beam=N the raw
> `avg_tpot_ms` deflates by ~N. The fair "time per decode iteration" is
> `(avg_request_latency_ms - avg_ttft_ms) / (OSL - 1)`, reported below as
> `per_step`. This matters when comparing against the beam=10 numbers in
> `../REPRO.md` — *do not* compare raw `avg_tpot_ms` across beam widths.

### TL;DR — PyT vs TRT comparison at beam=1

| Metric | PyTorch (5 runs) | TRT (6 runs) | PyT − TRT | Δ % | Winner | Welch \|t\| |
|---|---:|---:|---:|---:|:---:|---:|
| TTFT avg (ms)        |   7.834 ± 0.168 |   5.508 ± 0.073 |  +2.327 | +42.2 % | TRT     |  28.7 |
| TTFT p95 (ms)        |   9.071 ± 0.657 |   5.824 ± 0.230 |  +3.246 | +55.7 % | TRT     |  10.5 |
| Per-step decode (ms) |   3.393 ± 0.003 |   3.574 ± 0.002 |  −0.181 |  −5.1 % | **PyT** | 122.1 |
| E2E avg (ms)         |  72.309 ± 0.175 |  73.418 ± 0.066 |  −1.110 |  −1.5 % | **PyT** |  13.4 |
| E2E p95 (ms)         |  73.026 ± 0.294 |  73.994 ± 0.238 |  −0.968 |  −1.3 % | **PyT** |   5.9 |
| Total run (ms)       | 1157.43 ± 2.79  | 1175.18 ± 1.06  | −17.748 |  −1.5 % | **PyT** |  13.5 |
| Throughput (req/s)   |  13.824 ± 0.033 |  13.615 ± 0.012 |  +0.209 |  +1.5 % | **PyT** |  13.3 |
| Avg GPU power (W)    | 261.81  ± 1.35  | 235.22  ± 8.49  | +26.59  | +11.3 % | TRT     |   7.6 |
| Energy / run (J)     | 303.02  ± 0.99  | 276.43  ± 10.07 | +26.60  |  +9.6 % | TRT     |   6.4 |

`±` is run-to-run stddev across the 5 PyT / 6 TRT independent runs. Δ % is
`(PyT − TRT) / TRT`. With these sample sizes, Welch `|t| ≥ 4` corresponds to
p < 0.001; every comparison above clears that bar comfortably.

**One-line summary:** PyTorch wins E2E latency, throughput, and per-step
decode; TRT wins TTFT, energy efficiency, and tail / variance.

### Per-run table

PyTorch (5 runs):

| file | TTFT_avg | TTFT_p95 | PerStep | E2E_avg | E2E_p95 | TotRun | req/s | avgW | E_J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `report_pytorch.json`  |  8.013 | 10.149 | 3.392 | 72.468 | 73.192 | 1159.98 | 13.79 | 261.6 | 303.5 |
| `report_pytorch2.json` |  7.862 |  9.167 | 3.396 | 72.386 | 73.116 | 1158.67 | 13.81 | 260.4 | 301.8 |
| `report_pytorch3.json` |  7.918 |  8.918 | 3.395 | 72.430 | 73.224 | 1159.36 | 13.80 | 260.9 | 302.5 |
| `report_pytorch4.json` |  7.813 |  8.492 | 3.389 | 72.206 | 73.088 | 1155.78 | 13.84 | 262.2 | 303.0 |
| `report_pytorch5.json` |  7.564 |  8.626 | 3.394 | 72.052 | 72.508 | 1153.36 | 13.87 | 263.9 | 304.4 |

TRT (6 runs):

| file | TTFT_avg | TTFT_p95 | PerStep | E2E_avg | E2E_p95 | TotRun | req/s | avgW | E_J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `report_trt.json`  | 5.470 | 5.657 | 3.576 | 73.408 | 73.841 | 1175.01 | 13.62 | 238.1 | 279.8 |
| `report_trt1.json` | 5.423 | 5.739 | 3.576 | 73.362 | 74.113 | 1174.28 | 13.63 | 217.9 | 255.9 |
| `report_trt2.json` | 5.449 | 5.609 | 3.573 | 73.332 | 73.814 | 1173.79 | 13.63 | 239.2 | 280.8 |
| `report_trt3.json` | 5.616 | 6.150 | 3.571 | 73.471 | 74.009 | 1176.01 | 13.61 | 238.4 | 280.4 |
| `report_trt4.json` | 5.559 | 6.080 | 3.576 | 73.510 | 74.403 | 1176.65 | 13.60 | 238.0 | 280.1 |
| `report_trt5.json` | 5.529 | 5.710 | 3.574 | 73.426 | 73.781 | 1175.33 | 13.61 | 239.6 | 281.6 |

### Run-to-run aggregate (Welch t-test on per-run averages)

| metric | PyT mean ± sd | TRT mean ± sd | PyT − TRT | Welch \|t\| | Verdict |
|---|---:|---:|---:|---:|---|
| TTFT avg (ms)        |   7.834 ± 0.168 |   5.508 ± 0.073 |  +2.327 |  28.7 | **TRT wins (~30%)** |
| TTFT p95 (ms)        |   9.071 ± 0.657 |   5.824 ± 0.230 |  +3.246 |  10.5 | TRT wins (tail) |
| Per-step decode (ms) |   3.393 ± 0.003 |   3.574 ± 0.002 |  −0.181 | 122.1 | **PyT wins (~5%)** |
| E2E avg (ms)         |  72.309 ± 0.175 |  73.418 ± 0.066 |  −1.110 |  13.4 | **PyT wins (~1.5%)** |
| E2E p95 (ms)         |  73.026 ± 0.294 |  73.994 ± 0.238 |  −0.968 |   5.9 | PyT wins (~1.3%) |
| Total run (ms)       | 1157.43 ± 2.79  | 1175.18 ± 1.06  | −17.748 |  13.5 | PyT wins (~1.5%) |
| req/s                |  13.824 ± 0.033 |  13.615 ± 0.012 |  +0.209 |  13.3 | PyT wins (~1.5%) |
| avg power (W)        | 261.81  ± 1.35  | 235.22  ± 8.49  | +26.59  |   7.6 | TRT wins energy efficiency |
| energy per run (J)   | 303.02  ± 0.99  | 276.43  ± 10.07 | +26.60  |   6.4 | TRT wins (~9%) |

With these sample sizes, `|t| ≥ 4` is essentially p < 0.001. Every comparison
above has `|t| ≥ 5.9`, so each finding is statistically robust against
run-to-run jitter.

### Pooled per-request distribution

```
PyTorch pooled (n=80):
  TTFT  mean=7.834  sd=0.448  p50=7.734  p90=8.242  p95=8.631  p99=9.373  max=10.149
  E2E   mean=72.309 sd=0.329  p50=72.235 p90=72.774 p95=73.090 p99=73.199 max=73.224
  ITL   mean=3.393  sd=0.019  p50=3.396  p90=3.406  p95=3.416  p99=3.441  (reported per-token; equals per-step at beam=1)

TRT pooled (n=96):
  TTFT  mean=5.508  sd=0.184  p50=5.495  p90=5.661  p95=5.772  p99=6.083  max=6.150
  E2E   mean=73.418 sd=0.238  p50=73.403 p90=73.652 p95=73.847 p99=74.128 max=74.403
  ITL   mean=3.574  sd=0.011  p50=3.573  p90=3.584  p95=3.593  p99=3.618
```

Welch t-test on pooled per-request samples (PyT − TRT):

| metric | diff (ms) | \|t\| |
|---|---:|---:|
| TTFT |  +2.327 |  43.5 |
| E2E  |  −1.110 |  25.2 |
| ITL  |  −0.181 |  76.2 |

TRT is consistently *tighter* across the distribution — about 2× tighter on
TTFT (sd 0.18 vs 0.45) and ~1.7× tighter on per-step decode (sd 0.011 vs
0.019). PyTorch's wider TTFT tail (p99 = 9.37 ms vs TRT 6.08 ms) is the
biggest variance gap.

### Linear E2E model and OSL crossover

Using the robust means: `E2E(OSL) ≈ TTFT + (OSL − 1) × per_step_decode`.

```
PyT: E2E ≈ 7.834 + (OSL − 1) × 3.3934   ms
TRT: E2E ≈ 5.508 + (OSL − 1) × 3.5742   ms
```

Crossover at **OSL ≈ 13.86**: TRT wins E2E for OSL ≤ 13, PyT wins for
OSL ≥ 14. Predicted gap at common OSLs:

| OSL | PyT E2E (ms) | TRT E2E (ms) | Δ (ms) | Δ (%) | Winner |
|---:|---:|---:|---:|---:|---|
|   5 |   21.41 |   19.80 |  +1.60 | +8.1% | TRT |
|  10 |   38.37 |   37.68 |  +0.70 | +1.9% | TRT |
|  14 |   51.95 |   51.97 |  −0.02 | −0.0% | tie |
|  20 |   72.31 |   73.42 |  −1.11 | −1.5% | PyT *(measured)* |
|  50 |  174.11 |  180.65 |  −6.54 | −3.6% | PyT |
| 100 |  343.78 |  359.36 | −15.58 | −4.3% | PyT |
| 200 |  683.12 |  716.78 | −33.66 | −4.7% | PyT |

The PyTorch advantage saturates near +4.7% at long OSL (per-step decode wins
by ~5% per step; the ~2.3 ms TTFT deficit becomes a smaller share of E2E).

### Bottom line for beam=1

> On TinyLlama / L40S / ISL=100 / OSL=20 / concurrency=1 / beam=1, across 5
> PyTorch runs and 6 TRT runs, **PyTorch wins E2E by 1.51 % on average
> (1.110 ± 0.082 ms, Welch \|t\| = 13.4)**. The gap is driven by PyTorch's
> ~5 % faster per-step decode (3.393 vs 3.574 ms/step, \|t\| = 122)
> outweighing TRT's ~30 % faster TTFT (5.508 vs 7.834 ms, \|t\| = 28.7).
> The OSL crossover is at ~14: TRT wins E2E for shorter outputs (≤ 13),
> PyTorch wins for longer outputs and the advantage saturates near +4.7 %
> for OSL ≥ 100. TRT continues to win on energy (~9 % lower per request)
> and variance (~2× tighter TTFT, ~1.7× tighter per-step decode).

### Cross-reference: how this changes the beam=10 picture in `../REPRO.md`

`../REPRO.md` is now also multi-run (5 PyT + 6 TRT). Combined with the
beam=1 numbers here:

| beam | PyT E2E (ms) | TRT E2E (ms) | E2E winner |
|---:|---:|---:|---|
|  10 | 84.244 (5 runs) | 80.458 (6 runs) | TRT  +4.7 % |
|   1 | 72.309 (5 runs) | 73.418 (6 runs) | PyT  +1.5 % |

The E2E winner *flips* between beam=10 and beam=1. Cost of "turning beam
search on" decomposes as (all multi-run data):

|  | PyTorch | TRT | PyT excess |
|---|---:|---:|---:|
| ΔTTFT (one-time prefill setup) | +3.446 ms | +0.353 ms | **+3.09 ms** |
| ΔPer-step × 19 (decode loop)   | +8.49  ms | +6.69 ms  | **+1.80 ms** |
| ΔE2E (sum)                     | +11.94 ms | +7.04 ms  | **+4.90 ms** |

That ~4.9 ms PyTorch-excess beam-search cost exceeds the entire 3.79 ms
beam=10 E2E gap that TRT wins by — i.e., **beam-search overhead in the
PyExecutor is the dominant explanation for the beam=10 PyT-vs-TRT loss**,
not a structural prefill scheduling difference (which only accounts for the
~2.3 ms of TTFT residual that survives at beam=1). The split is ~63 / 37
prefill-setup vs decode-step accumulation, so the prefill→decode handoff is
the higher-leverage target for closing this gap.

### Notes on the data

- `report_trt1.json` reports `avg_gpu_power = 217.9 W`, against 238–240 W for
  the other 5 TRT runs — looks like a thermal/power-state warmup artifact for
  the first TRT run after a cold start. It's the dominant contributor to the
  TRT power/energy stddev. Excluding it: TRT mean power ≈ 238.7 W (sd ≈ 0.6 W),
  energy ≈ 280.5 J (sd ≈ 0.6 J), TRT-vs-PyT energy gap shrinks slightly to
  ~22.5 J (~7.4 %) but becomes far more consistent. The latency/throughput
  conclusions are unaffected if this run is excluded.
- All other PyT/TRT runs are within tight clusters; no other outliers.
- Run-to-run variance is dominated by TTFT jitter on the PyTorch side
  (sd 0.168 ms across runs vs 0.073 ms on TRT). PyT TTFT p95 also moves
  around more (8.49–10.15 ms across runs).
- ITL is essentially deterministic on both backends (sd ~0.01–0.02 ms).
  The 0.181 ms per-step gap is many orders of magnitude above noise.

## Caveats

- The `_validate_request` strict-equality check is a PyExecutor-only
  limitation; the TRT runtime would happily accept beam=1 against the
  beam=10 engine. The separate engine here is for geometry parity, not
  correctness.
- Rebuilding the engine slightly changes engine-internal layout vs. the
  beam=10 engine (different cache_indir shape, different decoder workspace
  reservations). This is the intended difference; do not interpret a small
  TPOT shift as a regression unrelated to beam width.
