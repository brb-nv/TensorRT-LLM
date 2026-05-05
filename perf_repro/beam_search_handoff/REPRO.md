# Beam-search prefill->decode handoff: before/after perf repro

Verifies the perf gains of the optimizations in
`user/brb/beam-search-handoff-perf` against `main`. The branch under test
is the cleaned-up MR (no NVTX, no testing artifacts) — these scripts live
**outside** the MR (`perf_repro/` is intentionally untracked) so branch
switching does not move them.

## Workload

| Param | Value | Why |
|---|---|---|
| Model | `TinyLlama-1.1B-Chat-v1.0` | Small enough that host overhead dominates per-step decode. |
| ISL / OSL | 100 / 20 | ISL > 64 exercises the bug-relevant piecewise-CUDA-graph path. |
| `beam_width` | 10 | Optimization targets beam-search-only buffers. |
| `max_batch_size` | 1 | Single in-flight beam set per step. |
| `--concurrency 1 --streaming` | — | Required for per-request `time_to_first_token` + `intertoken_latency` in `request_*.json`. |
| `--warmup 3` | — | Kicks `_prepare_beam_search` + the per-step decode loop into steady state before the timed `--num_requests 16`. |
| Runs per measurement | 5 | Per-request signal is ~0.1 ms; multi-run aggregation lifts it above noise. |

The optimization touches:

1. **One-shot prefill->decode handoff** (`_prepare_beam_search` + finish-reasons update) — should drop TTFT and E2E, leave ITL flat.
2. **Per-step decode** (cached `seq_offsets` / `beam_idx_arange` slices) — should drop ITL (and therefore E2E) by a small per-token amount.

## Files in this directory

| File | Role |
|---|---|
| `pytorch.yaml` | PyTorch backend `--config` (max_seq_len=129, beam=10, batch=1, piecewise CUDA graph). |
| `run_multirun.sh` | 5x `trtllm-bench` launcher. Writes `report_pytorch{,2..5}.json`, `request_pytorch{,2..5}.json`, etc. into a CLI-supplied directory. |
| `aggregate_runs.py` | Per-run-mean + pooled per-request aggregator with two-sided Welch's t-test. Accepts `--baseline` and `--experiment`. |
| `dataset_isl100_osl20.jsonl` | Generated once via Step 1 below; gitignored. |

## Prerequisites

- A working TRT-LLM env on a GPU node (original numbers were on L40S; trends should hold on similar GPUs — absolute deltas will shift).
- `trtllm-bench` on `$PATH`.
- TinyLlama weights at `$MODEL` (override with `MODEL=...`).

## Step 0 — Setup (run once per shell)

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM

export MODEL=/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
export WORKDIR=perf_repro/beam_search_handoff
export DATASET=$WORKDIR/dataset_isl100_osl20.jsonl
```

## Step 1 — Generate the dataset (once)

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

`--num-requests 32` gives headroom over the 16-request bench run + 3 warmup. `--model_path "$MODEL"` makes `trtllm-bench` skip `huggingface_hub.snapshot_download` (which errors on a local path).

## Step 2 — Baseline run on `main` (BEFORE)

```bash
git checkout main
# (rebuild / re-install if you have a dev wheel; needed only if your
#  environment was built from the MR branch.)

bash $WORKDIR/run_multirun.sh $WORKDIR/before
```

## Step 3 — Optimized run on the MR branch (AFTER)

```bash
git checkout user/brb/beam-search-handoff-perf
# (rebuild / re-install as above.)

bash $WORKDIR/run_multirun.sh $WORKDIR/after
```

## Step 4 — Aggregate and compare

```bash
python3 $WORKDIR/aggregate_runs.py \
    --baseline   $WORKDIR/before \
    --experiment $WORKDIR/after
```

## What to expect

Reported on TinyLlama-1.1B beam=10 ISL=100/OSL=20 single-stream (L40S),
n=80 pooled per-request samples (5 runs × 16 requests):

| Metric | Pre-fix | Post-fix | Delta | Welch's t / p |
|---|---:|---:|---:|---|
| TTFT (ms) | 11.281 | 10.881 | **−0.400 (−3.5 %)** | t = +5.84, p ≈ 2 × 10⁻⁷ |
| E2E (ms)  | 84.244 | 83.714 | **−0.530 (−0.63 %)** | t = +9.95, p ≈ 5 × 10⁻²³ |
| ITL (ms/token) | 0.367 | 0.366 | **−0.001 (−0.18 %)** | t = +1.80, p ≈ 0.017 |

Per-call NVTX breakdown (single nsys trace, for context — you don't need NVTX to verify the multi-run E2E delta):

| Range | Pre-fix | Post-fix | Delta |
|---|---:|---:|---:|
| `prepare_beam_search` body | 500 µs | 196 µs | −61 % |
| GPU kernels per `prepare_beam_search` | 21 | 8 | — |
| `setup.finish_reasons.update` | 131 µs | 78 µs | −40 % |
| `bss.cache_indirection_swap` (median) | 149 µs | 124 µs | −17 % |
| `bss.finished_beams_update` (median) | 127 µs | 101 µs | −20 % |

### Interpreting the result

- **TTFT and E2E should drop**; **ITL should be roughly flat or marginally lower**. ITL ≈ flat is the cleanest mechanistic confirmation that the win comes from the one-shot prefill->decode handoff and the per-step constant-cache (which together remove ~30 + 4 kernel launches), and not from anything that scales with output tokens.
- If TTFT regresses but E2E improves (or vice versa), inspect `report_pytorch*.json` in both directories — the `request_*.json` files contain the raw nanosecond timestamps that drive the t-test.
- On a faster GPU (H100/B200) the absolute deltas shrink because the host-overhead band is smaller; the directional sign and the per-call NVTX deltas should still hold.

## Quick single-shot variant (no t-test)

If you just want a sanity check (~3 min instead of ~15 min):

```bash
NUM_RUNS=1 NUM_REQUESTS=16 bash $WORKDIR/run_multirun.sh $WORKDIR/before
git checkout user/brb/beam-search-handoff-perf  # + rebuild
NUM_RUNS=1 NUM_REQUESTS=16 bash $WORKDIR/run_multirun.sh $WORKDIR/after

python3 -c "
import json
for tag in ('before', 'after'):
    r = json.load(open(f'$WORKDIR/{tag}/report_pytorch.json'))
    p = r['performance_overview']
    print(f'{tag:6s}  TTFT(p50)={p[\"average_time_to_first_token\"]:8.3f} ms  '
          f'E2E(p50)={p[\"average_request_latency\"]:8.3f} ms  '
          f'ITL(avg)={p[\"average_intertoken_latency\"]:8.4f} ms')
"
```

This is too noisy for the small per-request delta but is enough to catch a regression of >5 %.
