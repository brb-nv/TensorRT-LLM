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
| `report_pytorch.json` / `report_trt.json` | Aggregate report (avg / p50 / p95 of TTFT, TPOT, E2E). |
| `request_pytorch.json` / `request_trt.json` | Per-request timestamps + latencies (ns). |
| `output_pytorch.json` / `output_trt.json`   | Per-request generated tokens. |
| `run_pytorch.log` / `run_trt.log`           | Full stdout/stderr from each `trtllm-bench` run. |

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

### Step 2 — PyTorch backend (throughput, single in-flight request)

```bash
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
```

`--streaming` is required for `request_pytorch.json` to contain
`time_to_first_token` and per-request TPOT (`intertoken_latency`).

### Step 3 — TRT backend (throughput, same workload, prebuilt engine)

For the C++/TRT path, `--max_seq_len` cannot be passed at runtime
(`throughput.py` asserts it must be `None`), and `--engine_dir` supplies
the geometry via `<engine_dir>/config.json`. The existing
`nvbugs_5615248/tinyllama_trt_engine` is reused as-is.

```bash
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
```

### Step 4 — Compare per-request metrics

The `request_*.json` files have one entry per request with these fields
(times in **nanoseconds**, divide by 1e6 for ms):

- `time_to_first_token` → TTFT
- `end_to_end_latency`  → total request latency
- `intertoken_latency`  → TPOT (per output token, after the first)

Aggregates (avg / p50 / p90 / p95) are pre-computed in `report_*.json`
under `streaming_metrics` and `performance.request_latency_percentiles_ms`.

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

## Results from this run (NVIDIA L40S, TinyLlama, ISL=100 / OSL=20, beam=10, concurrency=1, 16 requests)

|                       | PyTorch | TRT    | Winner |
|---|---:|---:|:---|
| avg TTFT (ms)         | 11.494 | 6.332  | TRT wins (45%) |
| p50 TTFT (ms)         | 11.370 | 6.309  | TRT wins (45%) |
| p95 TTFT (ms)         | 13.393 | 6.918  | TRT wins (48%) |
| avg E2E latency (ms)  | 84.521 | 80.896 | TRT wins (4.3%) |
| p50 E2E (ms)          | 84.421 | 80.904 | TRT wins (4.2%) |
| p95 E2E (ms)          | 85.923 | 81.366 | TRT wins (5.3%) |
| avg TPOT (ms/tok)     | 0.3670 | 0.3747 | PyTorch wins (2.1%) |
| p50 TPOT (ms/tok)     | 0.3677 | 0.3747 | PyTorch wins (1.9%) |
| p95 TPOT (ms/tok)     | 0.3687 | 0.3756 | PyTorch wins (1.8%) |
| total run latency (ms)| 1352.92 | 1294.89 | TRT wins (4.3%) |
| req / s               | 11.83  | 12.36  | TRT wins (4.5%) |

Lower is better for TTFT / E2E / TPOT; higher is better for req/s.

## Takeaways

- TPOT (steady-state decode) is essentially on par — PyTorch is 2 % faster
  per token. The PyTorch CUDA-graph + piecewise capture is doing its job
  in the decode phase.
- TTFT (single 100-token prefill) is ~45 % faster on TRT (~5.2 ms gap).
  This is *not* the eager-fallback regression from the bug — the
  piecewise capture set now includes the 128 slot, so the 100-token chunk
  pads to 128 and runs on a CUDA graph. The remaining gap is structural
  PyExecutor scheduling + per-subgraph replay overhead vs TRT's monolithic
  prefill graph + lighter C++ scheduler.
- Net E2E latency: TRT wins by ~3.6 ms (4.3 %), entirely from the prefill
  phase, slightly offset by PyTorch's marginally better decode.
