<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Disagg vs agg TTFT — round5 (2026-05-21)

Continuation of [`FINDINGS_RWLT.md`](FINDINGS_RWLT.md). This writeup covers
the round5 ablation run that is the current best baseline, summarises the
patches it required, attributes the remaining disagg-vs-agg TTFT gap, and
hands off the most likely next experiments.

Branch: `feat/bench_x_investigate_ttft` (see `git log -8` for the two
patches landed alongside this writeup).

## TL;DR

| Quantity | agg | disagg | Δ |
|---|---:|---:|---:|
| TTFT p50 (server, ms) | **80.9** | **193.2** | **+112.4** |
| TTFT p99 (server, ms) | 229.1 | 513.5 | +284.4 |
| `gen_preproc_ms` p50 (disagg only) | — | **3.5** | (was 32.7 in round4) |
| `gen_queue_ms` p50 (disagg only) | — | 88.5 | dominant residual |
| `kv_transfer_gen_ms` p50 (disagg only) | — | 75.2 | ≈ `gen_queue_ms` |

For this workload (gpt-oss-120b multi-turn agentic coding, RWLT
top-5 trajectories, conc=1, B200) the residual disagg gap is ~110 ms
p50, of which ~80% is `gen_queue ≈ kv_transfer_gen` and the rest is
relay-hop + first-decode-on-fresh-slot.

Per-stage details, full breakdown of the worst trajectory, and the
recommended next ablation are below.

## What changed for round5

Two server-side patches landed on `tensorrt_llm/serve/openai_server.py`:

1. **Avoid duplicate harmony tokenization on gen** — when ctx has already
   harmony-tokenized the messages and forwarded the token IDs via
   `prompt_token_ids` (Lizhi's #14420 in `_get_gen_request`), the gen
   handler now consumes those token IDs instead of re-running
   `HarmonyAdapter.openai_to_harmony_tokens`. Effect: `gen_preproc_ms`
   p50 collapsed from 32.7 ms (round4) to 3.5 ms; overall TTFT gap p50
   dropped from +140.7 ms → +112.4 ms.
2. **Populate `/perf_metrics` on chat endpoints** — two distinct gaps
   prevented per-request perf metrics from being collected for chat
   (and therefore for the gpt-oss harmony streaming path used by RWLT):
   - The chat handlers never set `sampling_params.return_perf_metrics`
     per request; only `openai_completion` had the env-var gate
     (`TRTLLM_KVCACHE_TIME_OUTPUT_PATH`). Without it,
     `request.update_perf_metrics` in `py_executor.py:3857` was a no-op.
   - `chat_harmony.create_streaming_generator` never called
     `_extract_metrics`, which is the only place that appends entries to
     `self.perf_metrics`. It also never stamped
     `raw_request.state.server_first_token_time`.

   With both fixed, the harmony chat path produces full per-request
   timing records (`server_arrival_time`, `arrival_time`,
   `first_scheduled_time`, `first_token_time`,
   `server_first_token_time`, `last_token_time`, plus
   `kv_cache_metrics`).

Both fixes are committed:

```text
[None][fix] avoid duplicate harmony tokenization on disagg gen worker
[None][fix] populate /perf_metrics deque for chat endpoints
```

## Setup

| Component | Config |
|---|---|
| Model | `nvidia/gpt-oss-120b` |
| Eagle3 draft | `nvidia/gpt-oss-120b-Eagle3-v3` (max_draft_len=3) |
| GPUs | B200 ×2 (agg: 1; disagg: 1 ctx + 1 gen) |
| Quant | FP8 weights, FP8 KV cache |
| Workload | RWLT (multi-turn agentic coding), conc=1, 5 trajectories, 318 turns total, ~30 min wall |
| RWLT dataset | top-5 trajectories by median TTFT delta from round1 (see `configs/rwlt_round1_top5_signed.yaml`) |

### Configs

The round5 agg, ctx, gen, and proxy yaml configs live under
`configs_0521/round5/`. The key gen-worker knobs are:

```yaml
# configs_0521/round5/repro_disagg_gen_tp1_pp4_si20.yaml
tensor_parallel_size: 1
pipeline_parallel_size: 1
max_batch_size: 128
max_num_tokens: 512
stream_interval: 20            # <-- match across agg/ctx/gen
num_postprocess_workers: 4     # <-- match across agg/ctx/gen
enable_chunked_prefill: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: true
speculative_config:
  decoding_type: Eagle3
  max_draft_len: 3
```

Note: the filename says `pp4_si20`, where `pp4` historically meant
`num_postprocess_workers: 4` and `si20` means `stream_interval: 20`. It
does **not** mean `pipeline_parallel_size=4`. Both agg and ctx run with
the same `num_postprocess_workers: 4` and `stream_interval: 20` in
round5, so the only structural difference is the disagg topology itself.

## Per-stage breakdown (p50, ms)

Computed from `/perf_metrics`. agg stages come from one timing chain;
disagg has two chains (ctx and gen) joined on the proxy.

| Stage | agg r5 | disagg r5 | Δ (disagg − agg) | Notes |
|---|---:|---:|---:|---|
| `srv_preproc` / `ctx_preproc` (tokenize) | 30.2 | 29.8 | −0.4 | paired; cancels |
| `queue` / `ctx_queue` | 11.9 | 12.0 | +0.1 | paired; cancels |
| `prefill` / `ctx_prefill` | 34.1 | 32.9 | −1.2 | paired; ctx prefill ≈ agg prefill |
| `srv_postproc` / `ctx_postproc` | 2.0 | 2.6 | +0.6 | paired; ~equal |
| `proxy_to_ctx` | — | 1.9 | **+1.9** | disagg only |
| `relay_hop` (ctx → gen handoff) | — | 9.9 | **+9.9** | disagg only |
| `gen_preproc` (post-patch) | — | 3.5 | **+3.5** | disagg only |
| `gen_queue` (≈ KV transfer wait) | — | 88.5 | **+88.5** | disagg only ← **dominant** |
| `gen_first_decode` | — | 5.9 | **+5.9** | disagg only |
| `proxy_to_client` | — | 0.5 | **+0.5** | disagg only |
| **sum of disagg-only stages** |  |  | **≈ +110.7** | matches measured +112.4 |

Clock-attribution residual `sum(contribs) − delta_ttft = 0.000 ms` at
all percentiles → full attribution.

## Round4 vs round5 head-to-head

| Metric | r4 agg p50 | r5 agg p50 | r4 disagg p50 | r5 disagg p50 | Δ r4 | Δ r5 |
|---|---:|---:|---:|---:|---:|---:|
| `ttft_server_ms` | 83.1 | 80.9 | 223.8 | 193.2 | +140.7 | **+112.4** |
| `ctx_preproc_ms` | — | — | 30.3 | 29.8 | +30.3 | +29.8 |
| `gen_preproc_ms` | — | — | 32.7 | **3.5** | +32.7 | +3.5 |
| `gen_queue_ms` | — | — | 90.0 | 88.5 | +90.0 | +88.5 |
| `kv_transfer_gen_ms` | — | — | 75.1 | 75.2 | +75.1 | +75.2 |
| `relay_hop_ms` | — | — | 9.7 | 9.9 | +9.7 | +9.9 |

Every line that should not have moved did not move; every line the
retokenization fix targeted (`gen_preproc_ms`) dropped by ~29 ms, and
`ttft_server_ms` improved by the same amount. Clean before/after.

## Worst-trajectory deep dive — `aa-rwlt-coding-agent-056`

Ranked trajectories by `sum(delta_ttft_ms)` across all turns in the
trajectory; the worst is 61-turn `aa-rwlt-coding-agent-056` with a
cumulative gap of +16.62 s.

Top 5 ranking:

| conversation_id | turns | Σ Δ TTFT (ms) | max Δ TTFT (ms) | max ISL |
|---|---:|---:|---:|---:|
| **aa-rwlt-coding-agent-056** | 61 | **16617.6** | 759.0 | 81687 |
| aa-rwlt-coding-agent-010 | 78 | 9236.2 | 252.5 | 69259 |
| aa-rwlt-coding-agent-011 | 63 | 5161.4 | 167.1 | 54232 |
| aa-rwlt-coding-agent-051 | 47 | 5007.8 | 266.3 | 60964 |
| aa-rwlt-coding-agent-073 | 68 | 4602.0 | 269.4 | 51985 |

### Trajectory-level totals

| Quantity | Value |
|---|---:|
| Turns | 61 |
| Total ISL processed | 3.77 M tokens |
| Cached on agg | 3.69 M (97.8%) |
| Cached on disagg (ctx) | 3.69 M (97.8%) |
| Fresh tokens, agg | 81.5 K |
| Fresh tokens, disagg | 81.4 K |
| KV bytes transferred to gen | 3.09 GB |
| Σ `kv_transfer_gen_ms` | 13.92 s |
| Σ `gen_queue_ms` | 15.03 s |
| Σ TTFT (agg) | 7.28 s |
| Σ TTFT (disagg) | 23.90 s |
| **Σ Δ TTFT** | **+16.62 s** |
| Mean Δ per turn | +272 ms |

### What the per-turn table shows

Reproducible via:

```bash
python3 scripts/per_trajectory_report.py \
  --agg-tsv rwlt-results/agg_round5_patched_top5_0521/breakdown.tsv \
  --disagg-tsv rwlt-results/disagg_round5_patched_top5_0521/breakdown.tsv \
  --conversation aa-rwlt-coding-agent-056
```

Three patterns are visible across all 61 turns:

1. **`gen_queue ≈ kv_transfer_gen` turn-by-turn**, always within
   ~15–20 ms (the scheduler dispatch overhead). The two columns are
   measuring the same wall-clock event from different vantage points,
   so eliminating gen-side scheduling latency is essentially the same
   problem as eliminating KV-transfer latency.

2. **Two transfer regimes:**

   | Regime | When | Behaviour |
   |---|---|---|
   | Bandwidth-bound | Cache miss, large new KV (turns 0, 2, 3, 4, 6, 7, 60) | `kv_xfer ≈ KV_MB / 1–7 GB/s` |
   | **Latency-floor** | Cache hit, small new KV (turns 9+, most of the trajectory) | KV is 3–50 MB but transfer takes 170–360 ms → effective bandwidth 0.01–0.1 GB/s |

   The contrast turns 60 vs 10 makes this stark:

   ```text
   turn  fresh_d   KV_MB   kv_xfer    GB/s
     10       63     4.9   358.0ms    0.014
     60       79   354.7     6.3ms   56.0
   ```

   Same fresh-token count (~80), 70× difference in KV size, but turn 60
   transferred 30× more bytes in 1/57 the wall-clock time. So the
   200–360 ms steady-state transfer time is *not* bytes-on-the-wire —
   it is something that lifts when the gen worker is idle, almost
   certainly the cadence of its async loop.

3. **Prefix-cache hit rate is excellent and identical on both sides**
   (97.8% overall, 49/61 turns at ≥99%). The 6 turns with hit% < 99%
   correspond to large tool-output or file-paste insertions
   (turn 5 = 15.5 K fresh tokens, turn 8 = 18.5 K) — these are
   inherent to the agentic workload, not a caching failure. agg and
   disagg achieve the *same* hit rate turn-by-turn (matched to ±1
   token), so the gap is **not** explained by worse caching.

## Hypothesis for the residual gap

The residual ~88 ms p50 (and ~80% of the +16.6 s on this trajectory)
sits on `gen_queue_ms = first_scheduled_time − arrival_time` on the
gen-worker executor. With TP1/PP1 and concurrency 1, there is no
pipeline bubble and no real concurrent decode pressure on the gen
GPU, so "GPU busy decoding other users" cannot explain it.

The likely mechanism, given the steady-state pattern of
`kv_xfer ≈ gen_queue` rising under load and dropping to ~6 ms when the
gen worker is otherwise idle (turn 60):

> The gen worker's main asyncio loop is gated by the
> `stream_interval=20` × `num_postprocess_workers=4` cadence. The
> scheduler only re-examines its queue at iteration boundaries (or
> batches of them), so a new request gated on KV-transfer completion
> can only be picked up at the next scheduling slot. With `si=20` the
> loop accumulates 20 decode steps' worth of tokens before flushing
> through IPC to a postproc worker, and the next scheduling pass
> happens after that flush returns.

`gen_queue_ms` thus measures the wait for the next scheduler pass on
the gen worker, gated by the cadence of decoding the current
in-flight request and the postproc-IPC roundtrip — not the bytes on
the KV transport wire.

## Recommended next experiments

Pre-built configs live under `configs_0521/round5/`; for the
ablations below copy the round5 dir and edit only the noted knob, then
mirror the agg config to keep both layouts comparable.

### Tier A — direct ablations (cheap, fast, single-config changes)

| # | Change (gen only, match on agg) | Tests | Expected if hypothesis is right |
|---|---|---|---|
| **A1** | `stream_interval: 20 → 1` | Whether the SSE cadence drives the queue floor | `gen_queue` drops from 200–360 ms → tens of ms |
| **A2** | `num_postprocess_workers: 4 → 0` (revert to inline postproc) | Whether the postproc-worker IPC fanout is dominant | If A1 doesn't help but A2 does, IPC roundtrip is dominant. Tradeoff: inline postproc may hurt ITL |
| **A3** | Both A1 + A2 | Best-case latency floor | `gen_queue` near one decode-step (~10 ms) |

**Recommended first experiment: A1.** One-line config change, cleanly
separates cadence-vs-IPC. If `gen_queue_ms` p50 collapses from ~90 ms
and the steady-state 250 ms turns drop to tens of ms → cadence is the
cause. If it doesn't move → run A2.

### Tier B — measurement only (no behaviour change)

| # | Change | Why |
|---|---|---|
| **B1** | NVTX-instrument the gen executor around `iteration_step`, `scheduler.schedule`, `postproc.flush`, and IPC `send/recv`; nsys-profile 60 s of steady-state | Directly observe whether the loop is GPU-busy or IPC-blocked during queue-wait windows |
| **B2** | Add a counter at `arrival_time` stamp recording (a) executor queue depth and (b) time since last scheduler pass; log per request | Cheaper than nsys; quantifies "arrived just after a scheduler pass" vs "scheduler is genuinely behind" |

### Tier C — orthogonal levers (only after A clarifies)

| # | Change | Why |
|---|---|---|
| **C1** | Disable Eagle3 spec on gen | Spec-decode iterations are heavier/more variable; could pad the cadence enough to raise the floor |
| **C2** | `enable_chunked_prefill: false` on gen | Gen has no real prefill in disagg; confirm chunked-prefill scheduling phase isn't adding cost |
| **C3** | Multiple gen workers + `kv_cache_aware` routing with sticky conversation_id | Steady-state turns reuse the gen pool — `kv_transfer_gen_ms → 0`. Doesn't address the scheduling floor on the "first" turn |

## Reproducing the round5 result

The repo is wired so a fresh round5 reproduction is a 4-command
sequence, assuming you're running on a B200 box with 2+ GPUs visible
and a checked-out copy of this branch:

```bash
# Build / install the patched tensorrt_llm (these commits land
# the two openai_server.py fixes -- they MUST be present, or
# perf_metrics will be empty and gen_preproc will not collapse):
git log --oneline -3 -- tensorrt_llm/serve/openai_server.py

cd repro-gptoss-ttft

# 1. Agg
CUDA_VISIBLE_DEVICES=0 \
  CONFIG_DIR=configs_0521/round5 \
  scripts/run_session.sh agg rwlt_round1_top5_signed agg_round5

# 2. Disagg (ctx GPU 0, gen GPU 1)
CTX_GPU=0 GEN_GPU=1 \
  CONFIG_DIR=configs_0521/round5 \
  GEN_CONFIG_BASE=repro_disagg_gen_tp1_pp4_si20 \
  scripts/run_session.sh disagg rwlt_round1_top5_signed disagg_round5

# 3. Per-stage TTFT breakdown
python3 scripts/perf_metrics_breakdown.py \
  rwlt-results/agg_round5/perf_metrics.json \
  --rwlt rwlt-results/agg_round5/rwlt_requests.jsonl \
  --label agg --out rwlt-results/agg_round5/breakdown.tsv

python3 scripts/perf_metrics_breakdown.py \
  rwlt-results/disagg_round5/perf_metrics_proxy.json \
  --rwlt rwlt-results/disagg_round5/rwlt_requests.jsonl \
  --label disagg --disagg \
  --out rwlt-results/disagg_round5/breakdown.tsv

# 4. Side-by-side diff (per-turn TSV + summary)
python3 scripts/diff_perf_breakdown.py \
  rwlt-results/agg_round5/breakdown.tsv \
  rwlt-results/disagg_round5/breakdown.tsv \
  --label-a agg --label-b disagg \
  --per-turn-out rwlt-results/diff_round5.tsv
```

Single-trajectory deep dive (any conversation):

```bash
python3 scripts/per_trajectory_report.py \
  --agg-tsv rwlt-results/agg_round5/breakdown.tsv \
  --disagg-tsv rwlt-results/disagg_round5/breakdown.tsv \
  --conversation aa-rwlt-coding-agent-056
```

The RWLT subset config used here
(`configs/rwlt_round1_top5_signed.yaml`) points at a pre-filtered
dataset under `configs/datasets/round1_top5_signed.jsonl`. That JSONL
is too large for git (~57 MB); regenerate it from the source dataset
with `scripts/make_topk_subset.py`:

```bash
python3 scripts/make_topk_subset.py \
  --diff rwlt-results/diff_round1_0521.tsv \
  --k 5 --metric median_abs_delta \
  --label round1_top5_signed
```

(Requires the source `aa-rwlt_coding-agent-scenario_tuning_v2.jsonl`
dataset listed in the round1 diff TSV header.)

## File map for the next person picking this up

```text
repro-gptoss-ttft/
├── configs_0521/round5/                      # current best-baseline server configs
│   ├── repro_agg_tp1_eagle3.yaml
│   ├── repro_disagg_ctx_tp1.yaml
│   ├── repro_disagg_gen_tp1_pp4_si20.yaml    # npp=4, si=20 -- match across all 3
│   └── repro_disagg_proxy.yaml
├── configs/
│   ├── rwlt_round1_top5_signed.yaml          # RWLT client config (5 trajectories)
│   └── datasets/                             # NOT committed; regen with make_topk_subset.py
├── scripts/
│   ├── launch_agg.sh / launch_disagg.sh      # background launchers + perf_metrics env-var
│   ├── run_session.sh                        # end-to-end launch + drain + teardown
│   ├── drain_perf_metrics.sh                 # snapshots /perf_metrics ONCE (destructive read)
│   ├── perf_metrics_breakdown.py             # /perf_metrics JSON -> per-turn breakdown TSV
│   ├── diff_perf_breakdown.py                # agg-vs-disagg TSV diff w/ stage attribution
│   ├── per_trajectory_report.py              # per-conversation per-turn detail
│   ├── make_topk_subset.py                   # build top-K trajectory subset from a diff TSV
│   └── verify_subset_against_baseline.py     # sanity check subset is a prefix of baseline
└── rwlt-results/
    ├── FINDINGS_RWLT.md                      # earlier writeup (round 1-2 era)
    ├── FINDINGS_RWLT_0521.md                 # this document
    ├── agg_round5_patched_top5_0521/         # raw r5 agg result snapshot
    ├── disagg_round5_patched_top5_0521/      # raw r5 disagg result snapshot
    └── diff_round5_patched_top5_0521.tsv     # joined per-turn diff
```

## Open questions / known gaps

- The `gen_queue_ms` ≈ `kv_transfer_gen_ms` correspondence is
  consistent at single-trajectory level, but we have not yet
  instrumented the gen executor's scheduler-pass cadence directly.
  Tier-B (NVTX + queue-depth logging) is the right next measurement
  step if Tier-A ablations don't immediately confirm the hypothesis.
- `gen_first_decode_ms` is 5.9 ms in disagg but the ctx-side
  equivalent is effectively 0. That's a clean +6 ms wall clock not
  explained by KV transfer; smells like first-step-on-fresh-slot
  overhead on the gen pool. Worth a small investigation independent
  of the Tier-A work.
- The top-level `server_arrival_time` / `server_first_token_time`
  fields on `/perf_metrics` records are `None` for chat requests
  because `_extract_metrics` reads `raw_request.state.server_*_time`,
  which only the ASGI middleware writes. They're populated correctly
  inside `perf_metrics.timing_metrics`, which is what
  `perf_metrics_breakdown.py` reads — so this is harmless for our
  analysis, but worth a cleanup PR.
- Eagle3 spec acceptance is RNG-state dependent per executor instance;
  once a turn produces a slightly different assistant message the rest
  of the conversation diverges in output length but not in TTFT (we
  pair on `(conversation_id, conversation_idx)`).

---

## Round 6 update (later 2026-05-21): UCX & stream_interval ablations

Round 5 identified the residual disagg gap as `gen_queue_ms ≈
kv_transfer_gen_ms` (~75-90 ms p50) and proposed three Tier-A ablations
to attribute it. Round 6 ran two of those ablations
(UCX backend, `stream_interval=1`) and produced a sharper hypothesis
about where the remaining wait actually lives. The third
(`num_postprocess_workers=0`) is in flight.

### Headline results

| run | p50 client TTFT (ms) | p90 (ms) | mean Δ vs agg (ms) | p50 Δ (ms) |
|---|---:|---:|---:|---:|
| agg (baseline) | 180.4 | 233.3 | — | — |
| disagg ref (NIXL, si=20, npp=4) | 288.8 | 493.1 | 127.6 | 106.0 |
| disagg UCX backend | 292.7 | 575.3 | 147.1 | 112.9 |
| **disagg si=1** | **210.0** | **404.3** | **52.4** | **28.4** |

`stream_interval=1` recovers ~75-80 ms median Δ TTFT across all
trajectories (uniform improvement: 4/5 trajectories drop to near
parity; agent-073 actually beats agg). The byte-side `Transmissions`
time from the transceiver CSV is identical between `ref` and `si1`
(p50 = 73 ms in both), so the savings come entirely from the
post-transceiver path — specifically the SSE emission cadence.

### Two hypotheses falsified, one bug identified

1. **"NIXL is adding overhead on top of UCX"** — falsified. UCX is
   worse than NIXL at every percentile (transceiver `Transmissions`
   p90: NIXL 214 ms vs UCX 307 ms, +43%; p99: 349 vs 511 ms). NIXL is
   the right default. The default `cache_transceiver_config.backend:
   DEFAULT` (= NIXL with UCX sub-backend) is faster than `backend: UCX`
   on this workload.

2. **"stream_interval matters only for inter-token latency, not TTFT"**
   — falsified. With overlap scheduler ON (the default), the
   `py_decoding_iter == 1` first-token fast-path in `_handle_responses`
   is shadowed by an earlier `continue` (`py_executor.py:3909-3917`):

   ```text
   iter N:    promote (TRANS_COMPLETE -> GENERATION_IN_PROGRESS,
              py_decoding_iter=1).  Early-continue skips response emit.
   iter N+1:  py_decoding_iter still 1 (or 2 with overlap).
              Early-continue still skips OR response gate sees
              decoding_iter=2.
   iter N+2..N+19 (stream_interval=20):
              decoding_iter % 20 != 0 -> NO SSE frame leaves gen.
   iter N+20: first SSE frame -> proxy -> client.
   ```

   ~19 iters × ~5 ms/iter ≈ 95 ms of delay before the first token
   crosses to the client. Measured saving from `si=20` -> `si=1` is
   75-80 ms (consistent; small gap is SSE+proxy hop).

   **Real bug, not just a tuning lever.** The intended behavior is for
   `py_decoding_iter == 1` to bypass `stream_interval`, but the
   overlap-scheduler short-circuit hides it. Fix candidates: hoist the
   first-token emission to before the overlap-early-continue, or
   teach the early-continue to let `py_decoding_iter == 1 OR
   is_finished` through.

3. **"gen worker is busy decoding while waiting"** — partially
   falsified. The transceiver `Transmissions` time (the actual byte
   transfer wall-clock as the cache transceiver measures it) tracks
   `kv_xfer_ms` from `/perf_metrics` almost exactly (NIXL p50=73 ms
   vs perf_metrics p50=75 ms). So `wait_ms ≈ kv_xfer_ms ≈
   Transmissions` — the wait is genuinely *inside the transceiver*,
   not in some pre-transfer scheduling gap. The gen worker isn't
   "busy"; it's waiting on the transceiver to finish.

### New puzzle: per-block transceiver overhead on agent-056

After si=1, four trajectories collapse to ±35 ms median Δ. One
trajectory (agent-056) still carries +190 ms median residual:

| conv | n | NIXL ref Δ | UCX Δ | **si=1 Δ** |
|---|---:|---:|---:|---:|
| agent-010 | 79 | +93.1 | +101.7 | **+11.8** |
| agent-011 | 63 | +83.9 | +89.0 | **+3.2** |
| agent-051 | 47 | +111.1 | +112.0 | **+32.2** |
| **agent-056** | 61 | **+262.4** | **+364.2** | **+190.0** |
| agent-073 | 68 | +64.2 | +67.7 | **−15.7** |

agent-056 is unusual: 60-80k reused tokens per turn (vs 30-45k for
others), small fresh delta per turn (often <500 tokens). The
transceiver `Transmissions` time on its worst rows shows **effective
bandwidth of 0.09-1 Gbps** (NVLink should support 500+ Gbps):

```text
turn  fresh  reused   kv_xfer  kv_bw      delta(si=1)
 60     83   81603    332.7    0.09 Gbps  +305.6
 44     83   75966    316.7    0.10 Gbps  +285.1
 47    112   76677    278.3    0.15 Gbps  +252.1
 54    122   78225    285.8    0.18 Gbps  +260.1
```

A 83-token fresh delta should not take 332 ms to transfer. Strong
evidence of per-block fixed overhead in the cache transceiver: the
trajectory has many reused blocks, and even when only a small number
are *new* to gen, the transceiver appears to scale with #blocks
rather than #bytes. Next: instrument the transceiver to log
`num_blocks_transferred` alongside `Transmissions` and check
`Transmissions / num_blocks` on agent-056 vs the rest.

### Diagnostic instrumentation landed this round

Gated by `TRTLLM_DEBUG_DISAGG_GEN_TIMING=1`
(default-off, allocation-free when off):

- `tensorrt_llm/_torch/pyexecutor/llm_request.py`: 5 `py_dbg_*`
  timestamp fields on `LlmRequest` for arrival / xfer_init /
  xfer_done / promote.
- `tensorrt_llm/_torch/pyexecutor/py_executor.py`:
  `_disagg_dbg_enabled()` gate, one-shot startup banner (so the user
  can confirm the env var reached the MPI-spawned worker), per-request
  `[disagg-dbg] req=<id> ...` emit at first-token time with a
  five-component breakdown of `gen_queue_ms`. Emit point is **before**
  the overlap-early-continue (which is where the original placement
  silently dropped the log; see fix in commit log).
- The env var is forwarded automatically because
  `MpiPoolSession._start_mpi_pool` allowlists `TRTLLM*` / `TLLM*`
  prefixes.

The instrumentation was not ultimately needed to draw the round 6
conclusions (the transceiver CSV proved sufficient), but it's
permanent code that will be useful for follow-up gen-side investigations.

### Repro safety: drain-on-teardown

`scripts/stop_disagg.sh` now drains `/perf_metrics` automatically
before killing the workers. Reading `/perf_metrics` is destructive
(pops the worker deques), so this MUST run exactly once per session.
The liveness probe uses `/health` (idempotent), NOT `/perf_metrics`
— hitting `/perf_metrics` as a probe was a bug in an earlier
iteration that silently discarded an entire run's records.

If you forget the `DRAIN_OUT_DIR=...` env var, the script falls back
to `LOG_DIR` (which `launch_disagg.sh` sets per-run).

### Per-request side-by-side joiner

`scripts/per_request_side_by_side.py` joins `rwlt_requests.jsonl` and
`/perf_metrics_proxy.json` from a paired (agg, disagg) session and
emits one row per request with `(fresh, reused, kv_xfer_ms, wait_ms,
ttft_agg, ttft_disagg, delta_ttft)`. Pair by `(conversation_id,
conversation_idx)` after sorting each session's rwlt rows by
`start_time` (valid at concurrency=1).

Outputs:
- `rwlt-results/side_by_side_round5.tsv` (317 rows, joined from
  `agg_round5_patched_top5_0521` and `disagg_round5_patched_top5_0521`).
- `rwlt-results/side_by_side_si1.tsv` (318 rows, joined using
  transceiver CSV in place of `/perf_metrics` because that run's drain
  was lost to the earlier bug; the join key is row index since both
  files are written in completion order at concurrency=1).

### File map additions (round 6)

```text
repro-gptoss-ttft/
├── configs_0521/round6/                        # round 6 ablation configs
│   ├── repro_agg_tp1_eagle3.yaml               # = round5 agg (unchanged)
│   ├── repro_disagg_ctx_tp1.yaml               # = round5 ctx (unchanged)
│   ├── repro_disagg_ctx_tp1_ucx.yaml           # ctx with UCX backend
│   ├── repro_disagg_gen_tp1_pp4_si20_ucx.yaml  # gen UCX (ablation A)
│   ├── repro_disagg_gen_tp1_pp4_si1.yaml       # gen si=1   (ablation B)
│   ├── repro_disagg_gen_tp1_pp0_si20.yaml      # gen npp=0  (ablation C)
│   └── repro_disagg_proxy.yaml
├── scripts/
│   ├── per_request_side_by_side.py             # (new) per-request joiner
│   ├── launch_disagg.sh                        # (updated) defaults
│                                               # TRTLLM_DEBUG_DISAGG_GEN_TIMING=1
│   └── stop_disagg.sh                          # (updated) auto-drains
│                                               # /perf_metrics (uses /health
│                                               # for liveness)
└── rwlt-results/
    ├── disagg_round6_ref/                      # NIXL backend baseline (re-run)
    ├── disagg_round6_ucx/                      # ablation A: UCX backend
    ├── disagg_round6_si1/                      # ablation B: stream_interval=1
    ├── side_by_side_round5.tsv                 # joined per-request table (round 5)
    └── side_by_side_si1.tsv                    # joined per-request table (si=1)
```

### Recommendations

1. **Ship `stream_interval=1` as the disagg gen default for low-latency
   workloads.** Uniform 75-80 ms p50 TTFT win at concurrency 1. Worth
   re-validating at higher concurrency since si=1 means more SSE frames
   per second (more postproc IPC traffic), but at conc=1 it's strictly
   better with no measured downside.

2. **File a bug for the overlap-scheduler shadowing the
   `py_decoding_iter == 1` first-token fast-path.** This is independent
   of disagg; agg with overlap also pays the same penalty whenever
   `stream_interval > 1`. One-line fix candidate: hoist the
   `py_decoding_iter == 1` clause out of (or above) the overlap
   early-continue in `_handle_responses`.

3. **Drop UCX as an ablation direction.** NIXL is uniformly faster on
   this workload — no further investigation needed there.

4. **Next investigation: per-block transceiver overhead on long-context
   trajectories.** agent-056's residual Δ at si=1 is the clearest
   remaining attribution gap. Measure `Transmissions / num_blocks` to
   either confirm the per-block hypothesis or rule it out.

5. **Bake drain-on-teardown into all future scripts** that touch
   `/perf_metrics`. The endpoint's destructive read semantics are easy
   to violate by accident (we lost two runs' worth of data to this).
   The new `stop_disagg.sh` and the `/health` liveness check guard
   against the most common form of the bug.
