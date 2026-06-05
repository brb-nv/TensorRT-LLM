# Round1 — native trtllm-serve reproduction of NVBug 6266370

[NVBug 6266370](https://nvbugspro.nvidia.com/bug/6266370) (TRTLLM-13167):
*"GPT-OSS B200 Dynamo/TRT-LLM disagg: high p95 TTFT at 3P TP1 / 2D TP2 C=160
dominated by TRT-LLM prefill engine time."*

The original run was **Dynamo + TRT-LLM**. This round reproduces the same
topology and workload **natively with `trtllm-serve` (disaggregated) + the
RWLT client only** — no Dynamo, no sflow, no nsys — so the prefill-engine TTFT
tail can be attributed to the TRT-LLM engine path alone.

## What the bug reports

p95 TTFT breakdown from the Dynamo `[TTFT-TRACE]` at C=160:

| stage | p95 |
|---|---:|
| frontend TTFT | 4.943 s |
| prefill wait | 0.020 s |
| **prefill engine** | **4.873 s** |
| KV transfer | 0.096 s |
| decode engine | 0.078 s |

Prefill-engine time grows sharply with ctx in-flight count (engine p50 ~0.10 s
when ≤2 active → ~4.80 s when >48 active). The tail is almost entirely inside
the TRT-LLM prefill engine. **Reproduced** if native p95 TTFT is likewise
dominated by ctx/prefill engine time while wait, KV transfer, and decode stay
small.

## Topology (1× 8-GPU B200 node)

| process | role | GPU(s) | port | config |
|---|---|---|---|---|
| ctx0 | CONTEXT (TP1) | 0 | 8001 | `configs/ctx_tp1.yaml` |
| ctx1 | CONTEXT (TP1) | 1 | 8002 | `configs/ctx_tp1.yaml` |
| ctx2 | CONTEXT (TP1) | 2 | 8003 | `configs/ctx_tp1.yaml` |
| gen0 | GENERATION (TP2) | 3,4 | 8004 | `configs/gen_tp2.yaml` |
| gen1 | GENERATION (TP2) | 5,6 | 8005 | `configs/gen_tp2.yaml` |
| proxy | disagg router | — | 8000 | `configs/proxy_3ctx2gen.yaml` |

3 ctx (TP1) + 2 gen (TP2) = 7 GPUs; GPU 7 idle. Both pools use the
`kv_cache_aware` router (native equivalent of the bug's `dynamo_kvrouter`).

## Configs — provenance

`configs/ctx_tp1.yaml` and `configs/gen_tp2.yaml` are faithful copies of the
**runtime** configs that produced the bug (`../from_pei/.../ctx.yaml`,
`gen.yaml`), including the fact that the Dynamo run **disabled torch.compile on
both roles** (`REMOVE_TORCH_COMPILE_CTX=1`, `REMOVE_TORCH_COMPILE_GEN=1`) and
ran the ctx workers with a 128 GiB **host KV cache** (`HOSTCACHE=1`).
`configs/rwlt_c160.yaml` mirrors `../from_pei/.../rwlt_config.yaml` (C=160,
240 s settling, 900 s measurement, 30 trajectories/user, reasoning high,
routing headers on).

The only intentional deltas from the bug run are **observability-only**:
`return_perf_metrics: true` + `*_max_*` knobs on the workers so the proxy
`/perf_metrics` endpoint yields a per-stage TTFT breakdown.

## How to run (on the B200 node)

```bash
cd high_conc_ttft/round1

# One-shot: launch -> RWLT C=160 -> drain /perf_metrics -> teardown.
# Server logs + perf_metrics_proxy.json land in rwlt-results/c160/.
scripts/run_session.sh c160

# Or drive the pieces manually:
scripts/launch_disagg_3ctx2gen.sh                 # 5 workers + proxy
scripts/run_rwlt.sh c160 http://localhost:8000/v1 rwlt_c160
scripts/stop_disagg.sh                            # drains /perf_metrics, kills all
```

Outputs (under `rwlt-results/c160/`):
- `rwlt_requests.jsonl` — per-request client log (TTFT, TPOT, E2EL, ...).
- `openai_gpt-oss-120b__160u__phase0__*.json/.txt` — RWLT phase summary.
- `perf_metrics_proxy.json` — joined ctx+gen server-side stage breakdown.
- `ctx*.log`, `gen*.log`, `proxy.log` — server logs (`print_iter_log` on).

## Paths used (defaults) — please confirm

| what | round1 default | bug run (lustre, **not reachable here**) |
|---|---|---|
| model | `/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b` | `/lustre/.../models--openai--gpt-oss-120b/snapshots/b5c939...` |
| eagle draft | `/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3` | `/lustre/.../models--nvidia--gpt-oss-120b-Eagle3-next/snapshots/975baa...` |
| RWLT dataset | `/home/scratch.shobhitv_coreai/aa-rwlt-shared-datasets/aa-rwlt_coding-agent-scenario_tuning_v2_500traj.jsonl` | `/lustre/.../data/agentic_coding_v2_full.jsonl` |
| RWLT client | `/home/scratch.bbuddharaju_gpu/artificial-analysis` (`rwlt` pkg) | same repo on lustre |

**Resolved (confirmed by owner):**

1. **Dataset.** The bug used `agentic_coding_v2_full.jsonl` on lustre (not
   mounted here). Using the local coding-agent v2 **500-trajectory** set for
   trajectory variety at C=160 (`trajectories_per_user=30`), which avoids the
   artificial KV reuse the 100-traj set would cause. Note this is still a
   different file from the bug's `..._full`, so treat absolute TTFT as
   directionally comparable rather than 1:1.
2. **Eagle checkpoint.** Bug's `Eagle3-next` → local released `Eagle3-v3`
   (same model, pre-release vs released name). Confirmed OK.
3. **Node.** Single B200 node with ≥7 GPUs (3×TP1 + 2×TP2). Confirmed fine;
   override `CTX_GPUS` / `GEN_GPU_GROUPS` only if the allocation changes.
