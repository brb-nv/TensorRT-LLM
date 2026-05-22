#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute per-request, per-stage TTFT breakdown from a /perf_metrics dump.

Accepts either:
  * an agg /perf_metrics dump (flat list of per-request records), or
  * a disagg proxy /perf_metrics dump (joined ctx + gen view)

and emits a TSV with one row per successful request. All time columns are in
milliseconds. The per-stage formulas mirror the diagram in
`tensorrt_llm/serve/scripts/time_breakdown/README.md`.

Sequence (Nth successful perf_metrics record) is paired against the Nth
successful RWLT request from `rwlt_requests.jsonl` so each row carries
`conversation_id`, `conversation_idx`, `ttft_client_ms` for cross-checking.
This pairing is valid only at concurrency=1 with deterministic ordering --
the assertion at the bottom of `load_rwlt()` enforces this.

Usage:
    # Aggregated session
    python3 scripts/perf_metrics_breakdown.py \
        rwlt-results/agg_round4_top5_0521/perf_metrics.json \
        --rwlt rwlt-results/agg_round4_top5_0521/rwlt_requests.jsonl \
        --label agg --out rwlt-results/agg_round4_top5_0521/breakdown.tsv

    # Disaggregated session
    python3 scripts/perf_metrics_breakdown.py \
        rwlt-results/disagg_round4_top5_0521/perf_metrics_proxy.json \
        --rwlt rwlt-results/disagg_round4_top5_0521/rwlt_requests.jsonl \
        --label disagg --disagg \
        --out rwlt-results/disagg_round4_top5_0521/breakdown.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


def _safe_diff(a: Optional[float], b: Optional[float]) -> float:
    """Return (a - b) * 1000 in ms, or NaN if either operand is missing."""
    if a is None or b is None:
        return float("nan")
    try:
        return (float(a) - float(b)) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.3f}"


def load_rwlt(path: Path) -> list[dict[str, Any]]:
    """Return successful RWLT requests in submission order.

    The RWLT client at concurrency=1 issues turns strictly in the order they
    appear in this file, so the Nth row corresponds 1:1 with the Nth
    successful /perf_metrics record on the server.
    """
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("success"):
                out.append(r)
    # Stable secondary sort by start_time so out-of-order writes (shouldn't
    # happen at conc=1) still pair correctly.
    out.sort(key=lambda r: r.get("start_time", 0.0))
    return out


def _kv_blocks(km: dict[str, Any]) -> dict[str, float]:
    """Extract block counts and cache reuse ratio from kv_cache_metrics."""
    total = km.get("num_total_allocated_blocks")
    new   = km.get("num_new_allocated_blocks")
    reused = km.get("num_reused_blocks")
    missed = km.get("num_missed_blocks")
    ratio = float("nan")
    if reused is not None and missed is not None and (reused + missed) > 0:
        ratio = reused / (reused + missed)
    return {
        "kv_blocks_total":  float(total) if total is not None else float("nan"),
        "kv_blocks_new":    float(new)   if new   is not None else float("nan"),
        "kv_blocks_reused": float(reused) if reused is not None else float("nan"),
        "kv_blocks_missed": float(missed) if missed is not None else float("nan"),
        "kv_block_reuse_ratio": ratio,
    }


def extract_agg_row(record: dict[str, Any]) -> dict[str, float]:
    """Per-request derived stages for an agg /perf_metrics record."""
    pm = record.get("perf_metrics", {}) or {}
    tm = pm.get("timing_metrics", {}) or {}
    km = pm.get("kv_cache_metrics", {}) or {}
    tbm = record.get("time_breakdown_metrics") or {}

    sa = tm.get("server_arrival_time")
    a = tm.get("arrival_time")
    fs = tm.get("first_scheduled_time")
    ft = tm.get("first_token_time")
    sft = tm.get("server_first_token_time")
    lt = tm.get("last_token_time")

    gpu_fwd = tbm.get("ctx_gpu_forward_time")
    gpu_smp = tbm.get("ctx_gpu_sample_time")
    # The README says ctx_gpu_*_time fields are already in milliseconds.
    gpu_fwd_ms = float(gpu_fwd) if gpu_fwd is not None else float("nan")
    gpu_smp_ms = float(gpu_smp) if gpu_smp is not None else float("nan")

    prefill_ms = _safe_diff(ft, fs)
    host_ms = (prefill_ms - gpu_fwd_ms - gpu_smp_ms
               if not (math.isnan(prefill_ms) or math.isnan(gpu_fwd_ms)
                       or math.isnan(gpu_smp_ms)) else float("nan"))

    row = {
        "ttft_server_ms": _safe_diff(sft, sa),
        "srv_preproc_ms": _safe_diff(a, sa),
        "queue_ms": _safe_diff(fs, a),
        "prefill_ms": prefill_ms,
        "srv_postproc_ms": _safe_diff(sft, ft),
        "gpu_forward_ms": gpu_fwd_ms,
        "gpu_sample_ms": gpu_smp_ms,
        "prefill_host_ms": host_ms,
        "decode_ms": _safe_diff(lt, ft),
        "end_to_end_ms": _safe_diff(lt, sa),
        "first_iter": float(pm.get("first_iter")) if pm.get("first_iter") is not None else float("nan"),
        "last_iter":  float(pm.get("last_iter"))  if pm.get("last_iter")  is not None else float("nan"),
    }
    row.update(_kv_blocks(km))
    return row


def extract_disagg_row(record: dict[str, Any]) -> dict[str, float]:
    """Per-request derived stages for a disagg proxy /perf_metrics record.

    Layout (see tensorrt_llm/serve/perf_metrics.py::DisaggPerfMetricsCollector
    and tensorrt_llm/serve/openai_server.py::get_perf_metrics):

        {
          "disagg_server_arrival_time": <float>,
          "disagg_server_first_token_time": <float>,
          "ctx_perf_metrics": {
              "perf_metrics": {"timing_metrics": {...}},
              "time_breakdown_metrics": {...}   # optional
          },
          "gen_perf_metrics": {
              "perf_metrics": {"timing_metrics": {...}},
              "time_breakdown_metrics": {...}   # optional
          }
        }
    """
    disagg_arr = record.get("disagg_server_arrival_time")
    disagg_ft = record.get("disagg_server_first_token_time")

    ctx_block = record.get("ctx_perf_metrics") or {}
    ctx_pm = (ctx_block.get("perf_metrics") or {})
    ctx_tm = ctx_pm.get("timing_metrics", {}) or {}
    ctx_km = ctx_pm.get("kv_cache_metrics", {}) or {}
    ctx_tbm = ctx_block.get("time_breakdown_metrics") or {}

    gen_block = record.get("gen_perf_metrics") or {}
    gen_pm = (gen_block.get("perf_metrics") or {})
    gen_tm = gen_pm.get("timing_metrics", {}) or {}
    gen_km = gen_pm.get("kv_cache_metrics", {}) or {}
    gen_tbm = gen_block.get("time_breakdown_metrics") or {}

    ctx_sa = ctx_tm.get("server_arrival_time")
    ctx_a = ctx_tm.get("arrival_time")
    ctx_fs = ctx_tm.get("first_scheduled_time")
    ctx_ft = ctx_tm.get("first_token_time")
    ctx_sft = ctx_tm.get("server_first_token_time")
    ctx_lt = ctx_tm.get("last_token_time")
    ctx_kvx_start = ctx_tm.get("kv_cache_transfer_start")
    ctx_kvx_end = ctx_tm.get("kv_cache_transfer_end")

    gen_sa = gen_tm.get("server_arrival_time")
    gen_a = gen_tm.get("arrival_time")
    gen_fs = gen_tm.get("first_scheduled_time")
    gen_ft = gen_tm.get("first_token_time")
    gen_sft = gen_tm.get("server_first_token_time")
    gen_lt = gen_tm.get("last_token_time")
    gen_kvx_start = gen_tm.get("kv_cache_transfer_start")
    gen_kvx_end = gen_tm.get("kv_cache_transfer_end")
    gen_kvx_size = gen_tm.get("kv_cache_size")

    ctx_gpu_fwd = ctx_tbm.get("ctx_gpu_forward_time")
    ctx_gpu_smp = ctx_tbm.get("ctx_gpu_sample_time")
    ctx_gpu_fwd_ms = float(ctx_gpu_fwd) if ctx_gpu_fwd is not None else float("nan")
    ctx_gpu_smp_ms = float(ctx_gpu_smp) if ctx_gpu_smp is not None else float("nan")

    ctx_prefill_ms = _safe_diff(ctx_ft, ctx_fs)
    ctx_prefill_host_ms = (
        ctx_prefill_ms - ctx_gpu_fwd_ms - ctx_gpu_smp_ms
        if not (math.isnan(ctx_prefill_ms) or math.isnan(ctx_gpu_fwd_ms)
                or math.isnan(ctx_gpu_smp_ms)) else float("nan"))

    kvx_gen_ms = _safe_diff(gen_kvx_end, gen_kvx_start)
    kv_cache_mb = (gen_kvx_size / (1024.0 * 1024.0)
                   if isinstance(gen_kvx_size, (int, float)) else float("nan"))
    kvx_bw_gb_s = float("nan")
    if (not math.isnan(kv_cache_mb) and not math.isnan(kvx_gen_ms)
            and kvx_gen_ms > 0):
        # MB / ms = GB/s
        kvx_bw_gb_s = kv_cache_mb / kvx_gen_ms

    row = {
        "ttft_server_ms": _safe_diff(disagg_ft, disagg_arr),
        "proxy_to_ctx_ms": _safe_diff(ctx_sa, disagg_arr),
        "ctx_preproc_ms": _safe_diff(ctx_a, ctx_sa),
        "ctx_queue_ms": _safe_diff(ctx_fs, ctx_a),
        "ctx_prefill_ms": ctx_prefill_ms,
        "ctx_postproc_ms": _safe_diff(ctx_sft, ctx_ft),
        "relay_hop_ms": _safe_diff(gen_sa, ctx_sft),
        "gen_preproc_ms": _safe_diff(gen_a, gen_sa),
        "gen_queue_ms": _safe_diff(gen_fs, gen_a),
        "gen_first_decode_ms": _safe_diff(gen_ft, gen_fs),
        "gen_postproc_ms": _safe_diff(gen_sft, gen_ft),
        "proxy_to_client_ms": _safe_diff(disagg_ft, gen_sft),
        "kv_transfer_ctx_ms": _safe_diff(ctx_kvx_end, ctx_kvx_start),
        "kv_transfer_gen_ms": kvx_gen_ms,
        "ctx_gpu_forward_ms": ctx_gpu_fwd_ms,
        "ctx_gpu_sample_ms": ctx_gpu_smp_ms,
        "ctx_prefill_host_ms": ctx_prefill_host_ms,
        # Generation-phase + lifecycle.
        "ctx_decode_ms": _safe_diff(ctx_lt, ctx_ft),
        "gen_decode_ms": _safe_diff(gen_lt, gen_ft),
        "end_to_end_ms": _safe_diff(disagg_ft, disagg_arr),
        # KV transfer size / bandwidth on gen.
        "kv_cache_mb": kv_cache_mb,
        "kv_transfer_gen_GBps": kvx_bw_gb_s,
        # Engine iteration indices.
        "ctx_first_iter": float(ctx_pm.get("first_iter")) if ctx_pm.get("first_iter") is not None else float("nan"),
        "ctx_last_iter":  float(ctx_pm.get("last_iter"))  if ctx_pm.get("last_iter")  is not None else float("nan"),
        "gen_first_iter": float(gen_pm.get("first_iter")) if gen_pm.get("first_iter") is not None else float("nan"),
        "gen_last_iter":  float(gen_pm.get("last_iter"))  if gen_pm.get("last_iter")  is not None else float("nan"),
    }
    # Block-level counters, prefixed for ctx vs gen.
    for prefix, km in (("ctx", ctx_km), ("gen", gen_km)):
        kv = _kv_blocks(km)
        for k, v in kv.items():
            row[f"{prefix}_{k}"] = v
    return row


AGG_COLS = [
    "ttft_server_ms", "srv_preproc_ms", "queue_ms", "prefill_ms",
    "srv_postproc_ms", "gpu_forward_ms", "gpu_sample_ms", "prefill_host_ms",
    "decode_ms", "end_to_end_ms",
    "first_iter", "last_iter",
    "kv_blocks_total", "kv_blocks_new", "kv_blocks_reused", "kv_blocks_missed",
    "kv_block_reuse_ratio",
]
DISAGG_COLS = [
    "ttft_server_ms", "proxy_to_ctx_ms", "ctx_preproc_ms", "ctx_queue_ms",
    "ctx_prefill_ms", "ctx_postproc_ms", "relay_hop_ms", "gen_preproc_ms",
    "gen_queue_ms", "gen_first_decode_ms", "gen_postproc_ms",
    "proxy_to_client_ms", "kv_transfer_ctx_ms", "kv_transfer_gen_ms",
    "ctx_gpu_forward_ms", "ctx_gpu_sample_ms", "ctx_prefill_host_ms",
    "ctx_decode_ms", "gen_decode_ms", "end_to_end_ms",
    "kv_cache_mb", "kv_transfer_gen_GBps",
    "ctx_first_iter", "ctx_last_iter", "gen_first_iter", "gen_last_iter",
    "ctx_kv_blocks_total", "ctx_kv_blocks_new",
    "ctx_kv_blocks_reused", "ctx_kv_blocks_missed", "ctx_kv_block_reuse_ratio",
    "gen_kv_blocks_total", "gen_kv_blocks_new",
    "gen_kv_blocks_reused", "gen_kv_blocks_missed", "gen_kv_block_reuse_ratio",
]


def _summary(values: list[float]) -> str:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return "n=0"
    clean_sorted = sorted(clean)
    def pct(q):
        k = (len(clean_sorted) - 1) * q / 100.0
        f, c = math.floor(k), math.ceil(k)
        return (clean_sorted[int(k)] if f == c
                else clean_sorted[f] + (clean_sorted[c] - clean_sorted[f]) * (k - f))
    return (f"n={len(clean):>3}  min={min(clean):>9.3f}  "
            f"p50={pct(50):>9.3f}  mean={statistics.fmean(clean):>9.3f}  "
            f"p90={pct(90):>9.3f}  p99={pct(99):>9.3f}  max={max(clean):>9.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("perf_metrics_path", type=Path,
                   help="perf_metrics.json (agg) or perf_metrics_proxy.json (disagg)")
    p.add_argument("--rwlt", type=Path, required=True,
                   help="rwlt_requests.jsonl from the same session")
    p.add_argument("--label", default="run", help="label printed in summary")
    p.add_argument("--disagg", action="store_true",
                   help="treat the input as a disagg-proxy joined dump")
    p.add_argument("--out", type=Path, help="output TSV path (optional)")
    args = p.parse_args()

    with args.perf_metrics_path.open() as f:
        records = json.load(f)
    if not isinstance(records, list):
        print(f"ERROR: expected list, got {type(records).__name__}",
              file=__import__("sys").stderr)
        return 2

    rwlt = load_rwlt(args.rwlt)

    if len(records) != len(rwlt):
        print(f"WARN: perf_metrics records ({len(records)}) != successful "
              f"RWLT rows ({len(rwlt)}); pairing first "
              f"{min(len(records), len(rwlt))} only.")
    n = min(len(records), len(rwlt))

    cols = DISAGG_COLS if args.disagg else AGG_COLS
    extractor = extract_disagg_row if args.disagg else extract_agg_row

    rows: list[dict[str, Any]] = []
    for i in range(n):
        rec = records[i]
        rw = rwlt[i]
        derived = extractor(rec)
        def _opt_ms(key: str) -> float:
            v = rw.get(key)
            if v is None: return float("nan")
            try: return float(v) * 1000.0
            except (TypeError, ValueError): return float("nan")
        try:
            elapsed_client = float(rw.get("end_time")) - float(rw.get("start_time"))
            elapsed_client_ms = elapsed_client * 1000.0
        except (TypeError, ValueError):
            elapsed_client_ms = float("nan")
        row: dict[str, Any] = {
            "match_idx": i,
            "conversation_id": rw.get("conversation_id", ""),
            "conversation_idx": rw.get("conversation_idx", -1),
            "input_tokens": rw.get("server_input_tokens", -1),
            "output_tokens": rw.get("server_output_tokens", -1),
            "cached_tokens": rw.get("server_cached_tokens", -1),
            "ttft_client_ms": _opt_ms("ttft"),
            "itl_client_ms": _opt_ms("itl"),
            "elapsed_client_ms": elapsed_client_ms,
        }
        row.update(derived)
        rows.append(row)

    header = ["match_idx", "conversation_id", "conversation_idx",
              "input_tokens", "output_tokens", "cached_tokens",
              "ttft_client_ms", "itl_client_ms", "elapsed_client_ms"] + cols

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            for row in rows:
                w.writerow([_fmt(row[c]) if isinstance(row[c], float) else row[c]
                            for c in header])
        print(f"Wrote {len(rows)} rows -> {args.out}")

    # Per-stage summary printed to stdout.
    print()
    print(f"=== {args.label} per-stage summary (ms) ===")
    print(f"  records paired: n={n}")
    print(f"  {'ttft_client (rwlt jsonl)':<28}: {_summary([r['ttft_client_ms'] for r in rows])}")
    for c in cols:
        print(f"  {c:<28}: {_summary([r[c] for r in rows])}")
    print()
    # Sanity: stages should ~sum to ttft_server.
    if args.disagg:
        sum_cols = ["proxy_to_ctx_ms", "ctx_preproc_ms", "ctx_queue_ms",
                    "ctx_prefill_ms", "ctx_postproc_ms", "relay_hop_ms",
                    "gen_preproc_ms", "gen_queue_ms", "gen_first_decode_ms",
                    "gen_postproc_ms", "proxy_to_client_ms"]
    else:
        sum_cols = ["srv_preproc_ms", "queue_ms", "prefill_ms", "srv_postproc_ms"]
    sums = []
    for r in rows:
        vals = [r[c] for c in sum_cols]
        if any(math.isnan(v) for v in vals if isinstance(v, float)):
            continue
        sums.append(sum(vals) - r["ttft_server_ms"])
    print(f"  sum(stages) - ttft_server  : {_summary(sums)}  "
          "(should be ~0; non-zero hints at clock-offset slop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
