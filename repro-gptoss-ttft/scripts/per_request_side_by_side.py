#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request side-by-side: agg vs disagg.

Emits ONE row per request with the columns the TTFT investigation cares about:

    conv_id  turn  fresh  reused  reuse%  KV_MB  kv_xfer_ms  wait_ms
      ttft_agg_ms  ttft_disagg_ms  delta_ttft_ms

Where (for disagg, all from `gen_perf_metrics.perf_metrics.timing_metrics`):
  * fresh        = server_input_tokens - server_cached_tokens  (from rwlt)
  * reused       = server_cached_tokens
  * kv_xfer_ms   = kv_cache_transfer_end - kv_cache_transfer_start
  * wait_ms      = first_scheduled_time - arrival_time          (gen_queue_ms)
  * ttft_disagg  = first_token_time - disagg_server_arrival_time
  * ttft_agg     = (agg) server_first_token_time - server_arrival_time
  * delta_ttft   = ttft_disagg - ttft_agg

Pairing strategy (valid at concurrency=1, the way RWLT is run):
  * Sort each session's rwlt_requests.jsonl by start_time and pair the
    Nth successful perf_metrics record with the Nth successful RWLT row.
  * Join across agg<->disagg by (conversation_id, conversation_idx).

Usage:
    python3 scripts/per_request_side_by_side.py \\
        --agg-dir rwlt-results/agg_round5_patched_top5_0521 \\
        --disagg-dir rwlt-results/disagg_round5_patched_top5_0521 \\
        --out rwlt-results/side_by_side_round5.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


def _ms(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return float("nan")
    try:
        return (float(a) - float(b)) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, w: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "."
    return f"{v:.{w}f}"


def load_rwlt(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("success"):
                out.append(r)
    out.sort(key=lambda r: r.get("start_time", 0.0))
    return out


def load_perf(path: Path) -> list[dict[str, Any]]:
    """Load /perf_metrics dump. Skip non-success records (server-side errors)."""
    with path.open() as f:
        return json.load(f)


def agg_row(perf_rec: dict[str, Any], rwlt_rec: dict[str, Any]) -> dict[str, Any]:
    pm = perf_rec.get("perf_metrics", {}) or {}
    tm = pm.get("timing_metrics", {}) or {}
    sa = tm.get("server_arrival_time")
    sft = tm.get("server_first_token_time")
    fresh = rwlt_rec["server_input_tokens"] - rwlt_rec["server_cached_tokens"]
    return {
        "conv_id": rwlt_rec["conversation_id"],
        "turn": rwlt_rec["conversation_idx"],
        "fresh": fresh,
        "reused": rwlt_rec["server_cached_tokens"],
        "ttft_agg_ms": _ms(sft, sa),
    }


def disagg_row(perf_rec: dict[str, Any], rwlt_rec: dict[str, Any]) -> dict[str, Any]:
    da = perf_rec.get("disagg_server_arrival_time")
    dft = perf_rec.get("disagg_server_first_token_time")

    gen_block = perf_rec.get("gen_perf_metrics") or {}
    gen_pm = gen_block.get("perf_metrics") or {}
    gtm = gen_pm.get("timing_metrics", {}) or {}

    gen_arr = gtm.get("arrival_time")
    gen_fs = gtm.get("first_scheduled_time")
    gen_kvx_s = gtm.get("kv_cache_transfer_start")
    gen_kvx_e = gtm.get("kv_cache_transfer_end")
    gen_kvx_size = gtm.get("kv_cache_size")

    kv_xfer_ms = _ms(gen_kvx_e, gen_kvx_s)
    wait_ms = _ms(gen_fs, gen_arr)
    kv_mb = (gen_kvx_size / (1024.0 * 1024.0)
             if isinstance(gen_kvx_size, (int, float)) else float("nan"))

    fresh = rwlt_rec["server_input_tokens"] - rwlt_rec["server_cached_tokens"]
    return {
        "conv_id": rwlt_rec["conversation_id"],
        "turn": rwlt_rec["conversation_idx"],
        "fresh": fresh,
        "reused": rwlt_rec["server_cached_tokens"],
        "kv_mb": kv_mb,
        "kv_xfer_ms": kv_xfer_ms,
        "wait_ms": wait_ms,
        "ttft_disagg_ms": _ms(dft, da),
    }


def build_index(perf: list[dict[str, Any]], rwlt: list[dict[str, Any]],
                row_fn) -> dict[tuple[str, int], dict[str, Any]]:
    if len(perf) != len(rwlt):
        print(f"[warn] perf records ({len(perf)}) != successful rwlt rows "
              f"({len(rwlt)}) -- pairing by index may misalign at the tail.")
    n = min(len(perf), len(rwlt))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for i in range(n):
        row = row_fn(perf[i], rwlt[i])
        out[(row["conv_id"], row["turn"])] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agg-dir", type=Path, required=True)
    p.add_argument("--disagg-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output TSV (one row per joined request)")
    p.add_argument("--agg-perf", default="perf_metrics.json",
                   help="filename inside --agg-dir (default: perf_metrics.json)")
    p.add_argument("--disagg-perf", default="perf_metrics_proxy.json",
                   help="filename inside --disagg-dir (default: perf_metrics_proxy.json)")
    p.add_argument("--sort", choices=("delta", "trajectory"), default="trajectory",
                   help="row order: descending delta, or (conv_id,turn).")
    p.add_argument("--topn", type=int, default=None,
                   help="if given, also print the top-N rows to stdout.")
    args = p.parse_args()

    agg_perf = load_perf(args.agg_dir / args.agg_perf)
    agg_rwlt = load_rwlt(args.agg_dir / "rwlt_requests.jsonl")
    agg_idx = build_index(agg_perf, agg_rwlt, agg_row)

    disagg_perf = load_perf(args.disagg_dir / args.disagg_perf)
    disagg_rwlt = load_rwlt(args.disagg_dir / "rwlt_requests.jsonl")
    disagg_idx = build_index(disagg_perf, disagg_rwlt, disagg_row)

    # Join: keep keys that exist on both sides.
    common = sorted(set(agg_idx) & set(disagg_idx),
                    key=lambda k: (k[0], k[1]))
    only_agg = set(agg_idx) - set(disagg_idx)
    only_disagg = set(disagg_idx) - set(agg_idx)
    print(f"agg rows={len(agg_idx)}  disagg rows={len(disagg_idx)}  "
          f"common={len(common)}  agg_only={len(only_agg)}  "
          f"disagg_only={len(only_disagg)}")

    joined = []
    for key in common:
        a = agg_idx[key]
        d = disagg_idx[key]
        fresh = d["fresh"]
        reused = d["reused"]
        total = fresh + reused
        reuse_pct = (100.0 * reused / total) if total else 0.0
        delta = (d["ttft_disagg_ms"] - a["ttft_agg_ms"]
                 if not (math.isnan(d["ttft_disagg_ms"])
                         or math.isnan(a["ttft_agg_ms"])) else float("nan"))
        joined.append({
            "conv_id": key[0],
            "turn": key[1],
            "fresh": fresh,
            "reused": reused,
            "reuse_pct": reuse_pct,
            "kv_mb": d["kv_mb"],
            "kv_xfer_ms": d["kv_xfer_ms"],
            "wait_ms": d["wait_ms"],
            "ttft_agg_ms": a["ttft_agg_ms"],
            "ttft_disagg_ms": d["ttft_disagg_ms"],
            "delta_ttft_ms": delta,
        })

    if args.sort == "delta":
        joined.sort(key=lambda r: (math.isnan(r["delta_ttft_ms"]),
                                    -r["delta_ttft_ms"]))

    cols = ["conv_id", "turn", "fresh", "reused", "reuse_pct",
            "kv_mb", "kv_xfer_ms", "wait_ms",
            "ttft_agg_ms", "ttft_disagg_ms", "delta_ttft_ms"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in joined:
            w.writerow({k: (round(v, 3) if isinstance(v, float) and not math.isnan(v)
                            else v) for k, v in r.items()})
    print(f"wrote {args.out}  rows={len(joined)}")

    # Summary
    def stats(rows, key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        if not vals:
            return "n=0"
        vals_s = sorted(vals)
        def pct(q):
            k = (len(vals_s) - 1) * q / 100.0
            lo, hi = math.floor(k), math.ceil(k)
            return vals_s[int(k)] if lo == hi else vals_s[lo] + (vals_s[hi] - vals_s[lo]) * (k - lo)
        return (f"n={len(vals)} mean={statistics.fmean(vals):>8.1f}  "
                f"p50={pct(50):>8.1f}  p90={pct(90):>8.1f}  "
                f"p99={pct(99):>8.1f}  max={max(vals):>8.1f}")

    print()
    print("=== summary (ms unless otherwise noted) ===")
    print(f"  ttft_agg     : {stats(joined, 'ttft_agg_ms')}")
    print(f"  ttft_disagg  : {stats(joined, 'ttft_disagg_ms')}")
    print(f"  delta_ttft   : {stats(joined, 'delta_ttft_ms')}")
    print(f"  wait_ms      : {stats(joined, 'wait_ms')}")
    print(f"  kv_xfer_ms   : {stats(joined, 'kv_xfer_ms')}")
    print(f"  kv_mb        : {stats(joined, 'kv_mb')}")
    print(f"  fresh        : {stats(joined, 'fresh')}")
    print(f"  reused       : {stats(joined, 'reused')}")
    print(f"  reuse_pct    : {stats(joined, 'reuse_pct')}")

    if args.topn:
        joined_sorted = sorted(
            joined,
            key=lambda r: (math.isnan(r["delta_ttft_ms"]), -r["delta_ttft_ms"]))
        print()
        print(f"=== top-{args.topn} by delta_ttft_ms ===")
        h = f"{'conv':>6} {'t':>3} {'fresh':>7} {'reused':>7} {'reuse%':>7} " \
            f"{'KV_MB':>7} {'kv_xfer':>9} {'wait':>9} " \
            f"{'ttft_agg':>9} {'ttft_dis':>9} {'delta':>9}"
        print(h)
        for r in joined_sorted[:args.topn]:
            print(f"{r['conv_id'].split('-')[-1]:>6} {r['turn']:>3} "
                  f"{r['fresh']:>7} {r['reused']:>7} "
                  f"{_fmt(r['reuse_pct'], 1):>7} "
                  f"{_fmt(r['kv_mb'], 1):>7} {_fmt(r['kv_xfer_ms'], 1):>9} "
                  f"{_fmt(r['wait_ms'], 1):>9} "
                  f"{_fmt(r['ttft_agg_ms'], 1):>9} "
                  f"{_fmt(r['ttft_disagg_ms'], 1):>9} "
                  f"{_fmt(r['delta_ttft_ms'], 1):>9}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
