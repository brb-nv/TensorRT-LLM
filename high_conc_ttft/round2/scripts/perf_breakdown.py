#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge disagg /perf_metrics records (final drain + periodic snapshots, which
split records due to destructive reads) and report the per-request TTFT stage
breakdown: ctx queue, ctx prefill-engine, ctx->gen handoff/KV-transfer, and
gen first-token. All times are steady-clock seconds.

Usage: perf_breakdown.py <results_dir>
"""
import glob
import json
import os
import statistics as st
import sys


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def tm(pm):
    """Pull timing_metrics dict from a worker perf_metrics blob."""
    if not pm:
        return None
    inner = pm.get("perf_metrics", pm)
    return inner.get("timing_metrics")


def main():
    rd = sys.argv[1]
    files = [os.path.join(rd, "perf_metrics_proxy.json")]
    files += sorted(glob.glob(os.path.join(rd, "perf_snapshots", "*.json")))

    merged = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for r in d:
            ctxpm = r.get("ctx_perf_metrics") or {}
            rid = (ctxpm.get("request_id"), r.get("ctx_server"), r.get("gen_server"))
            merged[rid] = r  # dedup by ctx request id + servers
    recs = list(merged.values())
    print(f"merged {len(recs)} unique records from {len(files)} files")

    rows = {"ctx_queue": [], "ctx_prefill_engine": [], "ctx_total": [],
            "handoff_to_gen_ftt": [], "e2e_ttft": []}
    for r in recs:
        c = tm(r.get("ctx_perf_metrics"))
        if not c:
            continue
        arr = c.get("arrival_time"); sched = c.get("first_scheduled_time")
        ftt = c.get("first_token_time"); sarr = c.get("server_arrival_time")
        if None in (arr, sched, ftt):
            continue
        rows["ctx_queue"].append(sched - arr)
        rows["ctx_prefill_engine"].append(ftt - sched)
        if sarr is not None:
            rows["ctx_total"].append(ftt - sarr)
        da = r.get("disagg_server_arrival_time")
        dftt = r.get("disagg_server_first_token_time")
        if da is not None and dftt is not None:
            rows["e2e_ttft"].append(dftt - da)

    print(f"\n{'stage':<24}{'n':>6}{'p50 (ms)':>11}{'p95 (ms)':>11}{'max (ms)':>11}")
    for k in ["ctx_queue", "ctx_prefill_engine", "ctx_total", "e2e_ttft"]:
        v = [x * 1000 for x in rows[k]]
        if v:
            print(f"{k:<24}{len(v):>6}{pct(v,50):>11.1f}{pct(v,95):>11.1f}{max(v):>11.1f}")

    # KV-cache reuse on the ctx side (from kv_cache_metrics if present)
    reuse = []
    for r in recs:
        cm = (r.get("ctx_perf_metrics") or {}).get("perf_metrics", {}).get("kv_cache_metrics")
        if cm and cm.get("num_total_allocated_blocks"):
            reuse.append(cm.get("num_reused_blocks", 0) / cm["num_total_allocated_blocks"])
    if reuse:
        print(f"\nctx block-reuse fraction: p50={pct(reuse,50):.2f} mean={st.mean(reuse):.2f}")


if __name__ == "__main__":
    main()
