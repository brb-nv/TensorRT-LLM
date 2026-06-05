#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze an RWLT rwlt_requests.jsonl: TTFT distribution, growth over time, and
TTFT vs in-flight (concurrency) binning -- the core signal for NVBug 6266370.

Usage: analyze_requests.py <rwlt_requests.jsonl> [--settle SECONDS]
The "clean window" = requests whose lifetime ends before the first failed
request (i.e. before the prefill workers started crashing).
"""
import argparse
import bisect
import json
import statistics as st
import sys


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(label, recs):
    tt = [r["ttft"] for r in recs if r.get("ttft") is not None]
    isl = [r.get("server_input_tokens", 0) for r in recs]
    print(f"\n## {label}  (n={len(recs)})")
    if not tt:
        print("  (no ttft values)")
        return
    print(f"  TTFT  p50={pct(tt,50):.3f}s  p75={pct(tt,75):.3f}s  "
          f"p90={pct(tt,90):.3f}s  p95={pct(tt,95):.3f}s  "
          f"p99={pct(tt,99):.3f}s  max={max(tt):.3f}s  mean={st.mean(tt):.3f}s")
    print(f"  ISL   p50={pct(isl,50):.0f}  p95={pct(isl,95):.0f}  max={max(isl)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--settle", type=float, default=240.0,
                    help="seconds of settling to drop from the start")
    ap.add_argument("--first-n", type=int, default=320,
                    help="also report the first N successful measurement reqs")
    args = ap.parse_args()

    recs = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["start_time"])
    t0 = recs[0]["start_time"]
    tN = max(r["end_time"] for r in recs)
    print(f"file: {args.path}")
    print(f"records: {len(recs)}   wall span: {tN - t0:.0f}s")

    ok = [r for r in recs if r.get("success")]
    bad = [r for r in recs if not r.get("success")]
    print(f"success: {len(ok)}   failed: {len(bad)}   "
          f"fail-rate: {100*len(bad)/len(recs):.1f}%")

    # First failure marks the crash boundary.
    first_fail_t = min((r["start_time"] for r in bad), default=None)
    if first_fail_t is not None:
        print(f"first failure at t+{first_fail_t - t0:.0f}s "
              f"(epoch {first_fail_t:.0f})")

    # Settling window drop (steady-state C=160 measurement region).
    meas = [r for r in ok if r["start_time"] - t0 >= args.settle]

    # Clean window: successful measurement reqs that COMPLETED before the
    # first failure (so engine corruption can't have skewed them).
    if first_fail_t is not None:
        clean = [r for r in meas if r["end_time"] <= first_fail_t]
    else:
        clean = meas

    summarize("ALL successful", ok)
    summarize(f"MEASUREMENT (drop first {args.settle:.0f}s settling)", meas)
    summarize("CLEAN (measurement + ended before first crash)", clean)
    summarize(f"FIRST {args.first_n} clean reqs (by start)", clean[:args.first_n])

    # --- TTFT vs in-flight count (the bug's core claim) ---
    # In-flight at request r's start = #reqs with start <= r.start < end.
    starts = sorted(r["start_time"] for r in recs)
    ends = sorted(r["end_time"] for r in recs)
    bins = {"<=2": [], "3-8": [], "9-16": [], "17-32": [],
            "33-48": [], "49-80": [], ">80": []}

    def binlabel(n):
        if n <= 2: return "<=2"
        if n <= 8: return "3-8"
        if n <= 16: return "9-16"
        if n <= 32: return "17-32"
        if n <= 48: return "33-48"
        if n <= 80: return "49-80"
        return ">80"

    for r in clean:
        if r.get("ttft") is None:
            continue
        t = r["start_time"]
        started = bisect.bisect_right(starts, t)
        finished = bisect.bisect_right(ends, t)
        inflight = started - finished
        bins[binlabel(inflight)].append(r["ttft"])

    print("\n## TTFT vs in-flight count  (clean window)")
    print(f"  {'in-flight':>10} {'n':>6} {'p50':>8} {'p95':>8} {'max':>8}")
    for k in ["<=2", "3-8", "9-16", "17-32", "33-48", "49-80", ">80"]:
        v = bins[k]
        if v:
            print(f"  {k:>10} {len(v):>6} {pct(v,50):>7.3f}s "
                  f"{pct(v,95):>7.3f}s {max(v):>7.3f}s")

    # --- TTFT over wall-clock minutes ---
    print("\n## TTFT over time  (60s buckets, successful reqs)")
    print(f"  {'t+min':>6} {'n':>6} {'p50':>8} {'p95':>8} {'inflight~':>9}")
    buckets = {}
    for r in ok:
        m = int((r["start_time"] - t0) // 60)
        buckets.setdefault(m, []).append(r)
    for m in sorted(buckets):
        rs = buckets[m]
        tt = [x["ttft"] for x in rs if x.get("ttft") is not None]
        if not tt:
            continue
        # rough in-flight at mid-bucket
        tm = t0 + m * 60 + 30
        infl = bisect.bisect_right(starts, tm) - bisect.bisect_right(ends, tm)
        print(f"  {m:>6} {len(rs):>6} {pct(tt,50):>7.3f}s "
              f"{pct(tt,95):>7.3f}s {infl:>9}")


if __name__ == "__main__":
    sys.exit(main())
