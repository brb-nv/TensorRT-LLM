#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-TTFT (>5s) analysis for an RWLT rwlt_requests.jsonl.

The client log has no server-id field, so true ctx/gen routing can't be read
directly. server_cached_tokens is used as a routing-affinity / KV-reuse proxy:
a turn routed back to the worker holding its conversation prefix shows high
cached tokens; a turn sent to a cold worker re-prefills (cached ~ 0).

Usage: analyze_ttft_high.py <rwlt_requests.jsonl> [--thresh 5.0] [--clean-only]
"""
import argparse
import json
import statistics as st


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def uncached(r):
    return max(0, (r.get("server_input_tokens") or 0) - (r.get("server_cached_tokens") or 0))


def hitrate(r):
    isl = r.get("server_input_tokens") or 0
    return ((r.get("server_cached_tokens") or 0) / isl) if isl else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--clean-only", action="store_true",
                    help="restrict to reqs that ended before the first failure")
    args = ap.parse_args()

    recs = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["start_time"])
    ok = [r for r in recs if r.get("success") and r.get("ttft") is not None]
    bad = [r for r in recs if not r.get("success")]
    first_fail = min((r["start_time"] for r in bad), default=None)
    if args.clean_only and first_fail is not None:
        ok = [r for r in ok if r["end_time"] <= first_fail]

    hi = [r for r in ok if r["ttft"] > args.thresh]
    lo = [r for r in ok if r["ttft"] <= args.thresh]
    print(f"analyzed {len(ok)} successful reqs "
          f"({'clean window only' if args.clean_only else 'all'})")
    print(f"TTFT > {args.thresh}s : {len(hi)} ({100*len(hi)/len(ok):.1f}%)   "
          f"<= {args.thresh}s : {len(lo)}")

    def blk(label, rs):
        if not rs:
            print(f"\n[{label}] (none)"); return
        tt = [r["ttft"] for r in rs]
        unc = [uncached(r) for r in rs]
        isl = [(r.get("server_input_tokens") or 0) for r in rs]
        hr = [hitrate(r) for r in rs]
        cidx = [(r.get("conversation_idx") or 0) for r in rs]
        print(f"\n[{label}]  n={len(rs)}")
        print(f"  TTFT      p50={pct(tt,50):.2f} p95={pct(tt,95):.2f} max={max(tt):.2f}")
        print(f"  ISL       p50={pct(isl,50):.0f} p95={pct(isl,95):.0f}")
        print(f"  uncached  p50={pct(unc,50):.0f} p95={pct(unc,95):.0f}  "
              f"(input tokens actually prefilled)")
        print(f"  cache-hit p50={pct(hr,50):.2f} mean={st.mean(hr):.2f}")
        print(f"  turn idx  p50={pct(cidx,50):.0f} p95={pct(cidx,95):.0f} max={max(cidx)}")

    blk(f"TTFT > {args.thresh}s", hi)
    blk(f"TTFT <= {args.thresh}s", lo)

    # TTFT vs uncached-prefill-tokens (the expected physical driver)
    print("\n## TTFT vs uncached prefill tokens (successful reqs)")
    edges = [0, 1, 1000, 4000, 8000, 16000, 32000, 64000, 1 << 30]
    names = ["0 (full reuse)", "1-1k", "1k-4k", "4k-8k", "8k-16k",
             "16k-32k", "32k-64k", ">64k"]
    print(f"  {'uncached':>16} {'n':>6} {'ttft p50':>9} {'ttft p95':>9} {'hit p50':>8}")
    ok = [r for r in ok if r.get("ttft") is not None]
    for i, nm in enumerate(names):
        sub = [r for r in ok if edges[i] <= uncached(r) < edges[i + 1]]
        if sub:
            tt = [r["ttft"] for r in sub]
            hr = [hitrate(r) for r in sub]
            print(f"  {nm:>16} {len(sub):>6} {pct(tt,50):>8.2f}s "
                  f"{pct(tt,95):>8.2f}s {pct(hr,50):>7.2f}")

    # Within-trajectory turn progression: does TTFT grow with turn index?
    print("\n## Within-trajectory TTFT vs turn index (conversation_idx)")
    bycid = {}
    for r in ok:
        bycid.setdefault(r["conversation_id"], []).append(r)
    grow, flat, shrink, total = 0, 0, 0, 0
    by_turn = {}
    for cid, rs in bycid.items():
        rs.sort(key=lambda r: r["conversation_idx"])
        for r in rs:
            by_turn.setdefault(r["conversation_idx"], []).append(r["ttft"])
        if len(rs) >= 3:
            total += 1
            # sign of correlation between turn idx and ttft (simple slope)
            xs = [r["conversation_idx"] for r in rs]
            ys = [r["ttft"] for r in rs]
            mx, my = st.mean(xs), st.mean(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs) or 1.0
            slope = num / den
            if slope > 0.05: grow += 1
            elif slope < -0.05: shrink += 1
            else: flat += 1
    print(f"  trajectories with >=3 turns: {total}")
    print(f"    TTFT rises with turn: {grow} ({100*grow/max(total,1):.0f}%)   "
          f"flat: {flat}   falls: {shrink} ({100*shrink/max(total,1):.0f}%)")
    print(f"  {'turn':>5} {'n':>6} {'ttft p50':>9} {'ttft p95':>9}")
    for t in sorted(by_turn)[:16]:
        v = by_turn[t]
        print(f"  {t:>5} {len(v):>6} {pct(v,50):>8.2f}s {pct(v,95):>8.2f}s")


if __name__ == "__main__":
    main()
