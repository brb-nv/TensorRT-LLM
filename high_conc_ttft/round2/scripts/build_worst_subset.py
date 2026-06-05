#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a subset trajectory dataset from the worst-TTFT conversations of a
previous RWLT run.

Step 1: rank conversations in <requests.jsonl> by TTFT (clean pre-crash,
        successful requests only) and pick the worst N.
Step 2: stream the full <dataset.jsonl> and copy every line whose
        conversation_id is in that set, preserving format/order.

Usage:
  build_worst_subset.py --requests <rwlt_requests.jsonl> \
      --dataset <full.jsonl> --out <subset.jsonl> [--n 250] [--rank-by mean]
"""
import argparse
import json
import statistics as st


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--rank-by", choices=["mean", "max", "p95"], default="mean")
    ap.add_argument("--min-turns-seen", type=int, default=1,
                    help="require at least this many successful reqs per convo")
    args = ap.parse_args()

    # --- Step 1: rank conversations by TTFT in the clean window ---
    recs = []
    with open(args.requests) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    bad = [r for r in recs if not r.get("success")]
    first_fail = min((r["start_time"] for r in bad), default=None)
    clean = [r for r in recs
             if r.get("success") and r.get("ttft") is not None
             and (first_fail is None or r["end_time"] <= first_fail)]

    by_cid = {}
    for r in clean:
        by_cid.setdefault(r["conversation_id"], []).append(r["ttft"])

    scored = []
    for cid, tts in by_cid.items():
        if len(tts) < args.min_turns_seen:
            continue
        if args.rank_by == "mean":
            score = st.mean(tts)
        elif args.rank_by == "max":
            score = max(tts)
        else:
            score = pct(tts, 95)
        scored.append((score, len(tts), cid))
    scored.sort(reverse=True)
    pick = scored[:args.n]
    pick_ids = {cid for _, _, cid in pick}

    print(f"distinct clean conversations: {len(by_cid)}")
    print(f"selected worst {len(pick_ids)} by {args.rank_by} TTFT")
    if pick:
        sc = [s for s, _, _ in pick]
        print(f"  selected score range: {min(sc):.2f}s .. {max(sc):.2f}s  "
              f"(median {pct(sc,50):.2f}s)")
        worst = pick[:5]
        print("  worst 5:")
        for s, n, cid in worst:
            print(f"    {cid}  {args.rank_by}={s:.2f}s  reqs={n}")

    # --- Step 2: stream the dataset, copy matching lines ---
    found = {}
    lines_written = 0
    with open(args.dataset) as fin, open(args.out, "w") as fout:
        for line in fin:
            # cheap prefilter to avoid json.loads on every line
            try:
                d = json.loads(line)
            except Exception:
                continue
            cid = d.get("conversation_id")
            if cid in pick_ids:
                fout.write(line if line.endswith("\n") else line + "\n")
                found[cid] = found.get(cid, 0) + 1
                lines_written += 1

    missing = pick_ids - set(found)
    turns = sorted(found.values())
    print(f"\nwrote {lines_written} lines for {len(found)} conversations -> {args.out}")
    if turns:
        print(f"  turns/conversation: min={turns[0]} p50={pct(turns,50):.0f} "
              f"p95={pct(turns,95):.0f} max={turns[-1]}")
    if missing:
        print(f"  WARNING: {len(missing)} selected conversations not found in dataset "
              f"(e.g. {list(missing)[:3]})")


if __name__ == "__main__":
    main()
