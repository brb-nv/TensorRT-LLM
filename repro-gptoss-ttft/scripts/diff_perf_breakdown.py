#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-turn agg-vs-disagg TTFT delta, attributed to specific server stages.

Pairs an agg breakdown TSV (from perf_metrics_breakdown.py without --disagg)
with a disagg breakdown TSV (with --disagg) by (conversation_id,
conversation_idx) and computes:

  Delta(stage) = disagg.stage - agg.equivalent

For disagg-exclusive stages (proxy_to_ctx, relay_hop, kv_transfer, ...) the
agg side is implicitly 0. For stages with an agg counterpart (prefill, GPU
forward) we diff like-for-like.

The bottom of the report sanity-checks:
  sum(disagg_exclusive) + (disagg.ctx_prefill - agg.prefill) +
                          (disagg.gen_postproc - agg.srv_postproc)
   ~= disagg.ttft_server - agg.ttft_server

so the per-stage contributions add up to the observed TTFT gap.

Usage:
    python3 scripts/diff_perf_breakdown.py \
        rwlt-results/agg_round4_top5_0521/breakdown.tsv \
        rwlt-results/disagg_round4_top5_0521/breakdown.tsv \
        --label-a agg --label-b disagg \
        --per-turn-out rwlt-results/diff_round4_top5_0521.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Optional


def _f(x: str) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        return v if not math.isnan(v) else None
    except ValueError:
        return None


def _load(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            key = (row["conversation_id"], int(row["conversation_idx"]))
            out[key] = row
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _stats(label: str, values: list[Optional[float]]) -> str:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return f"  {label:<32}: n=0"
    return (f"  {label:<32}: n={len(clean):>3}  "
            f"min={min(clean):>+9.3f}  "
            f"p50={_pct(clean,50):>+9.3f}  "
            f"mean={statistics.fmean(clean):>+9.3f}  "
            f"p90={_pct(clean,90):>+9.3f}  "
            f"p99={_pct(clean,99):>+9.3f}  "
            f"max={max(clean):>+9.3f}")


# Stages that exist only in disagg; their full value is the per-turn
# contribution to the TTFT gap.
DISAGG_EXCLUSIVE = [
    "proxy_to_ctx_ms",
    "ctx_postproc_ms",
    "relay_hop_ms",
    "gen_preproc_ms",
    "gen_queue_ms",
    "gen_first_decode_ms",
    "proxy_to_client_ms",
]

# Stages that have a clean agg counterpart. Per-turn contribution to the
# TTFT gap is (disagg_stage - agg_stage). Sanity sum below relies on these
# being the complete set of paired stages.
#
# Note: (ctx_prefill, prefill) is an *almost* clean pair. Agg's prefill_ms
# spans first_scheduled -> first_token, which includes the first decode
# step. Disagg's ctx_prefill_ms spans first_scheduled -> first_token on the
# ctx worker, where "first_token" is just the marker after prefill compute
# completes (no real token is sampled on ctx). The first decode in disagg
# is in gen_first_decode_ms above, so this delta captures "extra prefill
# work on disagg minus agg's first-decode work" -- typically a small
# negative number when compute is identical.
LIKE_FOR_LIKE_PAIRED = [
    ("ctx_preproc_ms", "srv_preproc_ms",  "delta_preproc_ms"),
    ("ctx_queue_ms",   "queue_ms",        "delta_queue_ms"),
    ("ctx_prefill_ms", "prefill_ms",      "delta_prefill_ms"),
    ("gen_postproc_ms","srv_postproc_ms", "delta_postproc_ms"),
]

# Informational deltas; CUDA-event measurements. Not part of the attribution
# sum (those numbers are already inside delta_prefill_ms via ctx_prefill_ms),
# but the cleanest "is the forward call itself slower?" signal.
LIKE_FOR_LIKE_INFO = [
    ("ctx_gpu_forward_ms",  "gpu_forward_ms",  "delta_gpu_forward_ms"),
    ("ctx_prefill_host_ms", "prefill_host_ms", "delta_prefill_host_ms"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agg_tsv", type=Path, help="breakdown.tsv from agg run")
    p.add_argument("disagg_tsv", type=Path, help="breakdown.tsv from disagg run")
    p.add_argument("--label-a", default="agg")
    p.add_argument("--label-b", default="disagg")
    p.add_argument("--per-turn-out", type=Path,
                   help="optional TSV with one row per paired turn")
    args = p.parse_args()

    a = _load(args.agg_tsv)
    b = _load(args.disagg_tsv)

    common = sorted(set(a.keys()) & set(b.keys()),
                    key=lambda k: (k[0], k[1]))
    only_a = sorted(set(a.keys()) - set(b.keys()))
    only_b = sorted(set(b.keys()) - set(a.keys()))

    print(f"Pairs matched : {len(common)}")
    print(f"Only in {args.label_a:<10}: {len(only_a)}")
    print(f"Only in {args.label_b:<10}: {len(only_b)}")
    if only_a[:5]:
        print(f"  e.g. only_{args.label_a}: {only_a[:5]}")
    if only_b[:5]:
        print(f"  e.g. only_{args.label_b}: {only_b[:5]}")

    # Build per-turn rows.
    out_rows: list[dict[str, Any]] = []
    per_stage_deltas: dict[str, list[float]] = {s: [] for s in DISAGG_EXCLUSIVE}
    for _, _, lbl in LIKE_FOR_LIKE_PAIRED:
        per_stage_deltas[lbl] = []
    for _, _, lbl in LIKE_FOR_LIKE_INFO:
        per_stage_deltas[lbl] = []
    ttft_gap: list[float] = []
    sum_check: list[float] = []
    isl_mismatches: list[tuple[int, int]] = []

    for key in common:
        ra = a[key]
        rb = b[key]

        # ISL agreement check
        ia = _f(ra.get("input_tokens", ""))
        ib = _f(rb.get("input_tokens", ""))
        if ia is not None and ib is not None and int(ia) != int(ib):
            isl_mismatches.append((int(ia), int(ib)))

        a_ttft = _f(ra.get("ttft_server_ms"))
        b_ttft = _f(rb.get("ttft_server_ms"))
        gap = (b_ttft - a_ttft) if (a_ttft is not None and b_ttft is not None) else None
        if gap is not None:
            ttft_gap.append(gap)

        row: dict[str, Any] = {
            "conversation_id": key[0],
            "conversation_idx": key[1],
            "input_tokens": ra.get("input_tokens", ""),
            "cached_tokens_agg": ra.get("cached_tokens", ""),
            "cached_tokens_disagg": rb.get("cached_tokens", ""),
            "ttft_server_agg_ms": a_ttft,
            "ttft_server_disagg_ms": b_ttft,
            "delta_ttft_ms": gap,
        }

        for s in DISAGG_EXCLUSIVE:
            v = _f(rb.get(s))
            row[s] = v
            if v is not None:
                per_stage_deltas[s].append(v)

        for d_col, a_col, lbl in LIKE_FOR_LIKE_PAIRED + LIKE_FOR_LIKE_INFO:
            dv = _f(rb.get(d_col))
            av = _f(ra.get(a_col))
            delta = (dv - av) if (dv is not None and av is not None) else None
            row[lbl] = delta
            if delta is not None:
                per_stage_deltas[lbl].append(delta)

        # Sanity sum: disagg-exclusive stages + paired like-for-like deltas
        # should equal delta_ttft (up to clock-offset slop).
        contribs = [row[s] for s in DISAGG_EXCLUSIVE]
        for _, _, lbl in LIKE_FOR_LIKE_PAIRED:
            contribs.append(row[lbl])
        if all(c is not None for c in contribs) and gap is not None:
            sum_check.append(sum(contribs) - gap)

        out_rows.append(row)

    if isl_mismatches:
        print(f"\nWARN: {len(isl_mismatches)} turns have ISL mismatch between "
              f"{args.label_a} and {args.label_b}. First 3: {isl_mismatches[:3]}")

    print()
    print(f"=== TTFT gap ({args.label_b} - {args.label_a}) ===")
    print(_stats("delta_ttft_ms", ttft_gap))

    print()
    print("=== Per-stage delta attribution (ms) ===")
    print("  Disagg-exclusive stages (full value contributes to gap):")
    for s in DISAGG_EXCLUSIVE:
        print(_stats(s, per_stage_deltas[s]))
    print("  Paired like-for-like deltas (disagg - agg, counted in sum):")
    for _, _, lbl in LIKE_FOR_LIKE_PAIRED:
        print(_stats(lbl, per_stage_deltas[lbl]))
    print("  Informational deltas (NOT in sum; clean GPU/host signal):")
    for _, _, lbl in LIKE_FOR_LIKE_INFO:
        print(_stats(lbl, per_stage_deltas[lbl]))

    print()
    print(_stats("sum(contribs) - delta_ttft", sum_check)
          + "  (~0 means full attribution; non-zero hints at clock-offset slop)")

    if args.per_turn_out:
        args.per_turn_out.parent.mkdir(parents=True, exist_ok=True)
        cols = ([
            "conversation_id", "conversation_idx", "input_tokens",
            "cached_tokens_agg", "cached_tokens_disagg",
            "ttft_server_agg_ms", "ttft_server_disagg_ms", "delta_ttft_ms",
        ] + DISAGG_EXCLUSIVE
        + [lbl for _, _, lbl in LIKE_FOR_LIKE_PAIRED]
        + [lbl for _, _, lbl in LIKE_FOR_LIKE_INFO])
        with args.per_turn_out.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols)
            for r in out_rows:
                w.writerow([
                    f"{r[c]:.3f}" if isinstance(r[c], float) and not math.isnan(r[c])
                    else "" if r[c] is None or (isinstance(r[c], float) and math.isnan(r[c]))
                    else r[c]
                    for c in cols
                ])
        print(f"\nWrote per-turn TSV: {args.per_turn_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
