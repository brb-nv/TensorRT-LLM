#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trajectory + per-turn breakdown report.

Reads breakdown.tsv produced by perf_metrics_breakdown.py and prints:
  (1) A per-trajectory rollup: one row per conversation_id, with count of
      turns and median/sum across the selected metrics.
  (2) Per-trajectory detail: per-turn table for each conversation_id, in
      conversation_idx order.

By default exposes the most useful columns for TTFT investigation, but the
full breakdown is preserved in the TSV (passed to --tsv).

Usage:
    # Agg (single TSV)
    python3 scripts/per_trajectory_report.py \\
        --tsv rwlt-results/agg_round4_fixed_top5_0521/breakdown.tsv \\
        --label agg

    # Disagg (single TSV)
    python3 scripts/per_trajectory_report.py \\
        --tsv rwlt-results/disagg_round4_genpp4si20_top5signed_0521/breakdown.tsv \\
        --label disagg --layout disagg

    # Restrict to one trajectory:
    python3 scripts/per_trajectory_report.py --tsv ... --conv aa-rwlt-coding-agent-056

    # Side-by-side (joins agg + disagg on conversation_id, conversation_idx):
    python3 scripts/per_trajectory_report.py \\
        --tsv rwlt-results/agg_round4_fixed_top5_0521/breakdown.tsv \\
        --tsv-disagg rwlt-results/disagg_round4_genpp4si20_top5signed_0521/breakdown.tsv

The output is plain text (markdown-friendly). Pipe to less -RS for wide
terminals, or use --out to dump to a file.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


def _f(s: str) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except ValueError:
        return None


def _load(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _p50(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _p99(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * 0.99
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[int(f)] + (s[int(c)] - s[int(f)]) * (k - f)


def _fmt(v: Optional[float], unit: str = "ms", w: int = 9) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'-':>{w}}"
    if unit == "ms":
        return f"{v:>{w}.1f}"
    if unit == "k":
        return f"{v/1000.0:>{w}.1f}k"
    if unit == "%":
        return f"{v*100:>{w-1}.1f}%"
    if unit == "MB":
        return f"{v:>{w-2}.1f}MB"
    return f"{v:>{w}.1f}"


AGG_DETAIL_COLS = [
    ("idx",            "conversation_idx", "int", 4),
    ("isl",            "input_tokens",     "int", 7),
    ("cache_hit",      "cached_tokens",    "int", 7),
    ("osl",            "output_tokens",    "int", 5),
    ("reuse",          "kv_block_reuse_ratio", "%", 6),
    ("ttft_srv",       "ttft_server_ms",   "ms",  9),
    ("ttft_cli",       "ttft_client_ms",   "ms",  9),
    ("preproc",        "srv_preproc_ms",   "ms",  8),
    ("queue",          "queue_ms",         "ms",  7),
    ("prefill",        "prefill_ms",       "ms",  8),
    ("postproc",       "srv_postproc_ms",  "ms",  8),
    ("decode",         "decode_ms",        "ms",  9),
    ("e2e",            "end_to_end_ms",    "ms",  9),
]

DISAGG_DETAIL_COLS = [
    ("idx",            "conversation_idx", "int", 4),
    ("isl",            "input_tokens",     "int", 7),
    ("cache_hit",      "cached_tokens",    "int", 7),
    ("osl",            "output_tokens",    "int", 5),
    ("reuse",          "ctx_kv_block_reuse_ratio", "%", 6),
    ("kv_MB",          "kv_cache_mb",      "MB", 8),
    ("ttft_srv",       "ttft_server_ms",   "ms",  9),
    ("ttft_cli",       "ttft_client_ms",   "ms",  9),
    ("pr->ctx",        "proxy_to_ctx_ms",  "ms",  7),
    ("ctx_prep",       "ctx_preproc_ms",   "ms",  8),
    ("ctx_q",          "ctx_queue_ms",     "ms",  6),
    ("ctx_pref",       "ctx_prefill_ms",   "ms",  8),
    ("ctx_post",       "ctx_postproc_ms",  "ms",  8),
    ("relay",          "relay_hop_ms",     "ms",  6),
    ("gen_prep",       "gen_preproc_ms",   "ms",  8),
    ("gen_q",          "gen_queue_ms",     "ms",  8),
    ("kvx_gen",        "kv_transfer_gen_ms","ms", 8),
    ("g1stdec",        "gen_first_decode_ms","ms",7),
    ("gen_post",       "gen_postproc_ms",  "ms",  8),
    ("pr->cli",        "proxy_to_client_ms","ms", 7),
    ("gen_dec",        "gen_decode_ms",    "ms",  9),
    ("e2e",            "end_to_end_ms",    "ms",  9),
]


SUMMARY_COLS_AGG = [
    "ttft_server_ms", "queue_ms", "prefill_ms", "srv_postproc_ms",
    "decode_ms", "end_to_end_ms",
]
SUMMARY_COLS_DISAGG = [
    "ttft_server_ms", "proxy_to_ctx_ms", "ctx_preproc_ms", "ctx_queue_ms",
    "ctx_prefill_ms", "ctx_postproc_ms", "relay_hop_ms", "gen_preproc_ms",
    "gen_queue_ms", "kv_transfer_gen_ms", "gen_first_decode_ms",
    "gen_postproc_ms", "proxy_to_client_ms", "gen_decode_ms", "end_to_end_ms",
]


def render_detail(rows: list[dict[str, str]],
                  cols: list[tuple[str, str, str, int]],
                  out) -> None:
    header = "  " + " ".join(f"{h:>{w}}" for h, _, _, w in cols)
    out.write(header + "\n")
    out.write("  " + " ".join("-" * w for _, _, _, w in cols) + "\n")
    for r in rows:
        line = []
        for label, key, kind, w in cols:
            v = r.get(key, "")
            if kind == "int":
                if v in ("", "-1", None):
                    line.append(f"{'-':>{w}}")
                else:
                    try:    line.append(f"{int(float(v)):>{w}d}")
                    except: line.append(f"{'-':>{w}}")
            else:
                line.append(_fmt(_f(v), kind, w))
        out.write("  " + " ".join(line) + "\n")


def group_by_conv(rows: list[dict[str, str]]) -> "OrderedDict[str, list[dict[str, str]]]":
    out: "OrderedDict[str, list[dict[str, str]]]" = OrderedDict()
    for r in rows:
        cid = r.get("conversation_id", "")
        out.setdefault(cid, []).append(r)
    # Sort turns within each conversation by conversation_idx.
    for cid in out:
        out[cid].sort(key=lambda r: int(r.get("conversation_idx", 0)))
    return out


def render_rollup(conv_rows: "OrderedDict[str, list[dict[str, str]]]",
                  cols: list[str], out, label: str) -> None:
    """Per-trajectory rollup: median + max + sum of TTFT per conversation."""
    out.write(f"\n=== {label}: per-trajectory rollup ({len(conv_rows)} convs) ===\n")
    header_keys = ["conversation_id", "turns",
                   "isl_p50", "isl_max",
                   "cache_p50", "cache_p99",
                   "ttft_p50", "ttft_p99", "ttft_sum_s",
                   "decode_p50", "e2e_sum_s"]
    widths = [38, 5, 8, 8, 7, 7, 8, 8, 9, 8, 8]
    out.write("  " + " ".join(f"{h:>{w}}" for h, w in zip(header_keys, widths)) + "\n")
    out.write("  " + " ".join("-" * w for w in widths) + "\n")

    decode_key = "decode_ms" if "decode_ms" in (cols + []) or True else None
    for cid, turns in conv_rows.items():
        ttft = [v for v in (_f(t.get("ttft_server_ms")) for t in turns) if v is not None]
        isl  = [v for v in (_f(t.get("input_tokens"))   for t in turns) if v is not None]
        cache= [v for v in (_f(t.get("cached_tokens"))  for t in turns) if v is not None]
        dec  = [v for v in (_f(t.get("decode_ms") or t.get("gen_decode_ms")) for t in turns) if v is not None]
        e2e  = [v for v in (_f(t.get("end_to_end_ms")) for t in turns) if v is not None]
        vals = [
            cid,
            f"{len(turns):>5d}",
            f"{int(_p50(isl)):>8d}" if isl else f"{'-':>8}",
            f"{int(max(isl)):>8d}" if isl else f"{'-':>8}",
            f"{int(_p50(cache)):>7d}" if cache else f"{'-':>7}",
            f"{int(_p99(cache)):>7d}" if cache else f"{'-':>7}",
            f"{_p50(ttft):>8.1f}" if ttft else f"{'-':>8}",
            f"{_p99(ttft):>8.1f}" if ttft else f"{'-':>8}",
            f"{sum(ttft)/1000.0:>9.2f}" if ttft else f"{'-':>9}",
            f"{_p50(dec):>8.1f}" if dec else f"{'-':>8}",
            f"{sum(e2e)/1000.0:>8.2f}" if e2e else f"{'-':>8}",
        ]
        out.write("  " + " ".join(f"{v:>{w}}" if i==0 else v for i,(v,w) in enumerate(zip(vals, widths))) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", required=True, type=Path,
                   help="primary breakdown.tsv (agg or disagg)")
    p.add_argument("--tsv-disagg", type=Path,
                   help="optional second TSV (treated as disagg) for side-by-side")
    p.add_argument("--layout", choices=["agg", "disagg"],
                   help="layout of --tsv; auto-detected from columns if omitted")
    p.add_argument("--label", default="primary",
                   help="label used in headings")
    p.add_argument("--conv", action="append", default=[],
                   help="limit detail to one or more conversation_ids "
                        "(repeatable; default: all)")
    p.add_argument("--top-n", type=int, default=0,
                   help="if >0, show detail only for the top-N conversations "
                        "ranked by median TTFT (default: 0 = all)")
    p.add_argument("--no-detail", action="store_true",
                   help="suppress per-turn detail tables, print rollup only")
    p.add_argument("--out", type=Path, help="write report to file (default stdout)")
    args = p.parse_args()

    rows = _load(args.tsv)
    if not rows:
        print(f"ERROR: empty TSV: {args.tsv}", file=sys.stderr)
        return 1

    if args.layout is None:
        args.layout = "disagg" if "proxy_to_ctx_ms" in rows[0] else "agg"
    detail_cols = DISAGG_DETAIL_COLS if args.layout == "disagg" else AGG_DETAIL_COLS
    summary_cols = SUMMARY_COLS_DISAGG if args.layout == "disagg" else SUMMARY_COLS_AGG

    out = open(args.out, "w") if args.out else sys.stdout
    try:
        out.write(f"# {args.label} ({args.layout}) -- {args.tsv}\n")
        out.write(f"# turns paired: {len(rows)}\n")

        # Filter to conv if requested.
        if args.conv:
            keep = set(args.conv)
            primary_rows = [r for r in rows if r.get("conversation_id") in keep]
        else:
            primary_rows = rows
        conv_rows = group_by_conv(primary_rows)

        if args.top_n > 0:
            # Rank by median ttft_server_ms.
            ranked = sorted(conv_rows.items(),
                            key=lambda kv: -_p50([_f(t.get("ttft_server_ms"))
                                                  for t in kv[1]
                                                  if _f(t.get("ttft_server_ms")) is not None]
                                                 or [0.0]))[:args.top_n]
            conv_rows = OrderedDict(ranked)

        render_rollup(conv_rows, summary_cols, out, args.label)

        if not args.no_detail:
            for cid, turns in conv_rows.items():
                out.write(f"\n--- {cid} ({len(turns)} turns) ---\n")
                render_detail(turns, detail_cols, out)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
