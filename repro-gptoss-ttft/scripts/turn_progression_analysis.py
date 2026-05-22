#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round 7 (2026-05-22) analysis: turn-progression of disagg-vs-agg TTFT.

Tests the hypothesis (FINDINGS_RWLT_0522.md): TTFT delta grows monotonically
with turn-number / cumulative conversation prefix length, NOT with the bytes
the cache transceiver actually moves.

Inputs (per session):
    --agg-dir DIR               Aggregated run dir with:
                                    perf_metrics.json
                                    rwlt_requests.jsonl
    --disagg-dir DIR            Disaggregated run dir with:
                                    perf_metrics_proxy.json
                                    rwlt_requests.jsonl
                                Optional:
                                    disagg_kvcache_time/rank_0_recv.csv
                                    (used for transceiver per-phase breakdown)

Pairing strategy (valid at concurrency=1, which RWLT enforces):
    * Sort each session's rwlt_requests.jsonl by `start_time` and pair the
      i-th successful perf_metrics record with the i-th successful RWLT row.
    * Join across agg<->disagg by (conversation_id, conversation_idx).

Outputs (under --out-dir):
    turn_progression.tsv        One row per joined request, sortable by turn
                                or by conv. Columns include:
                                    conv, turn, fresh, reused_tok,
                                    num_total_blk, num_new_blk, num_reused_blk,
                                    kv_size_bytes, kv_xfer_ms, wait_ms,
                                    ttft_agg_ms, ttft_disagg_ms, delta_ttft_ms,
                                    (optional) xfer_prep_ms, xfer_preproc_ms,
                                    xfer_transmissions_ms, xfer_postproc_ms

    correlations.txt            Pooled and per-trajectory Pearson r between
                                kv_xfer_ms / delta_ttft_ms and each of
                                {num_new_blk, num_total_blk, num_reused_blk,
                                 kv_size_bytes, turn, reused_tok}.

    growth_summary.txt          Per-trajectory early-half vs late-half
                                medians (delta_ttft, kv_xfer, reused_blk) and
                                per-bucket xfer_ms/new_blk ratio.

Usage:
    python3 scripts/turn_progression_analysis.py \\
        --agg-dir   rwlt-results/agg_round7_si1_top5_0522 \\
        --disagg-dir rwlt-results/disagg_round7_si1_top5_0522 \\
        --out-dir   rwlt-results/turn_progression_round7_si1
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _ms(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return float("nan")
    try:
        return (float(b) - float(a)) * 1000.0
    except (TypeError, ValueError):
        return float("nan")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy > 0 else float("nan")


def _load_rwlt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("success"):
                rows.append(r)
    rows.sort(key=lambda r: r.get("start_time", 0.0))
    return rows


def _load_perf(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def _load_transceiver_csv(path: Path) -> list[dict[str, float]]:
    """Optional per-request transceiver phase breakdown (rank_0_recv.csv).

    Each row: RequestID, RequestInfo, Preparation, Preprocess, Transmissions,
              Postprocess, Delay, Duration, Bandwidth(Gbps) [, more measures].
    Rows are in transceiver-arrival order. At concurrency=1 this matches RWLT
    completion order, so we pair by ordinal.
    """
    out: list[dict[str, float]] = []
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append({
                "Preparation": float(row.get("Preparation", "0") or 0),
                "Preprocess": float(row.get("Preprocess", "0") or 0),
                "Transmissions": float(row.get("Transmissions", "0") or 0),
                "Postprocess": float(row.get("Postprocess", "0") or 0),
                "Delay": float(row.get("Delay", "0") or 0),
                "Duration": float(row.get("Duration", "0") or 0),
            })
    return out


def _build_agg_index(perf: list[dict[str, Any]],
                     rwlt: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    n = min(len(perf), len(rwlt))
    if len(perf) != len(rwlt):
        print(f"[warn] agg: perf rows ({len(perf)}) != rwlt success rows "
              f"({len(rwlt)}); pairing by index, tail may misalign.")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for i in range(n):
        p = perf[i]
        r = rwlt[i]
        pm = p.get("perf_metrics", {}) or {}
        tm = pm.get("timing_metrics", {}) or {}
        sa = tm.get("server_arrival_time")
        sft = tm.get("server_first_token_time")
        out[(r["conversation_id"], r["conversation_idx"])] = {
            "ttft_agg_ms": _ms(sa, sft),
        }
    return out


def _build_disagg_index(perf: list[dict[str, Any]],
                        rwlt: list[dict[str, Any]],
                        transceiver: list[dict[str, float]]
                        ) -> dict[tuple[str, int], dict[str, Any]]:
    n = min(len(perf), len(rwlt))
    if len(perf) != len(rwlt):
        print(f"[warn] disagg: perf rows ({len(perf)}) != rwlt success rows "
              f"({len(rwlt)}); pairing by index, tail may misalign.")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for i in range(n):
        p = perf[i]
        r = rwlt[i]
        gpm = (p.get("gen_perf_metrics", {}) or {}).get("perf_metrics", {}) or {}
        gtm = gpm.get("timing_metrics", {}) or {}
        kvm = gpm.get("kv_cache_metrics", {}) or {}

        da = p.get("disagg_server_arrival_time")
        dft = p.get("disagg_server_first_token_time")

        row: dict[str, Any] = {
            "conv": r["conversation_id"],
            "turn": r["conversation_idx"],
            "fresh": r["server_input_tokens"] - r["server_cached_tokens"],
            "reused_tok": r["server_cached_tokens"],
            "num_total_blk": kvm.get("num_total_allocated_blocks"),
            "num_new_blk": kvm.get("num_new_allocated_blocks"),
            "num_reused_blk": kvm.get("num_reused_blocks"),
            "num_missed_blk": kvm.get("num_missed_blocks"),
            "kv_size_bytes": gtm.get("kv_cache_size"),
            "kv_xfer_ms": _ms(gtm.get("kv_cache_transfer_start"),
                              gtm.get("kv_cache_transfer_end")),
            "wait_ms": _ms(gtm.get("arrival_time"),
                           gtm.get("first_scheduled_time")),
            "ttft_disagg_ms": _ms(da, dft),
        }

        if i < len(transceiver):
            t = transceiver[i]
            row.update({
                "xfer_prep_ms": t["Preparation"],
                "xfer_preproc_ms": t["Preprocess"],
                "xfer_transmissions_ms": t["Transmissions"],
                "xfer_postproc_ms": t["Postprocess"],
            })

        out[(r["conversation_id"], r["conversation_idx"])] = row
    return out


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stats(vals: list[float]) -> str:
    valid = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return "n=0"
    return (f"n={len(valid):>3} mean={statistics.fmean(valid):>8.1f} "
            f"p50={_percentile(valid, 50):>8.1f} "
            f"p90={_percentile(valid, 90):>8.1f} "
            f"p99={_percentile(valid, 99):>8.1f} "
            f"max={max(valid):>8.1f}")


def _emit_correlations(rows: list[dict[str, Any]], out: Path) -> None:
    """Pearson r between (kv_xfer_ms, delta_ttft_ms) and predictors."""
    predictors = ["num_new_blk", "num_total_blk", "num_reused_blk",
                  "kv_size_bytes", "turn", "reused_tok", "fresh"]
    targets = ["kv_xfer_ms", "delta_ttft_ms", "wait_ms"]

    def _valid(rows: list[dict[str, Any]], target: str, pred: str
               ) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for r in rows:
            x = r.get(pred)
            y = r.get(target)
            if x is None or y is None:
                continue
            if isinstance(x, float) and math.isnan(x):
                continue
            if isinstance(y, float) and math.isnan(y):
                continue
            xs.append(float(x))
            ys.append(float(y))
        return xs, ys

    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_conv[r["conv"]].append(r)

    lines: list[str] = []
    lines.append("=== Pearson r ===")
    lines.append("")
    lines.append(f"POOLED (n={len(rows)})")
    header = f"  {'target':>16}  " + "  ".join(f"{p:>14}" for p in predictors)
    lines.append(header)
    for tgt in targets:
        cells = []
        for p in predictors:
            xs, ys = _valid(rows, tgt, p)
            cells.append(f"{_pearson(xs, ys):>+14.3f}")
        lines.append(f"  {tgt:>16}  " + "  ".join(cells))

    lines.append("")
    lines.append("PER-TRAJECTORY (n>=5)")
    for tgt in targets:
        lines.append(f"  target = {tgt}")
        lines.append(f"    {'conv':>8}  {'n':>3}  " +
                     "  ".join(f"{p:>14}" for p in predictors))
        for conv in sorted(by_conv):
            recs = by_conv[conv]
            if len(recs) < 5:
                continue
            cells = []
            for p in predictors:
                xs, ys = _valid(recs, tgt, p)
                cells.append(f"{_pearson(xs, ys):>+14.3f}")
            lines.append(f"    {conv.split('-')[-1]:>8}  {len(recs):>3}  " +
                         "  ".join(cells))
        lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def _emit_growth_summary(rows: list[dict[str, Any]], out: Path) -> None:
    """Per-trajectory early-half vs late-half median + per-tertile xfer ratio.

    The tertile bucketing is by `num_reused_blk` so we can see whether the
    per-block transfer cost (`kv_xfer_ms / num_new_blk`) grows with gen-side
    cached prefix length.
    """
    by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_conv[r["conv"]].append(r)

    lines: list[str] = []
    lines.append("=== Per-trajectory early-half vs late-half median (ms) ===")
    lines.append(f"  {'conv':>8} {'n':>3}  "
                 f"{'delta_e/l':>20}  {'xfer_e/l':>20}  "
                 f"{'reused_blk_e/l':>20}")
    for conv in sorted(by_conv):
        recs = sorted(by_conv[conv], key=lambda r: r["turn"])
        if len(recs) < 6:
            continue
        half = len(recs) // 2
        early, late = recs[:half], recs[half:]

        def med(rs: list[dict[str, Any]], key: str) -> float:
            vs = [r[key] for r in rs if r.get(key) is not None
                  and not (isinstance(r[key], float) and math.isnan(r[key]))]
            return statistics.median(vs) if vs else float("nan")

        de = med(early, "delta_ttft_ms")
        dl = med(late, "delta_ttft_ms")
        xe = med(early, "kv_xfer_ms")
        xl = med(late, "kv_xfer_ms")
        re = med(early, "num_reused_blk")
        rl = med(late, "num_reused_blk")
        lines.append(
            f"  {conv.split('-')[-1]:>8} {len(recs):>3}  "
            f"{de:>+8.1f}/{dl:>+9.1f}  "
            f"{xe:>+8.1f}/{xl:>+9.1f}  "
            f"{re:>+8.0f}/{rl:>+9.0f}")

    lines.append("")
    lines.append("=== Per-tertile xfer cost (ms / block) by num_reused_blk ===")
    lines.append("  If per-block cost is byte-driven, ratio is ~flat.")
    lines.append("  If per-block bookkeeping scales with cached prefix length,")
    lines.append("  ratio grows monotonically from low->high tertile.")
    lines.append("")
    lines.append(f"  {'conv':>8}  {'bucket':>8}  {'n':>3}  "
                 f"{'med xfer/new_blk(ms)':>22}")
    for conv in sorted(by_conv):
        recs = [r for r in by_conv[conv]
                if r.get("num_new_blk") and r.get("num_reused_blk") is not None
                and r.get("kv_xfer_ms") is not None
                and r["num_new_blk"] > 0
                and not math.isnan(r["kv_xfer_ms"])]
        if len(recs) < 6:
            continue
        recs.sort(key=lambda r: r["num_reused_blk"])
        third = len(recs) // 3
        if third < 2:
            continue
        buckets = [("low ", recs[:third]),
                   ("mid ", recs[third:2 * third]),
                   ("high", recs[2 * third:])]
        for tag, b in buckets:
            ratios = [r["kv_xfer_ms"] / r["num_new_blk"] for r in b]
            if not ratios:
                continue
            lines.append(f"  {conv.split('-')[-1]:>8}  {tag:>8}  "
                         f"{len(ratios):>3}  "
                         f"{statistics.median(ratios):>22.3f}")

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agg-dir", type=Path, required=True)
    p.add_argument("--disagg-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--agg-perf", default="perf_metrics.json")
    p.add_argument("--disagg-perf", default="perf_metrics_proxy.json")
    p.add_argument("--transceiver-csv",
                   default="disagg_kvcache_time/rank_0_recv.csv",
                   help="relative to --disagg-dir; optional")
    args = p.parse_args()

    agg_perf = _load_perf(args.agg_dir / args.agg_perf)
    agg_rwlt = _load_rwlt(args.agg_dir / "rwlt_requests.jsonl")
    agg_idx = _build_agg_index(agg_perf, agg_rwlt)

    disagg_perf = _load_perf(args.disagg_dir / args.disagg_perf)
    disagg_rwlt = _load_rwlt(args.disagg_dir / "rwlt_requests.jsonl")
    transceiver = _load_transceiver_csv(args.disagg_dir / args.transceiver_csv)
    disagg_idx = _build_disagg_index(disagg_perf, disagg_rwlt, transceiver)

    common = sorted(set(agg_idx) & set(disagg_idx), key=lambda k: (k[0], k[1]))
    only_agg = set(agg_idx) - set(disagg_idx)
    only_disagg = set(disagg_idx) - set(agg_idx)
    print(f"agg rows={len(agg_idx)}  disagg rows={len(disagg_idx)}  "
          f"common={len(common)}  agg_only={len(only_agg)}  "
          f"disagg_only={len(only_disagg)}  "
          f"transceiver_rows={len(transceiver)}")

    rows: list[dict[str, Any]] = []
    for key in common:
        a = agg_idx[key]
        d = dict(disagg_idx[key])
        a_ttft = a["ttft_agg_ms"]
        d_ttft = d["ttft_disagg_ms"]
        d["ttft_agg_ms"] = a_ttft
        d["delta_ttft_ms"] = (d_ttft - a_ttft
                              if not (math.isnan(a_ttft) or math.isnan(d_ttft))
                              else float("nan"))
        rows.append(d)

    cols = ["conv", "turn", "fresh", "reused_tok",
            "num_total_blk", "num_new_blk", "num_reused_blk", "num_missed_blk",
            "kv_size_bytes", "kv_xfer_ms", "wait_ms",
            "ttft_agg_ms", "ttft_disagg_ms", "delta_ttft_ms",
            "xfer_prep_ms", "xfer_preproc_ms",
            "xfer_transmissions_ms", "xfer_postproc_ms"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.out_dir / "turn_progression.tsv"
    with tsv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float)
                            and not math.isnan(v) else v)
                        for k, v in r.items()})
    print(f"wrote {tsv_path}  rows={len(rows)}")

    _emit_correlations(rows, args.out_dir / "correlations.txt")
    _emit_growth_summary(rows, args.out_dir / "growth_summary.txt")

    print()
    print("=== headline summary ===")
    print(f"  ttft_agg     : {_stats([r['ttft_agg_ms'] for r in rows])}")
    print(f"  ttft_disagg  : {_stats([r['ttft_disagg_ms'] for r in rows])}")
    print(f"  delta_ttft   : {_stats([r['delta_ttft_ms'] for r in rows])}")
    print(f"  kv_xfer_ms   : {_stats([r['kv_xfer_ms'] for r in rows])}")
    print(f"  num_new_blk  : {_stats([r['num_new_blk'] for r in rows if r.get('num_new_blk') is not None])}")
    print(f"  num_reused_blk: {_stats([r['num_reused_blk'] for r in rows if r.get('num_reused_blk') is not None])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
