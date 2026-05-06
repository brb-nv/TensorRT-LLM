#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-iteration decode-loop analysis on an nsys SQLite trace.

Implements the perf-host-analysis Phase-1 verdict (M1/M2/M4) and Phase-2
NVTX/kernel breakdown over a steady-state window, derived from the per-step
``[Executor] _forward_step N: ...`` iteration markers TRT-LLM already emits.

The script is intentionally pure-stdlib (sqlite3 + statistics) so it can run
on a frontend without numpy/pandas.

Usage::

    python3 nvbugs_5615248/trtllm_bench/analyze_decode_loop.py \\
        --sqlite nvbugs_5615248/trtllm_bench/nsys_decode_loop_v1/pyt_beam10_decode.sqlite \\
        --skip-prefills 5

``--skip-prefills`` should match ``trtllm-bench --warmup``: the first N
context-only iterations are warmup and are excluded from the steady-state
window. The window ends at the last context-only iteration in the trace
(start of the last benchmark request) so the very last request's tail is
still inside the window.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass

NS_PER_MS = 1e6
NS_PER_US = 1e3

CTX_MARKER_RE = re.compile(
    r"\[Executor\] _forward_step (?P<idx>\d+): (?P<ctx>\d+) ctx reqs, (?P<gen>\d+) gen reqs"
)


@dataclass
class IterMarker:
    idx: int
    ctx: int
    gen: int
    start: int  # ns
    end: int  # ns


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_iter_markers(conn: sqlite3.Connection) -> list[IterMarker]:
    rows = conn.execute(
        """
        SELECT n.start, n.end, s.value AS name
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE s.value LIKE '%[Executor] _forward_step%'
          AND n.end IS NOT NULL
        ORDER BY n.start
        """
    ).fetchall()
    out: list[IterMarker] = []
    for r in rows:
        m = CTX_MARKER_RE.search(r["name"])
        if not m:
            continue
        out.append(
            IterMarker(
                idx=int(m.group("idx")),
                ctx=int(m.group("ctx")),
                gen=int(m.group("gen")),
                start=int(r["start"]),
                end=int(r["end"]),
            )
        )
    return out


def _pick_steady_window(
    markers: list[IterMarker], skip_prefills: int
) -> tuple[int, int, list[IterMarker]]:
    """Return (window_start_ns, window_end_ns, markers_in_window).

    Strategy:
      - Find all context-only markers (`ctx > 0`); these are prefill iterations.
      - Drop the first ``skip_prefills`` of them (warmup).
      - Window starts at the *end* of the (skip_prefills+1)-th prefill marker so
        we begin in steady-state generation.
      - Window ends at the *start* of the last prefill marker so the final
        benchmark request's tail is still fully captured.
    """
    prefill = [m for m in markers if m.ctx > 0]
    if len(prefill) <= skip_prefills:
        raise SystemExit(
            f"Only {len(prefill)} prefill iterations in trace, "
            f"need > {skip_prefills} (--skip-prefills)."
        )
    window_start = prefill[skip_prefills].end
    window_end = prefill[-1].start
    if window_end <= window_start:
        raise SystemExit(
            f"Empty steady-state window after skipping {skip_prefills} prefills; "
            f"increase --num_requests on the trace run."
        )
    inside = [m for m in markers if m.start >= window_start and m.end <= window_end]
    return window_start, window_end, inside


def _gpu_active_us_in_window(conn: sqlite3.Connection, t0: int, t1: int) -> float:
    """Approximate GPU active time = sum of kernel durations clipped to window.

    Approximate: assumes kernels have minimal overlap (true on a single GPU
    with one stream, mostly true with two streams). For our single-GPU
    --concurrency=1 workload this is within ~1% of a strict range-merge.
    """
    row = conn.execute(
        """
        SELECT SUM(MIN(end, ?) - MAX(start, ?)) AS active_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start < ? AND end > ?
        """,
        (t1, t0, t1, t0),
    ).fetchone()
    return (row["active_ns"] or 0) / NS_PER_US


def _cuda_launch_us_in_window(conn: sqlite3.Connection, t0: int, t1: int) -> float:
    # nsys 2026 emits versioned API names (e.g. ``cudaLaunchKernel_v7000``,
    # ``cudaLaunchKernelExC_v11060``, ``cuLaunchKernel``); match any of them.
    row = conn.execute(
        """
        SELECT SUM(MIN(r.end, ?) - MAX(r.start, ?)) AS launch_ns
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        WHERE (s.value LIKE 'cudaLaunchKernel%' OR s.value LIKE 'cuLaunchKernel%')
          AND r.start < ? AND r.end > ?
        """,
        (t1, t0, t1, t0),
    ).fetchone()
    return (row["launch_ns"] or 0) / NS_PER_US


def _nvtx_range_breakdown(
    conn: sqlite3.Connection, t0: int, t1: int
) -> list[tuple[str, int, float, float, float, float]]:
    """For every NVTX range that *starts* in [t0, t1], compute count + stats.

    Returns rows: (name, n, total_us, mean_us, median_us, p95_us)
    sorted by total descending.
    """
    rows = conn.execute(
        """
        SELECT s.value AS name, n.start, n.end
        FROM NVTX_EVENTS n
        JOIN StringIds s ON n.textId = s.id
        WHERE n.end IS NOT NULL
          AND n.start >= ?
          AND n.start < ?
        """,
        (t0, t1),
    ).fetchall()
    durations: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        durations[r["name"]].append((r["end"] - r["start"]) / NS_PER_US)
    out: list[tuple[str, int, float, float, float, float]] = []
    for name, ds in durations.items():
        ds_sorted = sorted(ds)
        n = len(ds_sorted)
        total = sum(ds_sorted)
        mean = total / n
        med = ds_sorted[n // 2] if n % 2 else 0.5 * (ds_sorted[n // 2 - 1] + ds_sorted[n // 2])
        p95_idx = max(0, min(n - 1, int(0.95 * n) - 1))
        p95 = ds_sorted[p95_idx]
        out.append((name, n, total, mean, med, p95))
    out.sort(key=lambda r: -r[2])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sqlite", required=True, help="Path to nsys exported sqlite trace")
    p.add_argument(
        "--skip-prefills",
        type=int,
        default=5,
        help="Number of warmup-request prefill iterations to skip (default: 5).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Top N NVTX ranges by total time to print (default: 25).",
    )
    p.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional substring(s) for NVTX range names to print (in addition to the top-N).",
    )
    args = p.parse_args()

    conn = _connect(args.sqlite)

    markers = _fetch_iter_markers(conn)
    if not markers:
        raise SystemExit("No '[Executor] _forward_step ...' NVTX markers found in trace.")

    t0, t1, in_window = _pick_steady_window(markers, args.skip_prefills)
    window_us = (t1 - t0) / NS_PER_US
    n_iters = len(in_window)

    gpu_active_us = _gpu_active_us_in_window(conn, t0, t1)
    cuda_launch_us = _cuda_launch_us_in_window(conn, t0, t1)
    gpu_idle_us = max(0.0, window_us - gpu_active_us)

    print("=" * 78)
    print("Steady-state decode-loop analysis")
    print("=" * 78)
    print(f"  Trace:        {args.sqlite}")
    print(f"  Skipped:      first {args.skip_prefills} prefill iterations (warmup)")
    print(f"  Window:       [{t0/1e9:.3f}s .. {t1/1e9:.3f}s] = {window_us/1000:.1f} ms")
    print(f"  Iterations:   {n_iters} forward_step events in window")
    n_ctx_in = sum(1 for m in in_window if m.ctx > 0)
    n_gen_in = sum(1 for m in in_window if m.ctx == 0 and m.gen > 0)
    print(f"  Iter mix:     {n_ctx_in} ctx-only, {n_gen_in} gen-only")
    print()
    print("--- Phase 1: Host-bound detection metrics (perf-host-analysis) ---")
    m1 = gpu_idle_us / window_us
    m4 = gpu_active_us / window_us
    m2 = cuda_launch_us / window_us
    print(f"  M1  GPU idle ratio          {m1:6.3f}   (threshold > 0.30 -> host-bound)")
    print(f"  M4  GPU utilization         {m4:6.3f}   (threshold < 0.60 -> host-bound)")
    print(f"  M2  cudaLaunchKernel ratio  {m2:6.3f}   (threshold > 0.10 -> launch-bound)")
    print()
    print(f"  GPU active:    {gpu_active_us/1000:8.2f} ms")
    print(f"  GPU idle:      {gpu_idle_us/1000:8.2f} ms")
    print(f"  cudaLaunchKernel total: {cuda_launch_us/1000:8.2f} ms")
    crossed = sum(int(x) for x in [m1 > 0.30, m4 < 0.60, m2 > 0.10])
    verdict = "YES" if crossed >= 2 else "NO"
    print(f"  Verdict (>=2 of 3 thresholds crossed): {verdict}  ({crossed}/3)")
    print()

    print("--- Phase 2: Per-iteration NVTX breakdown (steady state) ---")
    print(f"  {n_iters} iterations of which {n_gen_in} are gen-only")
    print(f"  All durations are PER ITERATION (= total in window / iter count)")
    print()

    rows = _nvtx_range_breakdown(conn, t0, t1)
    # Filter out the [Executor] _forward_step N: ... per-iter markers (one of each)
    # so the table isn't drowned in 1-instance entries.
    rows = [r for r in rows if "_forward_step " not in r[0]]

    iters_for_per_iter = max(1, n_iters)
    print(
        f"{'Range':<55} {'n':>6}  {'total':>10}  {'mean':>9}  {'med':>9}  {'p95':>9}  {'/iter':>9}"
    )
    print(
        f"{'':<55} {'':>6}  {'(ms)':>10}  {'(us)':>9}  {'(us)':>9}  {'(us)':>9}  {'(us)':>9}"
    )
    print("-" * 120)

    include_keys = list(args.include)
    seen = set()
    printed = 0
    for name, n, total_us, mean_us, med_us, p95_us in rows:
        if printed >= args.top_n and not any(k in name for k in include_keys):
            continue
        per_iter_us = total_us / iters_for_per_iter
        short = name[len("TensorRT-LLM:") :] if name.startswith("TensorRT-LLM:") else name
        print(
            f"{short:<55} {n:>6}  {total_us/1000:>10.2f}  {mean_us:>9.1f}  {med_us:>9.1f}  {p95_us:>9.1f}  {per_iter_us:>9.1f}"
        )
        seen.add(name)
        printed += 1
    if include_keys:
        for name, n, total_us, mean_us, med_us, p95_us in rows:
            if name in seen:
                continue
            if not any(k in name for k in include_keys):
                continue
            per_iter_us = total_us / iters_for_per_iter
            short = name[len("TensorRT-LLM:") :] if name.startswith("TensorRT-LLM:") else name
            print(
                f"{short:<55} {n:>6}  {total_us/1000:>10.2f}  {mean_us:>9.1f}  {med_us:>9.1f}  {p95_us:>9.1f}  {per_iter_us:>9.1f}"
            )

    print()
    print(
        "Notes: 'med' is per-call median (steady-state representative). 'p95' shows "
        "tail variance. '/iter' divides the in-window total by the in-window iteration "
        "count, so it sums to the per-iteration host budget for sibling NVTX ranges."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
