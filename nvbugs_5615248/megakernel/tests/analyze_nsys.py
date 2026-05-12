# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an nsys .sqlite report and print per-kernel timing.

Replicates `nsys stats --report cuda_gpu_kern_sum` for environments where
the `nsys` CLI is not installed but the sqlite report is available.

Usage:
    python tests/analyze_nsys.py mega_nsys.sqlite
"""
import argparse
import os
import sqlite3
import sys


def fmt_us(ns: float) -> str:
    if ns < 1000:
        return f"{ns:.1f} ns"
    us = ns / 1000.0
    if us < 1000:
        return f"{us:.2f} us"
    return f"{us/1000:.3f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.sqlite_path):
        sys.exit(f"missing: {args.sqlite_path}")

    con = sqlite3.connect(args.sqlite_path)
    cur = con.cursor()

    # The kernel events live in CUPTI_ACTIVITY_KIND_KERNEL with a name id that
    # resolves via the StringIds table. Schema varies slightly across nsys
    # versions; we try the canonical layout first.
    try:
        rows = cur.execute("""
            SELECT s.value AS name,
                   COUNT(*) AS calls,
                   SUM(k.end - k.start) AS total_ns,
                   AVG(k.end - k.start) AS avg_ns,
                   MIN(k.end - k.start) AS min_ns,
                   MAX(k.end - k.start) AS max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            GROUP BY s.value
            ORDER BY total_ns DESC
        """).fetchall()
    except sqlite3.OperationalError as e:
        sys.exit(f"sql failed: {e}\nTry inspecting tables manually:\n"
                 f"  sqlite3 {args.sqlite_path} '.tables'")

    if not rows:
        sys.exit("no kernel events found")

    total_all = sum(r[2] for r in rows)
    print(f"Total GPU kernel time: {fmt_us(total_all)}")
    print(f"{'Name':<60s} {'Calls':>7s} {'Total':>10s} {'Avg':>10s} "
          f"{'Min':>9s} {'Max':>9s} {'% of total':>10s}")
    print("-" * 122)
    for name, calls, total_ns, avg_ns, min_ns, max_ns in rows[:args.limit]:
        pct = 100.0 * total_ns / total_all
        # Demangle is too much work; just truncate.
        n = name if len(name) <= 60 else (name[:57] + "...")
        print(f"{n:<60s} {calls:>7d} {fmt_us(total_ns):>10s} "
              f"{fmt_us(avg_ns):>10s} {fmt_us(min_ns):>9s} {fmt_us(max_ns):>9s} "
              f"{pct:>9.2f}%")

    print()
    # Also print megakernel-specific stats if present.
    cur.execute("""
        SELECT s.value, COUNT(*), SUM(k.end - k.start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        WHERE s.value LIKE '%tinyllama_megakernel%'
        GROUP BY s.value
    """)
    mk = cur.fetchall()
    if mk:
        for name, calls, total_ns in mk:
            avg = total_ns / max(1, calls)
            print(f"megakernel: {fmt_us(avg)} per call x {calls} calls "
                  f"= {fmt_us(total_ns)} total")


if __name__ == "__main__":
    main()
