#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confirm that a `rwlt_subset_*` run is a strict prefix of a `rwlt_baseline`
run, in terms of which (conversation_id, conversation_idx) keys are issued and
in what order.

A clean match is the contract that lets us trust subset TTFT numbers as a
faithful preview of the full 30-trajectory run.

Usage:
    python3 scripts/verify_subset_against_baseline.py \\
        rwlt-results/agg_round1_0521/rwlt_requests.jsonl \\
        rwlt-results/agg_subset_lizhi_0521/rwlt_requests.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(p: Path) -> list[tuple[str, int]]:
    rows = [json.loads(line) for line in p.open() if json.loads(line).get("success")]
    rows.sort(key=lambda r: r["start_time"])
    return [(r["conversation_id"], r["conversation_idx"]) for r in rows]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_subset_against_baseline.py FULL_JSONL SUBSET_JSONL",
              file=sys.stderr)
        return 2
    full_path, sub_path = Path(sys.argv[1]), Path(sys.argv[2])
    full = _load(full_path)
    sub = _load(sub_path)

    print(f"full   {full_path}: {len(full)} successful turns")
    print(f"subset {sub_path}: {len(sub)} successful turns")

    if len(sub) > len(full):
        print(f"FAIL: subset is longer than full ({len(sub)} > {len(full)}).")
        return 1

    diverge = next((i for i, (a, b) in enumerate(zip(full, sub)) if a != b), None)
    if diverge is None:
        print(f"PASS: subset is a strict prefix of full ({len(sub)}/{len(full)} turns match).")
        unique_convs = len({k[0] for k in sub})
        print(f"      subset covers {unique_convs} unique conversations.")
        return 0

    print(f"FAIL: first divergence at position {diverge}/{len(sub)}")
    lo, hi = max(0, diverge - 2), min(len(full), diverge + 3)
    print(f"      full[{lo}:{hi}]   = {full[lo:hi]}")
    print(f"      subset[{lo}:{hi}] = {sub[lo:hi]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
