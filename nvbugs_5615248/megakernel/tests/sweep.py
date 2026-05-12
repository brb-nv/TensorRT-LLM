# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M6 tile sweep: rebuild + bench across (BM, BN, BK, NSTAGES, WARP_ROWS, WARP_COLS).

For each configuration this script:
  1. Sets the MEGAKERNEL_* env vars
  2. Reinstalls the extension via `pip install -e . --no-build-isolation --force-reinstall`
  3. Runs `tests/bench.py --backend megakernel --iters 200` (short, just for sweep)
  4. Logs the result to sweep_results.csv

Run from the repo root:
    python tests/sweep.py --warmup 50 --iters 200
or limit to a subset:
    python tests/sweep.py --filter '128_128_32_3_*'

Manual tile constraints (enforced at compile time by static_asserts):
    BM % (WARP_ROWS * 16) == 0
    BN % (WARP_COLS * 16) == 0
    BK % 16 == 0
    WARP_ROWS * WARP_COLS == BLOCK_SIZE / 32   (= 8 for the default 256-thread block)

So with BLOCK_SIZE=256 only 8-warp configurations work: 8x1, 4x2, 2x4, 1x8.

WARNING: this script measures perf only; it does NOT validate numerics.
Always run `pytest tests/test_numerics.py` against the winning config before
promoting it to the default in `setup.py`. (The first round of this sweep
crowned `64_128_32_3_2_4` at 11.1 ms based on a bug in `grid_gemm` that
skipped every m_tile != 0 -- the kernel was producing the right timing for
the wrong amount of work. The bug has since been fixed.)
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from itertools import product

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# All 8-warp tile combos that satisfy the static_asserts.
DEFAULT_CONFIGS = [
    # (BM,  BN,  BK,  NSTAGES, WARP_ROWS, WARP_COLS)
    (128, 128, 32, 3, 4, 2),     # plan default
    (128, 128, 32, 2, 4, 2),
    (128, 128, 32, 4, 4, 2),
    (128, 128, 64, 3, 4, 2),     # bigger BK = fewer K-loop iters
    (128, 128, 16, 3, 4, 2),     # smaller BK = more pipelining
    (128, 128, 32, 3, 2, 4),     # transposed warp grid
    (128, 128, 32, 3, 8, 1),     # all-rows warp grid
    (128,  64, 32, 3, 4, 2),     # smaller N-tile (helps N=2048,2560 utilization)
    ( 64, 128, 32, 3, 2, 4),     # smaller M-tile (we only have M=128, so 2 m-blocks)
]


def cfg_id(c) -> str:
    return "_".join(str(x) for x in c)


def run_one(cfg, warmup, iters, log_dir):
    BM, BN, BK, NS, WR, WC = cfg
    env = os.environ.copy()
    env.update({
        "MEGAKERNEL_BM": str(BM),
        "MEGAKERNEL_BN": str(BN),
        "MEGAKERNEL_BK": str(BK),
        "MEGAKERNEL_NSTAGES": str(NS),
        "MEGAKERNEL_WARP_ROWS": str(WR),
        "MEGAKERNEL_WARP_COLS": str(WC),
    })

    log_path = os.path.join(log_dir, f"cfg_{cfg_id(cfg)}.log")
    with open(log_path, "w") as logf:
        logf.write(f"=== cfg = {cfg} ===\n")
        logf.flush()
        # 1. rebuild
        rc = subprocess.call(
            [sys.executable, "-m", "pip", "install", "-e", ".",
             "--no-build-isolation", "--force-reinstall", "--quiet"],
            cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
        if rc != 0:
            return {"cfg": cfg, "status": "build_fail", "ms": None}
        # 2. bench
        # Use synthetic weights to skip HF disk load each rebuild (~3 s saved per cfg).
        rc = subprocess.call(
            [sys.executable, "tests/bench.py", "--backend", "megakernel",
             "--use-synthetic",
             "--warmup", str(warmup), "--iters", str(iters)],
            cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    # parse "lucebox megakernel (synth)        mean=N.NNN ..." or any label
    # starting with "lucebox megakernel".
    with open(log_path) as f:
        text = f.read()
    m = re.search(r"lucebox megakernel[^\n]*?mean=\s*([\d.]+)", text)
    if not m:
        return {"cfg": cfg, "status": "run_fail", "ms": None}
    return {"cfg": cfg, "status": "ok", "ms": float(m.group(1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--filter", default="*", help="glob-style filter on cfg_id (e.g. '128_*_32_*')")
    ap.add_argument("--csv", default="sweep_results.csv")
    args = ap.parse_args()

    import fnmatch
    configs = [c for c in DEFAULT_CONFIGS if fnmatch.fnmatch(cfg_id(c), args.filter)]
    print(f"sweeping {len(configs)} configurations: {[cfg_id(c) for c in configs]}")

    log_dir = os.path.join(ROOT, "sweep_logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(ROOT, args.csv)

    results = []
    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["BM", "BN", "BK", "NSTAGES", "WARP_ROWS", "WARP_COLS",
                    "status", "mean_ms"])
        for cfg in configs:
            t0 = time.time()
            r = run_one(cfg, args.warmup, args.iters, log_dir)
            dt = time.time() - t0
            print(f"  {cfg_id(cfg):<22s} status={r['status']:<10s} "
                  f"mean={r['ms']!s:<10s}  ({dt:.1f}s)")
            row = list(cfg) + [r["status"], r["ms"] if r["ms"] is not None else ""]
            w.writerow(row)
            cf.flush()
            results.append(r)

    results = [r for r in results if r["ms"] is not None]
    results.sort(key=lambda r: r["ms"])
    print()
    print("Top configs by mean latency:")
    for r in results[:5]:
        print(f"  {cfg_id(r['cfg']):<22s} {r['ms']:.3f} ms")


if __name__ == "__main__":
    main()
