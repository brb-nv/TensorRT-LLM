#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse ctx-worker iter= logs to split host vs device step time and correlate
with prefill load. Tells us whether the prefill TTFT is GPU-engine compute or
host-side stalls (e.g. KV reuse-tree walk).

Usage: analyze_ctx_iters.py <ctx*.log> [...]
"""
import re
import sys
import statistics as st

PAT = re.compile(
    r"iter = (\d+).*?host_step_time = ([\d.]+)ms.*?"
    r"prev_device_step_time = ([\d.]+)ms.*?"
    r"num_scheduled_requests: (\d+).*?"
    r"'num_ctx_requests': (\d+), 'num_ctx_tokens': (\d+), "
    r"'num_generation_tokens': (\d+)")


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    for path in sys.argv[1:]:
        host, dev, nctxtok, nsched = [], [], [], []
        ctx_steps = 0
        host_minus_dev = []
        with open(path, errors="ignore") as f:
            for line in f:
                m = PAT.search(line)
                if not m:
                    continue
                h = float(m.group(2)); d = float(m.group(3))
                nsr = int(m.group(4)); nct = int(m.group(6))
                host.append(h); dev.append(d)
                nctxtok.append(nct); nsched.append(nsr)
                if nct > 0:  # steps doing prefill work
                    ctx_steps += 1
                    host_minus_dev.append(h - d)
        n = len(host)
        print(f"\n===== {path}  ({n} iters parsed, {ctx_steps} with ctx prefill) =====")
        if n == 0:
            continue
        print(f"  host_step_time   p50={pct(host,50):.1f} p90={pct(host,90):.1f} "
              f"p95={pct(host,95):.1f} p99={pct(host,99):.1f} max={max(host):.1f} ms")
        print(f"  device_step_time p50={pct(dev,50):.1f} p90={pct(dev,90):.1f} "
              f"p95={pct(dev,95):.1f} p99={pct(dev,99):.1f} max={max(dev):.1f} ms")
        print(f"  host SUM={sum(host)/1000:.1f}s  device SUM={sum(dev)/1000:.1f}s  "
              f"-> host/(host+device) = {100*sum(host)/(sum(host)+sum(dev)):.0f}%")
        # How much of host time is NOT explained by device compute?
        big = [x for x in host if x > 200]
        print(f"  host_step_time > 200ms: {len(big)} steps "
              f"({100*len(big)/n:.1f}%), summing {sum(big)/1000:.1f}s")
        # Step-time vs ctx-token bins
        print(f"  {'ctx_tokens':>12} {'nsteps':>7} {'host p50':>9} {'host p95':>9} {'dev p50':>8} {'dev p95':>8}")
        edges = [(1, 256), (256, 1024), (1024, 4096), (4096, 12000),
                 (12000, 20001)]
        for a, b in edges:
            hs = [host[i] for i in range(n) if a <= nctxtok[i] < b]
            ds = [dev[i] for i in range(n) if a <= nctxtok[i] < b]
            if hs:
                print(f"  {f'{a}-{b}':>12} {len(hs):>7} {pct(hs,50):>8.1f} "
                      f"{pct(hs,95):>8.1f} {pct(ds,50):>7.1f} {pct(ds,95):>7.1f}")


if __name__ == "__main__":
    main()
