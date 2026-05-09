#!/usr/bin/env python3
"""Aggregate multi-run trtllm-bench results for the NVBug 5615248 workload.

Loads ``request_<backend>{,2..N}.json`` files (per-request timestamps in ns)
from one or more directories and reports per-run statistics (n = number of
runs) and pooled per-request samples (n = num_runs * num_requests).

The PRIMARY headline statistic is the **median** (robust to single-sample
outliers that are common in latency micro-benchmarks). Mean / stdev /
Welch's t-test are reported as supplementary "informational" rows.

When two directories are passed (``--baseline`` and ``--experiment``) the
comparison reports:
  * Δmedian with a bootstrap 95% CI on the difference of medians
  * Mann-Whitney U two-sided p-value (rank-based; outlier-robust)
  * Hodges-Lehmann shift estimator (median of pairwise differences)
  * Δmean + Welch's t-test (informational; sensitive to outliers)

See ``nvbugs_5615248/trtllm_bench/PREPARE_BEAM_SEARCH_FIX.md`` for the
underlying multi-run protocol.

Usage::

    # Single directory
    python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \\
        --experiment nvbugs_5615248/trtllm_bench/optimized_v3

    # Before/after comparison
    python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \\
        --baseline   nvbugs_5615248/trtllm_bench \\
        --experiment nvbugs_5615248/trtllm_bench/optimized_v3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from dataclasses import dataclass

NS_PER_MS = 1e6

# Match request_<backend>.json or request_<backend>{2..9}.json. Avoids picking
# up unrelated files such as request_pytorch_early_emit_first.json that also
# live in this directory. Backend tag is letters-only so the trailing
# digit cleanly splits into the run-index group.
REQUEST_RE = re.compile(r"^request_(?P<backend>[a-zA-Z]+)(?P<idx>[0-9]?)\.json$")


@dataclass
class RunSample:
    backend: str
    idx: int  # 1 for the unsuffixed run, 2..N for suffixed runs
    path: str
    ttft_ms: list[float]
    e2e_ms: list[float]
    itl_ms: list[float]


@dataclass
class AggView:
    """Statistics for one (directory, backend) bucket.

    ``per_run_means`` has length ``num_runs`` (one mean per file).
    ``pooled`` has length ``num_runs * num_requests`` (every per-request sample).
    """

    label: str
    backend: str
    num_runs: int
    num_requests_per_run: int

    ttft_per_run_means_ms: list[float]
    e2e_per_run_means_ms: list[float]
    itl_per_run_means_ms: list[float]

    ttft_pooled_ms: list[float]
    e2e_pooled_ms: list[float]
    itl_pooled_ms: list[float]


def discover_runs(directory: str, backend: str) -> list[RunSample]:
    samples: list[RunSample] = []
    for fname in sorted(os.listdir(directory)):
        m = REQUEST_RE.match(fname)
        if not m or m.group("backend") != backend:
            continue
        idx_str = m.group("idx")
        # Treat the unsuffixed file (idx="") as run 1 to align with how
        # the multi-run launcher names artifacts.
        idx = 1 if idx_str == "" else int(idx_str)
        path = os.path.join(directory, fname)
        with open(path) as fh:
            payload = json.load(fh)
        if not isinstance(payload, list) or not payload:
            print(f"  skipping {path}: empty or non-list payload", file=sys.stderr)
            continue
        ttft = [r["time_to_first_token"] / NS_PER_MS for r in payload]
        e2e = [r["end_to_end_latency"] / NS_PER_MS for r in payload]
        itl = [r["intertoken_latency"] / NS_PER_MS for r in payload]
        samples.append(
            RunSample(
                backend=backend,
                idx=idx,
                path=path,
                ttft_ms=ttft,
                e2e_ms=e2e,
                itl_ms=itl,
            )
        )
    samples.sort(key=lambda s: s.idx)
    return samples


def aggregate(label: str, directory: str, backend: str) -> AggView | None:
    runs = discover_runs(directory, backend)
    if not runs:
        return None

    per_request_count = len(runs[0].ttft_ms)
    for r in runs:
        assert len(r.ttft_ms) == per_request_count, (
            f"run-to-run request count mismatch in {r.path}: "
            f"{len(r.ttft_ms)} vs {per_request_count}"
        )

    return AggView(
        label=label,
        backend=backend,
        num_runs=len(runs),
        num_requests_per_run=per_request_count,
        ttft_per_run_means_ms=[statistics.fmean(r.ttft_ms) for r in runs],
        e2e_per_run_means_ms=[statistics.fmean(r.e2e_ms) for r in runs],
        itl_per_run_means_ms=[statistics.fmean(r.itl_ms) for r in runs],
        ttft_pooled_ms=[v for r in runs for v in r.ttft_ms],
        e2e_pooled_ms=[v for r in runs for v in r.e2e_ms],
        itl_pooled_ms=[v for r in runs for v in r.itl_ms],
    )


def welch_t_p(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Welch's t-test. Pure-stdlib (Survival-function via lgamma).

    Returns (t, p_two_sided).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan")
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    se_sq = var_a / n_a + var_b / n_b
    if se_sq <= 0:
        return float("nan"), float("nan")
    t = (mean_a - mean_b) / math.sqrt(se_sq)

    # Welch–Satterthwaite degrees of freedom
    df_num = se_sq ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else float("nan")

    # Survival function of Student's t via the regularized incomplete beta
    # I_x(a, b) with x = df / (df + t^2), a = df/2, b = 1/2 (two-sided p =
    # I_x(df/2, 1/2)). Use the standard continued-fraction expansion.
    if not math.isfinite(df) or df <= 0:
        return t, float("nan")
    x = df / (df + t * t)
    p = _regularized_incomplete_beta(x, df / 2.0, 0.5)
    return t, p


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Symmetry: I_x(a, b) = 1 - I_{1-x}(b, a). Use it for fast convergence
    # of the continued fraction.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x)) / a
    return front * _betacf(x, a, b)


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-16) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile (matches numpy with method='higher')."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    k = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
    return sorted_vals[k]


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test with normal approximation + tie correction.

    Robust to outliers because it ranks the combined sample rather than using
    raw values. Returns ``(U_a, p_two_sided)``. ``U_a < n_a*n_b/2`` means
    samples in ``a`` tend to be smaller than samples in ``b``.
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan")
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_a = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == 0)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    mean_u = n_a * n_b / 2.0
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_term += (t ** 3 - t)
        i = j + 1
    n = n_a + n_b
    var_u = (n_a * n_b / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0:
        return u_a, float("nan")
    z = (u - mean_u) / math.sqrt(var_u)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return u_a, p


def hodges_lehmann_delta(a: list[float], b: list[float], cap: int = 600) -> float:
    """Median of all pairwise differences (b_j - a_i) — the most robust shift
    estimator. Subsamples each list to ``cap`` if needed (cap=600 → 360k pairs,
    fast and stable)."""
    if not a or not b:
        return float("nan")
    rng = random.Random(0)
    aa = a if len(a) <= cap else rng.sample(a, cap)
    bb = b if len(b) <= cap else rng.sample(b, cap)
    diffs = [y - x for x in aa for y in bb]
    return statistics.median(diffs)


def bootstrap_delta_ci(
    a: list[float],
    b: list[float],
    *,
    estimator: str = "median",
    iters: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap (point, lo, hi) for the chosen estimator on (b - a)."""
    if estimator == "median":
        agg = statistics.median
    elif estimator == "mean":
        agg = statistics.fmean
    else:
        raise ValueError(f"unknown estimator: {estimator!r}")
    if not a or not b:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n_a, n_b = len(a), len(b)
    deltas: list[float] = []
    for _ in range(iters):
        sa = [a[rng.randrange(n_a)] for _ in range(n_a)]
        sb = [b[rng.randrange(n_b)] for _ in range(n_b)]
        deltas.append(agg(sb) - agg(sa))
    deltas.sort()
    point = agg(b) - agg(a)
    lo = deltas[int(alpha / 2 * iters)]
    hi = deltas[int((1 - alpha / 2) * iters)]
    return point, lo, hi


def _row_per_run(view: AggView, metric_attr: str, label_metric: str) -> str:
    means = getattr(view, metric_attr)
    median = statistics.median(means)
    return (
        f"| {view.label} ({view.backend}, n={view.num_runs}) | {label_metric} "
        f"| {median:8.3f} "
        f"| {statistics.fmean(means):8.3f} "
        f"| {(statistics.stdev(means) if len(means) > 1 else float('nan')):8.3f} "
        f"| {min(means):8.3f} | {max(means):8.3f} |"
    )


def _row_pooled(view: AggView, metric_attr: str, label_metric: str) -> str:
    pooled = getattr(view, metric_attr)
    s = sorted(pooled)
    median = statistics.median(pooled)
    p10 = _percentile(s, 10)
    p90 = _percentile(s, 90)
    return (
        f"| {view.label} ({view.backend}, n={len(pooled)}) | {label_metric} "
        f"| {median:8.3f} "
        f"| {statistics.fmean(pooled):8.3f} "
        f"| {p10:8.3f} | {p90:8.3f} "
        f"| {statistics.stdev(pooled):8.3f} "
        f"| {min(pooled):8.3f} | {max(pooled):8.3f} |"
    )


def print_view(view: AggView) -> None:
    print(f"\n### {view.label} — {view.backend} backend")
    print(f"({view.num_runs} runs × {view.num_requests_per_run} requests)\n")
    print("Per-run statistics (one mean per file, then aggregated across runs):")
    print()
    print("| label | metric | median (ms) | mean (ms) | stdev (ms) | min (ms) | max (ms) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print(_row_per_run(view, "ttft_per_run_means_ms", "TTFT"))
    print(_row_per_run(view, "e2e_per_run_means_ms", "E2E"))
    print(_row_per_run(view, "itl_per_run_means_ms", "ITL"))

    print()
    print("Pooled per-request:")
    print()
    print("| label | metric | median (ms) | mean (ms) | p10 (ms) | p90 (ms) | stdev (ms) | min (ms) | max (ms) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    print(_row_pooled(view, "ttft_pooled_ms", "TTFT"))
    print(_row_pooled(view, "e2e_pooled_ms", "E2E"))
    print(_row_pooled(view, "itl_pooled_ms", "ITL"))


def print_comparison(baseline: AggView, experiment: AggView) -> None:
    assert baseline.backend == experiment.backend, (
        f"backend mismatch in comparison: baseline={baseline.backend}, "
        f"experiment={experiment.backend}"
    )

    print(f"\n## {experiment.label} vs {baseline.label}  (backend={baseline.backend})")

    # Sign convention for the printed columns:
    #   Δ (ms) = experiment − baseline   → negative = improvement.
    #   t      = raw Welch's t of (baseline, experiment) = (μ_base − μ_exp) / SE
    #                                    → positive = improvement.
    # This matches the convention used in PREPARE_BEAM_SEARCH_FIX.md.

    pooled_attrs = [("TTFT", "ttft_pooled_ms"),
                    ("E2E", "e2e_pooled_ms"),
                    ("ITL", "itl_pooled_ms")]
    per_run_attrs = [("TTFT", "ttft_per_run_means_ms"),
                     ("E2E", "e2e_per_run_means_ms"),
                     ("ITL", "itl_per_run_means_ms")]

    # ---- Headline: robust pooled comparison (median + bootstrap CI + MW + HL).
    print(f"\n### Pooled per-request — robust (n={len(baseline.ttft_pooled_ms)} vs "
          f"n={len(experiment.ttft_pooled_ms)})\n")
    print("| metric | base median | exp median | Δmedian (ms) | Δ (%) "
          "| 95% CI (bootstrap) | Mann-Whitney p | Hodges-Lehmann (ms) |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for label, attr in pooled_attrs:
        a = getattr(baseline, attr)
        b = getattr(experiment, attr)
        med_a, med_b = statistics.median(a), statistics.median(b)
        d, lo, hi = bootstrap_delta_ci(a, b, estimator="median", iters=5000)
        pct = 100.0 * d / med_a if med_a else float("nan")
        _, p_mw = mann_whitney_u(a, b)
        hl = hodges_lehmann_delta(a, b)
        print(f"| {label} | {med_a:8.3f} | {med_b:8.3f} | {d:+8.4f} "
              f"| {pct:+7.3f}% | [{lo:+.4f}, {hi:+.4f}] | {p_mw:8.3g} | {hl:+8.4f} |")

    # ---- Per-run medians (less powerful with small n, but a useful sanity check).
    print(f"\n### Per-run medians  (n={baseline.num_runs} vs n={experiment.num_runs})\n")
    print("| metric | base median | exp median | Δ (ms) | Δ (%) |")
    print("|---|---:|---:|---:|---:|")
    for label, attr in per_run_attrs:
        a = getattr(baseline, attr)
        b = getattr(experiment, attr)
        med_a, med_b = statistics.median(a), statistics.median(b)
        d = med_b - med_a
        pct = 100.0 * d / med_a if med_a else float("nan")
        print(f"| {label} | {med_a:8.3f} | {med_b:8.3f} | {d:+8.4f} | {pct:+7.3f}% |")

    # ---- Informational: mean comparison (sensitive to outliers, kept for
    # backward-compatible reading of older logs).
    print("\n### Pooled per-request — mean / Welch's t  (informational; "
          "sensitive to outliers)\n")
    print("| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, attr in pooled_attrs:
        a = getattr(baseline, attr)
        b = getattr(experiment, attr)
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        delta = mb - ma
        pct = 100.0 * delta / ma if ma else float("nan")
        t, p = welch_t_p(a, b)
        print(f"| {label} | {ma:8.3f} | {mb:8.3f} | {delta:+8.4f} | {pct:+7.3f}% "
              f"| {t:+6.2f} | {p:8.3g} |")

    print("\n### Per-run means — mean / Welch's t  (informational)\n")
    print("| metric | base mean | exp mean | Δmean (ms) | Δ (%) | Welch t | Welch p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, attr in per_run_attrs:
        a = getattr(baseline, attr)
        b = getattr(experiment, attr)
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        delta = mb - ma
        pct = 100.0 * delta / ma if ma else float("nan")
        t, p = welch_t_p(a, b)
        print(f"| {label} | {ma:8.3f} | {mb:8.3f} | {delta:+8.4f} | {pct:+7.3f}% "
              f"| {t:+6.2f} | {p:8.3g} |")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", help="Directory with baseline request_*.json files")
    p.add_argument("--experiment", required=True,
                   help="Directory with experiment request_*.json files")
    p.add_argument("--backend", default="pytorch",
                   help="Backend tag in filenames (default: pytorch)")
    p.add_argument("--baseline-label", default="baseline")
    p.add_argument("--experiment-label", default="experiment")
    args = p.parse_args()

    exp_view = aggregate(args.experiment_label, args.experiment, args.backend)
    if exp_view is None:
        print(f"ERROR: no request_{args.backend}*.json files in {args.experiment}",
              file=sys.stderr)
        return 1
    print_view(exp_view)

    if args.baseline:
        base_view = aggregate(args.baseline_label, args.baseline, args.backend)
        if base_view is None:
            print(f"ERROR: no request_{args.backend}*.json files in {args.baseline}",
                  file=sys.stderr)
            return 1
        print_view(base_view)
        print_comparison(base_view, exp_view)

    return 0


if __name__ == "__main__":
    sys.exit(main())
