#!/usr/bin/env python3
"""
Parse gen-only benchmark job directories and extract completion status,
saturation metrics, and configuration into a CSV file.

Usage:
    source /lustre/fsw/coreai_comparch_trtllm/bbuddharaju/venvs/pareto/bin/activate
    python parse_gen_only.py --dir_list <job_dirs> --output_dir <output_dir>

Or run directly with the venv Python:
    /lustre/fsw/coreai_comparch_trtllm/bbuddharaju/venvs/pareto/bin/python parse_gen_only.py ...
"""

import argparse
import csv
import glob
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

# Prefill requests/sec sustained by one TP=1 ctx server (1 GPU), measured
# separately. Used to size the ctx pool when rate matching.
DEFAULT_CTX_REQ_RATE_PER_GPU = 4.896


def is_job_directory(path: str) -> bool:
    """Check if a path is a job directory by looking for gen_config.yaml."""
    return os.path.isfile(os.path.join(path, "gen_config.yaml"))


def parse_isl_osl(job_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (isl, osl) from the enclosing '<isl>-<osl>' log directory."""
    match = re.fullmatch(r"(\d+)-(\d+)", os.path.basename(os.path.dirname(job_dir)))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def find_job_directories(dir_list: List[str]) -> List[str]:
    """
    Given a list of paths, find all job directories.
    Each path can be either a job directory or a parent containing job directories.
    """
    job_dirs = []
    for path in dir_list:
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            print(f"Warning: {path} is not a directory, skipping")
            continue

        if is_job_directory(path):
            job_dirs.append(path)
        else:
            # Scan subdirectories for job directories.
            for entry in os.listdir(path):
                subdir = os.path.join(path, entry)
                if os.path.isdir(subdir) and is_job_directory(subdir):
                    job_dirs.append(subdir)

    return job_dirs


def parse_gen_config(job_dir: str) -> Dict:
    """Parse gen_config.yaml and extract configuration values."""
    config_path = os.path.join(job_dir, "gen_config.yaml")
    config = {}

    try:
        with open(config_path, "r") as f:
            gen_config = yaml.safe_load(f)

        config["gen_tp"] = gen_config.get("tensor_parallel_size", None)
        config["gen_cp"] = gen_config.get("context_parallel_size", None)
        config["gen_pp"] = gen_config.get("pipeline_parallel_size", None)
        config["gen_ep"] = gen_config.get("moe_expert_parallel_size", None)
        config["is_attn_dp"] = gen_config.get("enable_attention_dp", False)

        # Get moe_config.backend.
        moe_config = gen_config.get("moe_config", {})
        config["moe_backend"] = moe_config.get("backend", None) if moe_config else None

        # Get cuda_graph_config.max_batch_size.
        cuda_graph_config = gen_config.get("cuda_graph_config", {})
        config["max_batch_size"] = (
            cuda_graph_config.get("max_batch_size", None)
            if cuda_graph_config
            else None
        )

    except Exception as e:
        print(f"Warning: Failed to parse gen_config.yaml in {job_dir}: {e}")

    return config


def parse_global_batch_size(job_dir: str) -> Optional[int]:
    """Extract global batch size from directory name (e.g., batch32 -> 32)."""
    dir_name = os.path.basename(job_dir)
    match = re.search(r"batch(\d+)", dir_name)
    if match:
        return int(match.group(1))
    return None


def get_slurm_id(job_dir: str) -> Optional[str]:
    """Get Slurm ID from 8_done_<slurm_id>.txt filename."""
    pattern = os.path.join(job_dir, "8_done_*.txt")
    done_files = glob.glob(pattern)
    if done_files:
        # Extract slurm_id from filename like 8_done_818972.txt.
        filename = os.path.basename(done_files[0])
        match = re.match(r"8_done_(\d+)\.txt", filename)
        if match:
            return match.group(1)
    return None


def parse_bench_log(job_dir: str) -> Tuple[bool, Optional[float]]:
    """
    Parse 6_bench.log for median TPOT.
    Returns (is_complete, median_tpot_ms).
    """
    bench_log_path = os.path.join(job_dir, "6_bench.log")

    if not os.path.isfile(bench_log_path):
        return False, None

    try:
        with open(bench_log_path, "r", errors="ignore") as f:
            content = f.read()

        # Look for "Median TPOT (ms):" line.
        match = re.search(r"Median TPOT \(ms\):\s+([\d.]+)", content)
        if match:
            median_tpot = float(match.group(1))
            return True, median_tpot
    except Exception as e:
        print(f"Warning: Failed to parse 6_bench.log in {job_dir}: {e}")

    return False, None


def parse_iteration_logs(
    job_dir: str, max_batch_size: Optional[int]
) -> Dict:
    """
    Parse 3_output_GEN_0.log for saturation analysis and iteration TPOT.

    Extracts:
    - Saturation metrics (how many iterations ran at full batch)
    - Median TPOT from prev_device_step_time of saturated iterations

    This provides a more accurate TPOT than the benchmark client reports,
    especially for CP>1 configurations where there's significant overhead
    between device step completion and client-observed latency.

    Returns dict with saturation metrics and iter_median_tpot_ms.
    """
    log_path = os.path.join(job_dir, "3_output_GEN_0.log")
    result = {
        "saturation_pct": None,
        "saturated_iters": 0,
        "total_iters": 0,
        "first_saturated_iter": None,
        "last_saturated_iter": None,
        "iter_median_tpot_ms": None,
    }

    if not os.path.isfile(log_path):
        return result

    if max_batch_size is None:
        return result

    try:
        # Fields are matched independently because their order and separator
        # vary across TRT-LLM versions (num_scheduled_requests appears both as
        # "= N" before prev_device_step_time and as ": N" after it).
        iter_pattern = re.compile(r"iter = (\d+)")
        step_time_pattern = re.compile(r"prev_device_step_time\s*=\s*([\d.]+)ms")
        scheduled_pattern = re.compile(r"num_scheduled_requests\s*[:=]\s*(\d+)")

        iterations = []
        saturated_device_step_times = []

        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                # Skip lines with N/A for prev_device_step_time
                if "prev_device_step_time = N/A" in line:
                    continue

                iter_match = iter_pattern.search(line)
                step_time_match = step_time_pattern.search(line)
                scheduled_match = scheduled_pattern.search(line)
                if iter_match and step_time_match and scheduled_match:
                    iter_num = int(iter_match.group(1))
                    prev_device_step_time = float(step_time_match.group(1))
                    num_scheduled = int(scheduled_match.group(1))
                    iterations.append((iter_num, num_scheduled, prev_device_step_time))

                    # Collect device step times for saturated iterations
                    if num_scheduled == max_batch_size:
                        saturated_device_step_times.append(prev_device_step_time)

        if not iterations:
            return result

        result["total_iters"] = len(iterations)

        # Find saturated iterations (where num_scheduled_requests == max_batch_size).
        saturated_iters = [
            (iter_num, num_scheduled)
            for iter_num, num_scheduled, _ in iterations
            if num_scheduled == max_batch_size
        ]

        result["saturated_iters"] = len(saturated_iters)

        if saturated_iters:
            result["first_saturated_iter"] = saturated_iters[0][0]
            result["last_saturated_iter"] = saturated_iters[-1][0]

        if result["total_iters"] > 0:
            result["saturation_pct"] = round(
                100.0 * result["saturated_iters"] / result["total_iters"], 2
            )

        # Calculate median TPOT from saturated iterations
        # TPOT = prev_device_step_time (time per iteration = time per token per request)
        if saturated_device_step_times:
            result["iter_median_tpot_ms"] = round(
                statistics.median(saturated_device_step_times), 4
            )

    except Exception as e:
        print(f"Warning: Failed to parse 3_output_GEN_0.log in {job_dir}: {e}")

    return result


def analyze_job(
    job_dir: str,
    isl: Optional[int] = None,
    osl: Optional[int] = None,
    ctx_req_rate: float = DEFAULT_CTX_REQ_RATE_PER_GPU,
) -> Dict:
    """Analyze a single job directory and return results."""
    result = {
        "job_dir": job_dir,
        "slurm_id": None,
        "num_gpus": None,
        "global_batch_size": None,
        "is_attn_dp": None,
        "gen_tp": None,
        "gen_cp": None,
        "gen_pp": None,
        "gen_ep": None,
        "moe_backend": None,
        "is_complete": False,
        "bench_median_tpot_ms": None,  # TPOT from benchmark client (6_bench.log)
        "iter_median_tpot_ms": None,   # TPOT from iteration logs (prev_device_step_time)
        "tpot_discrepancy_pct": None,  # % difference: (bench - iter) / iter * 100
        "done_file_present": False,
        "saturation_pct": None,
        "saturated_iters": 0,
        "total_iters": 0,
        "first_saturated_iter": None,
        "last_saturated_iter": None,
        "gen_output_tput_per_user": None,
        "gen_output_tput_per_gpu": None,
        # Rate-matched totals: decode GPUs plus the ctx GPUs needed to feed them
        "isl": None,
        "osl": None,
        "gen_req_rate_per_s": None,
        "ctx_gpus_needed": None,
        "total_gpus": None,
        "gen_total_tput_per_gpu": None,
    }

    # Get Slurm ID and check done file.
    slurm_id = get_slurm_id(job_dir)
    result["slurm_id"] = slurm_id
    result["done_file_present"] = slurm_id is not None

    # Parse global batch size from directory name.
    result["global_batch_size"] = parse_global_batch_size(job_dir)

    # Parse gen_config.yaml.
    config = parse_gen_config(job_dir)
    result["is_attn_dp"] = config.get("is_attn_dp")
    result["gen_tp"] = config.get("gen_tp")
    result["gen_cp"] = config.get("gen_cp")
    result["gen_pp"] = config.get("gen_pp")
    result["gen_ep"] = config.get("gen_ep")
    result["moe_backend"] = config.get("moe_backend")
    max_batch_size = config.get("max_batch_size")

    # Calculate num_gpus = tp * cp * pp.
    tp = result["gen_tp"]
    cp = result["gen_cp"]
    pp = result["gen_pp"]
    if all(v is not None for v in [tp, cp, pp]):
        result["num_gpus"] = tp * cp * pp

    # Verify global_batch_size from directory name matches expected value from config.
    # global_batch_size = cuda_graph_config.max_batch_size * pp * dp_size
    # where dp_size = tp if attention_dp is enabled, else 1
    global_batch_size = result["global_batch_size"]
    is_attn_dp = result["is_attn_dp"]
    assert all(v is not None for v in [global_batch_size, max_batch_size, pp, tp, is_attn_dp]), (
        f"Missing required config values in {job_dir}: "
        f"global_batch_size={global_batch_size}, max_batch_size={max_batch_size}, "
        f"pp={pp}, tp={tp}, is_attn_dp={is_attn_dp}"
    )
    dp_size = tp if is_attn_dp else 1
    expected_global_batch_size = max_batch_size * pp * dp_size
    assert global_batch_size == expected_global_batch_size, (
        f"global_batch_size mismatch in {job_dir}: "
        f"directory name says {global_batch_size}, but expected "
        f"{expected_global_batch_size} (cuda_graph_config.max_batch_size={max_batch_size} * pp={pp} * dp_size={dp_size})"
    )

    # Parse bench log for completion status only.
    # NOTE: bench_median_tpot_ms is kept for reference but NOT used for throughput.
    # The bench client measures higher latency than actual device step time,
    # especially for CP>1 configurations (~35-45% higher vs ~7.5% for CP=1).
    is_complete, bench_median_tpot = parse_bench_log(job_dir)
    result["is_complete"] = is_complete
    result["bench_median_tpot_ms"] = bench_median_tpot

    # Parse iteration logs for saturation analysis AND iteration-based TPOT.
    # iter_median_tpot_ms is computed from prev_device_step_time of saturated
    # iterations (where num_scheduled_requests == max_batch_size).
    # This is more accurate than bench TPOT as it measures actual GPU execution time.
    saturation_info = parse_iteration_logs(job_dir, max_batch_size)
    result["saturation_pct"] = saturation_info["saturation_pct"]
    result["saturated_iters"] = saturation_info["saturated_iters"]
    result["total_iters"] = saturation_info["total_iters"]
    result["first_saturated_iter"] = saturation_info["first_saturated_iter"]
    result["last_saturated_iter"] = saturation_info["last_saturated_iter"]
    result["iter_median_tpot_ms"] = saturation_info["iter_median_tpot_ms"]

    # Calculate TPOT discrepancy between bench and iteration logs.
    iter_tpot = result["iter_median_tpot_ms"]
    if bench_median_tpot is not None and iter_tpot is not None and iter_tpot > 0:
        result["tpot_discrepancy_pct"] = round(
            100.0 * (bench_median_tpot - iter_tpot) / iter_tpot, 2
        )

    # Calculate performance stats using ITERATION TPOT (not bench TPOT).
    # This provides more accurate throughput metrics, especially for CP>1.
    result["gen_output_tput_per_user"] = None
    result["gen_output_tput_per_gpu"] = None

    if iter_tpot is not None and iter_tpot > 0:
        # gen_output_tput_per_user = 1000 / iter_tpot (tokens/sec per user).
        result["gen_output_tput_per_user"] = round(1000.0 / iter_tpot, 2)

        # gen_output_tput_per_gpu = 1000 * global_bs / (iter_tpot * num_gpus),
        # where num_gpus = tp_size * cp_size * pp_size.
        global_bs = result["global_batch_size"]
        tp = result["gen_tp"]
        cp = result["gen_cp"]
        pp = result["gen_pp"]

        if all(v is not None for v in [global_bs, tp, cp, pp]):
            num_gpus = tp * cp * pp
            if num_gpus > 0:
                result["gen_output_tput_per_gpu"] = round(
                    1000.0 * global_bs / (iter_tpot * num_gpus), 2
                )

    # Rate-matched total throughput: charge each decode instance for the
    # fractional number of ctx GPUs required to keep it fed, then count both
    # input and output tokens against the combined GPU count.
    if isl is None or osl is None:
        path_isl, path_osl = parse_isl_osl(job_dir)
        isl = isl if isl is not None else path_isl
        osl = osl if osl is not None else path_osl
    result["isl"] = isl
    result["osl"] = osl

    num_gpus = result["num_gpus"]
    global_bs = result["global_batch_size"]
    if (
        iter_tpot is not None
        and iter_tpot > 0
        and isl
        and osl
        and num_gpus
        and global_bs
        and ctx_req_rate > 0
    ):
        # Requests retired per second = batch / (osl decode steps x step time).
        gen_req_rate = global_bs / (osl * iter_tpot / 1000.0)
        ctx_gpus = gen_req_rate / ctx_req_rate
        total_gpus = num_gpus + ctx_gpus
        result["gen_req_rate_per_s"] = round(gen_req_rate, 4)
        result["ctx_gpus_needed"] = round(ctx_gpus, 3)
        result["total_gpus"] = round(total_gpus, 3)
        result["gen_total_tput_per_gpu"] = round(
            gen_req_rate * (isl + osl) / total_gpus, 2
        )

    return result


def compute_pareto_frontier_with_results(
    results: List[Dict], y_key: str = "gen_output_tput_per_gpu"
) -> List[Dict]:
    """
    Compute Pareto frontier from result dicts, maximizing both X and Y.
    Returns list of result dicts that are on the Pareto frontier.
    """
    if not results:
        return []

    # Sort by X (descending) to find Pareto frontier.
    sorted_results = sorted(
        results, key=lambda r: -r["gen_output_tput_per_user"]
    )

    pareto = []
    max_y = float("-inf")

    for r in sorted_results:
        if r[y_key] > max_y:
            pareto.append(r)
            max_y = r[y_key]

    # Sort by X ascending for plotting.
    pareto.sort(key=lambda r: r["gen_output_tput_per_user"])
    return pareto


def dedupe_pareto_by_batch_size(
    pareto_results: List[Dict], y_key: str = "gen_output_tput_per_gpu"
) -> List[Dict]:
    """
    For each global_batch_size, keep only the point with highest tput_per_gpu.
    """
    if not pareto_results:
        return []

    # Group by global_batch_size.
    by_bs: Dict[int, List[Dict]] = {}
    for r in pareto_results:
        bs = r["global_batch_size"]
        if bs not in by_bs:
            by_bs[bs] = []
        by_bs[bs].append(r)

    # Keep only the one with highest tput_per_gpu for each batch size.
    deduped = []
    for bs, group in by_bs.items():
        best = max(group, key=lambda r: r[y_key])
        deduped.append(best)

    # Sort by X ascending.
    deduped.sort(key=lambda r: r["gen_output_tput_per_user"])
    return deduped


def plot_pareto(
    results: List[Dict],
    output_dir: str,
    y_key: str = "gen_output_tput_per_gpu",
    y_label: str = "Output Throughput per GPU (tok/s/gpu)",
    title: str = "MiniMax M3 NVFP4 8k/1k TRTLLM Pareto, GB300 NVL72, Gen-Only SOL, #Gen GPUs Per Instance<=32",
    file_prefix: str = "pareto_plot",
    csv_name: str = "pareto_frontier.csv",
) -> Optional[str]:
    """
    Plot Pareto frontier for CP jobs (CP>1) and TP-only jobs (CP=1).
    Creates two versions:
    1) Full plot with all points (dimmer) + Pareto frontier
    2) Denoised plot with only Pareto frontier, one point per global BS
    X-axis: gen_output_tput_per_user
    Y-axis: y_key
    """
    # Filter completed jobs with valid throughput data.
    valid_results = [
        r for r in results
        if r["is_complete"]
        and r["gen_output_tput_per_user"] is not None
        and r.get(y_key) is not None
        and r["gen_cp"] is not None
    ]

    if not valid_results:
        print(f"No valid results for Pareto plot ({y_key})")
        return None

    # Separate into CP jobs (CP > 1) and TP-only jobs (CP = 1).
    cp_jobs = [r for r in valid_results if r["gen_cp"] > 1]
    tp_only_jobs = [r for r in valid_results if r["gen_cp"] == 1]

    # Compute Pareto frontiers (with full result dicts).
    cp_pareto = compute_pareto_frontier_with_results(cp_jobs, y_key)
    tp_pareto = compute_pareto_frontier_with_results(tp_only_jobs, y_key)

    # Deduped Pareto (one per global BS with highest tput_per_gpu).
    cp_pareto_deduped = dedupe_pareto_by_batch_size(cp_pareto, y_key)
    tp_pareto_deduped = dedupe_pareto_by_batch_size(tp_pareto, y_key)

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # Plot 1: Full plot with all points (dimmer) + Pareto frontier
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all points (dimmer).
    if cp_jobs:
        cp_x = [r["gen_output_tput_per_user"] for r in cp_jobs]
        cp_y = [r[y_key] for r in cp_jobs]
        ax.scatter(cp_x, cp_y, alpha=0.25, color="blue", marker="o", s=40,
                   label="KVP>1")

    if tp_only_jobs:
        tp_x = [r["gen_output_tput_per_user"] for r in tp_only_jobs]
        tp_y = [r[y_key] for r in tp_only_jobs]
        ax.scatter(tp_x, tp_y, alpha=0.25, color="orange", marker="s", s=40,
                   label="KVP=1")

    # Plot Pareto frontiers (brighter).
    if cp_pareto:
        pareto_x = [r["gen_output_tput_per_user"] for r in cp_pareto]
        pareto_y = [r[y_key] for r in cp_pareto]
        ax.plot(pareto_x, pareto_y, "b-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="blue", s=80, zorder=5,
                   edgecolors="black", linewidths=1)
        # Add batch size labels.
        for r in cp_pareto:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r[y_key]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    if tp_pareto:
        pareto_x = [r["gen_output_tput_per_user"] for r in tp_pareto]
        pareto_y = [r[y_key] for r in tp_pareto]
        ax.plot(pareto_x, pareto_y, color="orange", linestyle="-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="orange", s=80, zorder=5,
                   edgecolors="black", linewidths=1)
        # Add batch size labels.
        for r in tp_pareto:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r[y_key]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    ax.set_xlabel("Output Throughput per User (tok/s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_path_full = os.path.join(output_dir, f"{file_prefix}_full.png")
    plt.savefig(plot_path_full, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 2: Denoised - only Pareto frontier, one point per global BS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    if cp_pareto_deduped:
        pareto_x = [r["gen_output_tput_per_user"] for r in cp_pareto_deduped]
        pareto_y = [r[y_key] for r in cp_pareto_deduped]
        ax.plot(pareto_x, pareto_y, "b-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="blue", s=100, zorder=5,
                   edgecolors="black", linewidths=1.5,
                   label="KVP>1")
        # Add batch size labels.
        for r in cp_pareto_deduped:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r[y_key]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    if tp_pareto_deduped:
        pareto_x = [r["gen_output_tput_per_user"] for r in tp_pareto_deduped]
        pareto_y = [r[y_key] for r in tp_pareto_deduped]
        ax.plot(pareto_x, pareto_y, color="orange", linestyle="-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_x, pareto_y, color="orange", s=100, zorder=5,
                   edgecolors="black", linewidths=1.5,
                   label="KVP=1")
        # Add batch size labels.
        for r in tp_pareto_deduped:
            ax.annotate(
                f"bs={r['global_batch_size']}",
                (r["gen_output_tput_per_user"], r[y_key]),
                textcoords="offset points", xytext=(5, 5), fontsize=4
            )

    ax.set_xlabel("Output Throughput per User (tok/s)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"{title} (Denoised)", fontsize=10)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plot_path_denoised = os.path.join(output_dir, f"{file_prefix}_denoised.png")
    plt.savefig(plot_path_denoised, dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Save Pareto frontier to CSV
    # =========================================================================
    pareto_csv_path = os.path.join(output_dir, csv_name)

    # Build set of deduped job_dirs for quick lookup.
    cp_deduped_dirs = {r["job_dir"] for r in cp_pareto_deduped}
    tp_deduped_dirs = {r["job_dir"] for r in tp_pareto_deduped}

    pareto_rows = []

    # Add KVP>1 Pareto points.
    for r in cp_pareto:
        row = {
            "category": "KVP>1",
            "deduped": r["job_dir"] not in cp_deduped_dirs,
            "slurm_id": r["slurm_id"],
            "global_batch_size": r["global_batch_size"],
            "gen_tp": r["gen_tp"],
            "gen_cp": r["gen_cp"],
            "gen_pp": r["gen_pp"],
            "gen_ep": r["gen_ep"],
            "is_attn_dp": r["is_attn_dp"],
            "moe_backend": r["moe_backend"],
            "iter_median_tpot_ms": r["iter_median_tpot_ms"],
            "bench_median_tpot_ms": r["bench_median_tpot_ms"],
            "tpot_discrepancy_pct": r["tpot_discrepancy_pct"],
            "gen_output_tput_per_user": r["gen_output_tput_per_user"],
            "gen_output_tput_per_gpu": r["gen_output_tput_per_gpu"],
            "ctx_gpus_needed": r.get("ctx_gpus_needed"),
            "total_gpus": r.get("total_gpus"),
            "gen_total_tput_per_gpu": r.get("gen_total_tput_per_gpu"),
            "job_dir": r["job_dir"],
        }
        pareto_rows.append(row)

    # Add KVP=1 Pareto points.
    for r in tp_pareto:
        row = {
            "category": "KVP=1",
            "deduped": r["job_dir"] not in tp_deduped_dirs,
            "slurm_id": r["slurm_id"],
            "global_batch_size": r["global_batch_size"],
            "gen_tp": r["gen_tp"],
            "gen_cp": r["gen_cp"],
            "gen_pp": r["gen_pp"],
            "gen_ep": r["gen_ep"],
            "is_attn_dp": r["is_attn_dp"],
            "moe_backend": r["moe_backend"],
            "iter_median_tpot_ms": r["iter_median_tpot_ms"],
            "bench_median_tpot_ms": r["bench_median_tpot_ms"],
            "tpot_discrepancy_pct": r["tpot_discrepancy_pct"],
            "gen_output_tput_per_user": r["gen_output_tput_per_user"],
            "gen_output_tput_per_gpu": r["gen_output_tput_per_gpu"],
            "ctx_gpus_needed": r.get("ctx_gpus_needed"),
            "total_gpus": r.get("total_gpus"),
            "gen_total_tput_per_gpu": r.get("gen_total_tput_per_gpu"),
            "job_dir": r["job_dir"],
        }
        pareto_rows.append(row)

    # Write Pareto CSV.
    if pareto_rows:
        pareto_fieldnames = [
            "category",
            "deduped",
            "slurm_id",
            "global_batch_size",
            "gen_tp",
            "gen_cp",
            "gen_pp",
            "gen_ep",
            "is_attn_dp",
            "moe_backend",
            "iter_median_tpot_ms",
            "bench_median_tpot_ms",
            "tpot_discrepancy_pct",
            "gen_output_tput_per_user",
            "gen_output_tput_per_gpu",
            "ctx_gpus_needed",
            "total_gpus",
            "gen_total_tput_per_gpu",
            "job_dir",
        ]
        with open(pareto_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pareto_fieldnames)
            writer.writeheader()
            writer.writerows(pareto_rows)

    print(f"\nPareto plots saved:")
    print(f"  Full plot: {plot_path_full}")
    print(f"  Denoised plot: {plot_path_denoised}")
    print(f"  Pareto CSV: {pareto_csv_path}")
    print(f"  KVP>1 jobs: {len(cp_jobs)} total, {len(cp_pareto)} Pareto, {len(cp_pareto_deduped)} kept after dedupe")
    print(f"  KVP=1 jobs: {len(tp_only_jobs)} total, {len(tp_pareto)} Pareto, {len(tp_pareto_deduped)} kept after dedupe")

    return plot_path_full


def write_csv(results: List[Dict], output_dir: str) -> str:
    """Write results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gen_only_results.csv")

    # Sort results by: num_gpus, global_batch_size, is_attn_dp, gen_tp, gen_cp.
    def sort_key(r):
        tp = r.get("gen_tp") or 0
        cp = r.get("gen_cp") or 0
        pp = r.get("gen_pp") or 0
        num_gpus = tp * cp * pp
        return (
            num_gpus,
            r.get("global_batch_size") or 0,
            1 if r.get("is_attn_dp") else 0,  # attnDP enabled first.
            tp,
            cp,
        )

    results = sorted(results, key=sort_key)

    fieldnames = [
        "slurm_id",
        "num_gpus",
        "global_batch_size",
        "is_attn_dp",
        "gen_tp",
        "gen_cp",
        "gen_pp",
        "gen_ep",
        "moe_backend",
        "is_complete",
        "iter_median_tpot_ms",      # TPOT from iteration logs (used for throughput)
        "bench_median_tpot_ms",     # TPOT from benchmark client (reference only)
        "tpot_discrepancy_pct",     # % difference: (bench - iter) / iter * 100
        "done_file_present",
        "saturation_pct",
        "saturated_iters",
        "total_iters",
        "first_saturated_iter",
        "last_saturated_iter",
        "gen_output_tput_per_user",
        "gen_output_tput_per_gpu",
        # Rate-matched totals (input + output tokens over gen + ctx GPUs)
        "isl",
        "osl",
        "gen_req_rate_per_s",
        "ctx_gpus_needed",
        "total_gpus",
        "gen_total_tput_per_gpu",
        "job_dir",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Parse gen-only benchmark job directories and extract metrics to CSV."
    )
    parser.add_argument(
        "--dir_list",
        nargs="+",
        required=True,
        help="List of paths; each can be a job directory or a parent containing job directories.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the CSV results file will be saved.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate Pareto frontier plot for CP vs TP-only jobs (default: True).",
    )
    parser.add_argument(
        "--ctx-req-rate",
        type=float,
        default=DEFAULT_CTX_REQ_RATE_PER_GPU,
        help="Prefill requests/sec per TP=1 ctx server (1 GPU), used to size "
        f"the ctx pool for rate-matched totals (default: {DEFAULT_CTX_REQ_RATE_PER_GPU}).",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=None,
        help="Input sequence length. Defaults to the '<isl>-<osl>' log dir name.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help="Output sequence length. Defaults to the '<isl>-<osl>' log dir name.",
    )
    args = parser.parse_args()

    # Find all job directories.
    job_dirs = find_job_directories(args.dir_list)
    print(f"Found {len(job_dirs)} job directories to analyze")

    if not job_dirs:
        print("No job directories found. Exiting.")
        return

    # Analyze each job.
    results = []
    for job_dir in job_dirs:
        print(f"Analyzing: {job_dir}")
        result = analyze_job(
            job_dir,
            isl=args.isl,
            osl=args.osl,
            ctx_req_rate=args.ctx_req_rate,
        )
        results.append(result)

    # Write results to CSV.
    output_path = write_csv(results, args.output_dir)
    print(f"\nResults written to: {output_path}")
    print(f"Total jobs analyzed: {len(results)}")

    # Summary.
    complete_count = sum(1 for r in results if r["is_complete"])
    print(f"Completed jobs: {complete_count}/{len(results)}")

    # Generate Pareto plots if requested: decode-only output throughput, and
    # rate-matched total throughput over gen + ctx GPUs.
    if args.plot:
        plot_pareto(results, args.output_dir)
        plot_pareto(
            results,
            args.output_dir,
            y_key="gen_total_tput_per_gpu",
            y_label="Total Throughput per GPU (in+out tok/s/gpu)",
            title=(
                "MiniMax M3 NVFP4 8k/1k TRTLLM Pareto, GB300 NVL72, Rate-Matched "
                f"Total (ctx {args.ctx_req_rate} req/s/gpu), #Gen GPUs Per Instance<=32"
            ),
            file_prefix="pareto_plot_total",
            csv_name="pareto_frontier_total.csv",
        )


if __name__ == "__main__":
    main()
