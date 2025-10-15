#!/usr/bin/env python3
"""
Script to parse benchmark.log files from multiple folders and calculate median timing values.

Usage:
    python parse_benchmark_logs.py <folder1> [folder2] [folder3] ...

The script looks for benchmark.log in each folder and extracts timing information
for different configurations (dense/moe, ctx_len, tp, kvp, ep).
It verifies all ranks are present and calculates the median time for each configuration.
"""

import csv
import re
import sys
from pathlib import Path

JOB_ID_SPEC_SEQ = re.compile(r'(\d+)\.\.(\d+)')
BENCH_LOG_SUB_FOLDER = re.compile(r"benchmark-(\d+)-(\d+)")
TPOT_MEAN_LINE = re.compile(r"Mean TPOT \(ms\):\s+(\d+\.\d+)")
TPOT_MEDIAN_LINE = re.compile(r"Median TPOT \(ms\):\s+(\d+\.\d+)")
TP_YAML = re.compile(r"tensor_parallel_size: (\d+)")
EP_YAML = re.compile(r"moe_expert_parallel_size: (\d+)")
CP_YAML = re.compile(r"context_parallel_size: (\d+)")


def parse_benchmark_log(log_file):
    """
    Parse a benchmark.log file and extract timing information.

    Returns:
        dict: Configuration key mapped to rank time.
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    timing_lines = [
        (i, line.strip()) for i, line in enumerate(lines)
        if "-----Time per Output Token (excl. 1st token)------" in line
    ]
    if len(timing_lines) != 1:
        print(
            f"Warning: {log_file} does not contain timing information. No line contained the TPOT header.",
            file=sys.stderr)
        return None

    i_line, timing_line = timing_lines[0]
    mean_match = TPOT_MEAN_LINE.search(lines[i_line + 1])
    median_match = TPOT_MEDIAN_LINE.search(lines[i_line + 2])
    if not mean_match or not median_match:
        print(
            f"Warning: {log_file} does not contain timing information. "
            f"Lines {lines[i_line + 1].strip()} and {lines[i_line + 2].strip()} "
            "did not contain the TPOT information.",
            file=sys.stderr)
        return None
    tpot_mean = float(mean_match.group(1))
    tpot_median = float(median_match.group(1))

    return tpot_mean, tpot_median


def calculate_stats(data):
    """
    Calculate medians for configurations with complete rank sets.

    Args:
        data: Dictionary mapping configuration keys to rank data.

    Returns:
        list: List of dictionaries containing configuration and median time.
    """
    results = []

    for config_key, config_data in data.items():
        type_name, ctx_len, ctx_len_per_gpu, tp, kvp, ep = config_key
        ranks = config_data['ranks']
        total_gpus = config_data['total_gpus']

        if total_gpus is None:
            print(f"Warning: Missing total_gpus for config {config_key}",
                  file=sys.stderr)
            continue
        if total_gpus <= 0:
            print(
                f"Warning: Total GPUs {total_gpus} for config {config_key} is invalid",
                file=sys.stderr)
            continue

        # Check if all ranks from 0 to total_gpus-1 are present
        expected_ranks = set(range(total_gpus))
        actual_ranks = set(ranks.keys())

        if expected_ranks == actual_ranks:
            # All ranks present, calculate median
            times = sorted(ranks[r][1] for r in sorted(ranks.keys()))
            print(
                f"All ranks present for config {config_key}: times {' '.join(f'{t:.2f}' for t in times)}"
            )
            min_time = times[0]
            if len(times) % 2 == 0:
                median_time = (times[len(times) // 2] +
                               times[len(times) // 2 - 1]) / 2.
            else:
                median_time = times[len(times) // 2]

            results.append({
                'type': type_name,
                'ctx_len': ctx_len,
                'ctx_len_per_gpu': ctx_len_per_gpu,
                'tp': tp,
                'kvp': kvp,
                'ep': ep,
                'median_time_ms': median_time,
                'min_time_ms': min_time
            })
        else:
            missing_ranks = expected_ranks - actual_ranks
            print(
                f"Warning: Missing ranks {missing_ranks} for config {config_key}",
                file=sys.stderr)

    return results


def main():
    """Main function to process benchmark logs from multiple folders."""
    job_id_pattern = "(<job_id_start>..<job_id_end>|<job_id1,job_id2,job_id3>...)"
    if len(sys.argv) < 2:
        print("Usage: python parse_benchmark_logs.py " + job_id_pattern,
              file=sys.stderr)
        sys.exit(1)

    job_id_spec = sys.argv[1]
    if JOB_ID_SPEC_SEQ.match(job_id_spec):
        job_id_start, job_id_end = map(int, job_id_spec.split('..'))
        job_ids = list(range(job_id_start, job_id_end + 1))
    else:
        job_ids = job_id_spec.split(',')
        try:
            job_ids = list(map(int, job_ids))
        except ValueError:
            print(
                f"Invalid job ID spec: {job_id_spec}, "
                f"expected format: {job_id_pattern}",
                file=sys.stderr)
            sys.exit(1)

    all_results = []

    for job_id in job_ids:
        folder_path = Path(f"slurm-{job_id}")

        if not folder_path.is_dir():
            print(f"Warning: {folder_path} not found, skipping",
                  file=sys.stderr)
            continue

        # we expect the folder to contain exactly one sub-folder
        sub_folders1 = list(folder_path.iterdir())
        if len(sub_folders1) != 1:
            print(
                f"Warning: {folder_path} contains {len(sub_folders1)} sub-folders, expected 1",
                file=sys.stderr)
            continue
        sub_folder1 = sub_folders1[0]
        match = BENCH_LOG_SUB_FOLDER.match(sub_folder1.name)
        if not match:
            print(
                f"Warning: {sub_folder1.name} is not a valid benchmark log sub-folder name",
                file=sys.stderr)
            continue
        isl = int(match.group(1))
        # we expect the sub-folder to contain exactly one sub-folder
        sub_folders2 = list(sub_folder1.iterdir())
        if len(sub_folders2) != 1:
            print(
                f"Warning: {sub_folder1} contains {len(sub_folders2)} sub-folders, expected 1",
                file=sys.stderr)
            continue
        sub_folder2 = sub_folders2[0]
        log_file = sub_folder2 / 'benchmark.log'
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping", file=sys.stderr)
            continue
        config_file = sub_folder2 / 'gen_config.yaml'
        if not config_file.exists():
            print(f"Warning: {config_file} not found, skipping",
                  file=sys.stderr)
            continue
        with open(config_file, 'r') as f:
            config = f.read()
        tp_match = TP_YAML.search(config)
        ep_match = EP_YAML.search(config)
        cp_match = CP_YAML.search(config)
        if not tp_match or not ep_match or not cp_match:
            print(
                f"Warning: {config_file} does not contain configuration information",
                file=sys.stderr)
            continue
        tp = int(tp_match.group(1))
        ep = int(ep_match.group(1))
        cp = int(cp_match.group(1))

        config_key = (tp, cp, ep, isl)

        print(f"Processing {log_file} for config {config_key}...",
              file=sys.stderr)
        data = parse_benchmark_log(log_file)
        if data is None:
            continue
        for i, d in enumerate(data):
            if len(all_results) <= i:
                all_results.append(dict())
            all_results[i][config_key] = d

    # Write results to CSV
    for i, results in enumerate(all_results):
        all_isls = set(config_key[-1] for config_key in results)
        all_isls = sorted(all_isls)

        all_configs = set(config_key[:-1] for config_key in results)
        all_configs = sorted(all_configs)

        fieldnames = [
            'tp',
            'kvp',
            'ep',
        ] + all_isls
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for config in all_configs:
            row = {'tp': config[0], 'kvp': config[1], 'ep': config[2]}
            for isl in all_isls:
                config_key = config + (isl, )
                if config_key in results:
                    row[isl] = results[config_key]
            writer.writerow(row)


if __name__ == '__main__':
    main()
