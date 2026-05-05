#!/usr/bin/env bash
# Generate a battery of nsys stats / analyze reports for the NVBug 5615248
# overlap-scheduler TTFT profiles. Designed to run inside a container that
# already has `nsys` on PATH.
#
# Outputs land in nvbugs_5615248/nsys_analysis/<tag>/ where <tag> is
# `overlap_on` or `overlap_off`.
#
# Each .nsys-rep file contains 5 NVTX-tagged measurement requests
# (overlap_<tag>_measurement_{0..4}) captured via the cudaProfilerApi range.

set -euo pipefail

REPO_DIR="/home/scratch.bbuddharaju_gpu/TensorRT-LLM"
PROF_DIR="${REPO_DIR}/nvbugs_5615248"
OUT_BASE="${PROF_DIR}/nsys_analysis"

mkdir -p "${OUT_BASE}"

# Sanity: confirm nsys is reachable, record version.
{
    echo "=== nsys version ==="
    nsys --version
    echo
    echo "=== nsys path ==="
    command -v nsys
} > "${OUT_BASE}/nsys_env.txt" 2>&1 || {
    echo "ERROR: nsys not found on PATH" >&2
    exit 1
}

# Reports we want for both runs.
# Each entry is "<report_name>:<format>". `column` is human-friendly text;
# `csv` is machine-friendly for follow-up analysis.
COMMON_REPORTS=(
    "cuda_gpu_kern_sum:column"      # top kernels by GPU time
    "cuda_gpu_kern_sum:csv"
    "cuda_api_sum:column"           # CUDA runtime/driver API time
    "cuda_api_sum:csv"
    "cuda_gpu_mem_time_sum:column"  # H2D/D2H/sync memcpy time
    "cuda_kern_exec_sum:column"     # API time vs queue time vs kernel time
    "cuda_gpu_trace:csv"            # full kernel trace (timestamps) -> per-iter slicing
    "nvtx_pushpop_sum:column"       # range duration aggregates
    "nvtx_pushpop_sum:csv"
    "nvtx_gpu_proj_sum:column"      # GPU activity projected onto NVTX ranges
    "nvtx_gpu_proj_sum:csv"
    "nvtx_kern_sum:column"          # kernels grouped by enclosing NVTX range
    "nvtx_kern_sum:csv"
    "osrt_sum:column"               # OS runtime calls (incl. sync waits)
)

run_reports_for_trace () {
    local trace="$1"   # absolute path to .nsys-rep
    local tag="$2"     # short label, e.g. overlap_on
    local out="${OUT_BASE}/${tag}"
    mkdir -p "${out}"

    echo "=================================================================="
    echo "Processing ${trace}"
    echo "  -> ${out}"
    echo "=================================================================="

    # 1. Expert-system anti-pattern detection (single combined output).
    echo "--- nsys analyze -r all ---"
    nsys analyze -r all "${trace}" \
        > "${out}/analyze_all.txt" 2>&1 || true

    # 2. Per-report stats. Use --force-export=true on the first invocation
    #    to materialize the SQLite cache; subsequent calls reuse it.
    local first_call=1
    for spec in "${COMMON_REPORTS[@]}"; do
        local report="${spec%%:*}"
        local fmt="${spec##*:}"
        local outfile="${out}/${report}.${fmt}.txt"
        [[ "${fmt}" == "csv" ]] && outfile="${out}/${report}.csv"

        echo "--- nsys stats -r ${report} (${fmt}) ---"
        if (( first_call )); then
            nsys stats --force-export=true -r "${report}" -f "${fmt}" \
                "${trace}" > "${outfile}" 2>&1 || true
            first_call=0
        else
            nsys stats -r "${report}" -f "${fmt}" \
                "${trace}" > "${outfile}" 2>&1 || true
        fi
    done
}

run_reports_for_trace "${PROF_DIR}/trace_overlap_on.nsys-rep"  "overlap_on"
run_reports_for_trace "${PROF_DIR}/trace_overlap_off.nsys-rep" "overlap_off"

# Final sanity dump: list everything we produced.
{
    echo "=== Generated files ==="
    find "${OUT_BASE}" -type f -printf "%p  (%s bytes)\n" | sort
} > "${OUT_BASE}/INDEX.txt"

echo
echo "Done. Outputs under: ${OUT_BASE}"
echo "See ${OUT_BASE}/INDEX.txt for the full file list."
