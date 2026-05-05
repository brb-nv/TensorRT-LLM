#!/usr/bin/env bash
# Generate nsys stats reports for the PyTorch-vs-TRT TTFT traces of NVBug 5615248.
#
# Run inside the container (where `nsys` is available), e.g.:
#   crun -q 'gpu.product_name=*L40S*' --gpus=1 -i -t 04:00:00 \
#       -img urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-26.02-...
#   # then inside:
#   bash nvbugs_5615248/trtllm_bench/run_nsys_stats.sh
#
# Outputs land in nvbugs_5615248/trtllm_bench/nsys_kernels/{pytorch,trt}/
# as text + CSV — small enough to read from the frontend afterwards.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
OUT_BASE="${WORKDIR}/nsys_kernels"

PYT_TRACE="${WORKDIR}/trace_pytorch_kernels.nsys-rep"
TRT_TRACE="${WORKDIR}/trace_trt_kernels.nsys-rep"

for t in "${PYT_TRACE}" "${TRT_TRACE}"; do
    if [[ ! -f "${t}" ]]; then
        echo "ERROR: missing trace file: ${t}" >&2
        exit 1
    fi
done

mkdir -p "${OUT_BASE}/pytorch" "${OUT_BASE}/trt"

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not on PATH (are you inside the TRT-LLM container?)" >&2
    exit 1
fi

{
    echo "=== nsys version ==="
    nsys --version
    echo
    echo "=== nsys path ==="
    command -v nsys
    echo
    echo "=== host ==="
    hostname
    uname -a
} > "${OUT_BASE}/nsys_env.txt" 2>&1

# Reports we want for both runs.
# Each entry is "<report_name>:<format>". `column` is human-friendly text;
# `csv` is machine-friendly for follow-up analysis from the frontend.
COMMON_REPORTS=(
    "cuda_gpu_kern_sum:column"      # top kernels by GPU time across the trace
    "cuda_gpu_kern_sum:csv"
    "nvtx_gpu_proj_sum:column"      # GPU time projected onto NVTX ranges
    "nvtx_gpu_proj_sum:csv"
    "nvtx_kern_sum:column"          # kernels grouped by enclosing NVTX range
    "nvtx_kern_sum:csv"
    "nvtx_pushpop_sum:column"       # CPU-side NVTX range duration aggregates
    "nvtx_pushpop_sum:csv"
    "cuda_kern_exec_sum:column"     # per-kernel API time vs queue time vs kernel time
    "cuda_kern_exec_sum:csv"
    "cuda_api_sum:column"           # CUDA runtime/driver API call counts and times
    "cuda_api_sum:csv"
    "cuda_gpu_mem_time_sum:column"  # H2D/D2H/sync memcpy time
    "cuda_gpu_trace:csv"            # full kernel trace with timestamps + NVTX context
)

run_reports_for_trace () {
    local trace="$1"
    local out="$2"

    echo "=================================================================="
    echo "Processing ${trace}"
    echo "  -> ${out}"
    echo "=================================================================="

    # 1) Anti-pattern detection.
    echo "--- nsys analyze -r all ---"
    nsys analyze -r all "${trace}" \
        > "${out}/analyze_all.txt" 2>&1 || true

    # 2) Per-report stats. Force sqlite export on the first call so the
    #    cache is materialized once; subsequent calls reuse it.
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

run_reports_for_trace "${PYT_TRACE}" "${OUT_BASE}/pytorch"
run_reports_for_trace "${TRT_TRACE}" "${OUT_BASE}/trt"

# Final index for the frontend reader.
{
    echo "=== Generated files (size in bytes) ==="
    find "${OUT_BASE}" -type f -printf "%p  (%s)\n" | sort
} > "${OUT_BASE}/INDEX.txt"

echo
echo "Done. Outputs under: ${OUT_BASE}"
echo "See ${OUT_BASE}/INDEX.txt for the full file list."
