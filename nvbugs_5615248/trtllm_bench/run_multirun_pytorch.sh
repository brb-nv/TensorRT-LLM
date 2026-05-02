#!/usr/bin/env bash
# Multi-run PyTorch trtllm-bench launcher for the NVBug 5615248 beam-10 workload.
#
# Mirrors the v3 protocol (5 independent runs, 16 requests each, --concurrency 1,
# --streaming, beam=10) so before/after comparisons stay apples-to-apples.
# Writes all artifacts into an output directory passed on the CLI.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_multirun_pytorch.sh \
#        nvbugs_5615248/trtllm_bench/optimized_v4
#
# Output layout (per run i in {1..5}; i=1 has no numeric suffix):
#   <OUT>/report_pytorch[$i].json   - aggregate report
#   <OUT>/request_pytorch[$i].json  - per-request timestamps + latencies (ns)
#   <OUT>/output_pytorch[$i].json   - generated tokens
#   <OUT>/run_pytorch[$i].log       - full stdout/stderr

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <output_dir>" >&2
    exit 2
fi

OUT_DIR="$1"
NUM_RUNS="${NUM_RUNS:-5}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
DATASET="${WORKDIR}/dataset_isl100_osl20.jsonl"
CONFIG="${WORKDIR}/pytorch.yaml"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset not found at ${DATASET} — see REPRO.md Step 1." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: pytorch.yaml not found at ${CONFIG}." >&2
    exit 1
fi
if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_one () {
    local suffix="$1"  # "" for run 1, "2".."5" for runs 2..5

    local report_json="${OUT_DIR}/report_pytorch${suffix}.json"
    local output_json="${OUT_DIR}/output_pytorch${suffix}.json"
    local request_json="${OUT_DIR}/request_pytorch${suffix}.json"
    local log_file="${OUT_DIR}/run_pytorch${suffix}.log"

    echo "=================================================================="
    echo "[multi-run pyt v4] run #${suffix:-1} -> ${report_json}"
    echo "=================================================================="

    trtllm-bench \
        --model "${MODEL}" \
        --model_path "${MODEL}" \
        --workspace "${WORKDIR}" \
        throughput \
        --backend pytorch \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --concurrency 1 \
        --warmup 3 \
        --num_requests 16 \
        --beam_width 10 \
        --max_batch_size 1 \
        --streaming \
        --report_json "${report_json}" \
        --output_json "${output_json}" \
        --request_json "${request_json}" \
        2>&1 | tee "${log_file}"
}

# Run 1 (unsuffixed), then runs 2..NUM_RUNS (suffixed) — mirrors v3 layout.
run_one ""
for ((i = 2; i <= NUM_RUNS; i++)); do
    run_one "${i}"
done

echo
echo "Done. ${NUM_RUNS} runs written under: ${OUT_DIR}"
echo "Aggregate with:"
echo "  python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \\"
echo "      --baseline nvbugs_5615248/trtllm_bench/optimized_v3 \\"
echo "      --experiment ${OUT_DIR}"
