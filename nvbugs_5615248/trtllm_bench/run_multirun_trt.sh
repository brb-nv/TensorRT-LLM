#!/usr/bin/env bash
# Multi-run TRT trtllm-bench launcher for the NVBug 5615248 beam-10 workload.
#
# Mirrors run_multirun_pytorch.sh but targets the prebuilt TRT engine at
# nvbugs_5615248/tinyllama_trt_engine. Used to capture a fresh TRT baseline
# alongside the v3+v4+v5 PyTorch numbers so the two are statistically
# comparable on the same node and the same trtllm-bench version.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_multirun_trt.sh \
#        nvbugs_5615248/trtllm_bench/optimized_v5_trt_run1
#
# Output layout (per run i in {1..NUM_RUNS}; i=1 has no numeric suffix):
#   <OUT>/report_trt[$i].json
#   <OUT>/request_trt[$i].json
#   <OUT>/output_trt[$i].json
#   <OUT>/run_trt[$i].log

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
CONFIG="${WORKDIR}/trt.yaml"
ENGINE_DIR="${ENGINE_DIR:-${REPO_DIR}/nvbugs_5615248/tinyllama_trt_engine}"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset not found at ${DATASET} - see REPRO.md Step 1." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: trt.yaml not found at ${CONFIG}." >&2
    exit 1
fi
if [[ ! -d "${ENGINE_DIR}" ]]; then
    echo "ERROR: TRT engine dir not found at ${ENGINE_DIR}." >&2
    exit 1
fi
if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_one () {
    local suffix="$1"

    local report_json="${OUT_DIR}/report_trt${suffix}.json"
    local output_json="${OUT_DIR}/output_trt${suffix}.json"
    local request_json="${OUT_DIR}/request_trt${suffix}.json"
    local log_file="${OUT_DIR}/run_trt${suffix}.log"

    echo "=================================================================="
    echo "[multi-run trt] run #${suffix:-1} -> ${report_json}"
    echo "=================================================================="

    trtllm-bench \
        --model "${MODEL}" \
        --model_path "${MODEL}" \
        --workspace "${WORKDIR}" \
        throughput \
        --backend tensorrt \
        --engine_dir "${ENGINE_DIR}" \
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

run_one ""
for ((i = 2; i <= NUM_RUNS; i++)); do
    run_one "${i}"
done

echo
echo "Done. ${NUM_RUNS} TRT runs written under: ${OUT_DIR}"
echo "Aggregate against the matching PyT dir with:"
echo "  python3 nvbugs_5615248/trtllm_bench/aggregate_runs.py \\"
echo "      --backend trt \\"
echo "      --baseline   <pyt_dir> \\"
echo "      --baseline-label pytorch \\"
echo "      --experiment ${OUT_DIR} \\"
echo "      --experiment-label trt"
