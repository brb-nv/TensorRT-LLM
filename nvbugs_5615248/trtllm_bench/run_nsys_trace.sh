#!/usr/bin/env bash
# Single-run nsys trace launcher for the NVBug 5615248 beam-10 workload.
#
# Captures one nsys trace per backend with the same workload as the multi-run
# bench, plus a 5-prefill warmup so analyze_decode_loop.py can carve out a
# clean steady-state window. Output is a .nsys-rep + an exported .sqlite that
# analyze_decode_loop.py reads.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_nsys_trace.sh pytorch \
#        nvbugs_5615248/trtllm_bench/nsys_optimized_v5_pyt
#   bash nvbugs_5615248/trtllm_bench/run_nsys_trace.sh tensorrt \
#        nvbugs_5615248/trtllm_bench/nsys_optimized_v5_trt

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <pytorch|tensorrt> <output_dir>" >&2
    exit 2
fi

BACKEND="$1"
OUT_DIR="$2"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
DATASET="${WORKDIR}/dataset_isl100_osl20.jsonl"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"
ENGINE_DIR="${ENGINE_DIR:-${REPO_DIR}/nvbugs_5615248/tinyllama_trt_engine}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi
if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH." >&2
    exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset not found at ${DATASET} - see REPRO.md Step 1." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

case "${BACKEND}" in
    pytorch)
        TAG="pyt_beam10"
        CONFIG="${WORKDIR}/pytorch.yaml"
        BACKEND_ARGS=(--backend pytorch --config "${CONFIG}")
        ;;
    tensorrt)
        TAG="trt_beam10"
        CONFIG="${WORKDIR}/trt.yaml"
        if [[ ! -d "${ENGINE_DIR}" ]]; then
            echo "ERROR: TRT engine dir not found at ${ENGINE_DIR}." >&2
            exit 1
        fi
        BACKEND_ARGS=(--backend tensorrt --engine_dir "${ENGINE_DIR}" --config "${CONFIG}")
        ;;
    *)
        echo "ERROR: unknown backend '${BACKEND}' (expected: pytorch | tensorrt)" >&2
        exit 2
        ;;
esac

REPORT="${OUT_DIR}/${TAG}.nsys-rep"
SQLITE="${OUT_DIR}/${TAG}.sqlite"
RUN_LOG="${OUT_DIR}/run_${BACKEND}.log"

echo "=================================================================="
echo "[nsys trace] backend=${BACKEND} -> ${REPORT}"
echo "=================================================================="

# nsys profile flags chosen to match nsys_decode_loop_v4 (CUDA + NVTX traces,
# OS-runtime + thread-state sampling, CUDA graph trace in graph mode).
nsys profile \
    --output "${REPORT%.nsys-rep}" \
    --force-overwrite=true \
    --sample=none \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=graph \
    --capture-range=none \
    --stats=false \
    -- \
    trtllm-bench \
        --model "${MODEL}" \
        --model_path "${MODEL}" \
        --workspace "${WORKDIR}" \
        throughput \
        "${BACKEND_ARGS[@]}" \
        --dataset "${DATASET}" \
        --concurrency 1 \
        --warmup 5 \
        --num_requests 20 \
        --beam_width 10 \
        --max_batch_size 1 \
        --streaming \
        --report_json  "${OUT_DIR}/report_${BACKEND}.json" \
        --output_json  "${OUT_DIR}/output_${BACKEND}.json" \
        --request_json "${OUT_DIR}/request_${BACKEND}.json" \
    2>&1 | tee "${RUN_LOG}"

echo
echo "[nsys trace] exporting sqlite -> ${SQLITE}"
nsys export --type sqlite --output "${SQLITE}" --force-overwrite=true "${REPORT}"

echo
echo "Done. Inputs ready for analyze_decode_loop.py:"
echo "  python3 nvbugs_5615248/trtllm_bench/analyze_decode_loop.py \\"
echo "      --sqlite ${SQLITE} --skip-prefills 5"
