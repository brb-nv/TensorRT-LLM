#!/usr/bin/env bash
# Multi-run trtllm-bench launcher for the beam-search prefill->decode
# handoff workload (NVBug 5615248, TinyLlama-1.1B-Chat beam=10).
#
# Each invocation does NUM_RUNS independent trtllm-bench `throughput` runs
# (default 5) into the supplied output directory; the PyExecutor is
# cold-started fresh per run, so runs are independent samples for the
# Welch's t-test in aggregate_runs.py.
#
# Workload (matches the original measurement protocol in the source branch):
#   model:        TinyLlama-1.1B-Chat-v1.0
#   ISL/OSL:      100 / 20
#   beam_width:   10
#   max_batch:    1
#   --concurrency 1 --streaming --warmup 3 --num_requests 16
#
# Usage (from repo root, inside the TRT-LLM container):
#   bash perf_repro/beam_search_handoff/run_multirun.sh <output_dir>
#
# Optional env overrides:
#   MODEL=<path>          override TinyLlama path
#   NUM_RUNS=<int>        number of independent runs (default 5)
#   NUM_REQUESTS=<int>    requests per run (default 16)
#   BEAM_WIDTH=<int>      beam width (default 10)
#
# Output layout (per run i in {1..NUM_RUNS}; i=1 is unsuffixed to match
# aggregate_runs.py's REQUEST_RE pattern `request_<backend>{,2..N}.json`):
#   <OUT>/report_pytorch[$i].json
#   <OUT>/request_pytorch[$i].json
#   <OUT>/output_pytorch[$i].json
#   <OUT>/run_pytorch[$i].log

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <output_dir>" >&2
    exit 2
fi

OUT_DIR="$1"
NUM_RUNS="${NUM_RUNS:-5}"
NUM_REQUESTS="${NUM_REQUESTS:-16}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/perf_repro/beam_search_handoff"
CONFIG="${WORKDIR}/pytorch.yaml"
DATASET="${DATASET:-${WORKDIR}/dataset_isl100_osl20.jsonl}"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: pytorch.yaml not found at ${CONFIG}." >&2
    exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
    cat >&2 <<EOF
ERROR: dataset not found at ${DATASET}

Generate it once with:

  trtllm-bench \\
      --model "${MODEL}" \\
      --model_path "${MODEL}" \\
      --workspace "${WORKDIR}" \\
      prepare-dataset \\
      --output "${DATASET}" \\
      token-norm-dist \\
      --num-requests 32 \\
      --input-mean 100 --input-stdev 0 \\
      --output-mean 20  --output-stdev 0
EOF
    exit 1
fi
if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM env?)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

run_one () {
    local suffix="$1"  # "" for run 1, "2".."NUM_RUNS" for runs 2..

    local report_json="${OUT_DIR}/report_pytorch${suffix}.json"
    local output_json="${OUT_DIR}/output_pytorch${suffix}.json"
    local request_json="${OUT_DIR}/request_pytorch${suffix}.json"
    local log_file="${OUT_DIR}/run_pytorch${suffix}.log"

    echo "=================================================================="
    echo "[multi-run] run #${suffix:-1} -> ${report_json}"
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
        --num_requests "${NUM_REQUESTS}" \
        --beam_width "${BEAM_WIDTH}" \
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
echo "Done. ${NUM_RUNS} runs written under: ${OUT_DIR}"
echo
echo "Aggregate (single dir):"
echo "  python3 ${WORKDIR}/aggregate_runs.py --experiment ${OUT_DIR}"
echo
echo "Compare (before/after):"
echo "  python3 ${WORKDIR}/aggregate_runs.py --baseline <BEFORE_DIR> --experiment ${OUT_DIR}"
