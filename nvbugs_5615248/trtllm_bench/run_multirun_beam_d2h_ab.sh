#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A/B multi-run trtllm-bench launcher for the beam-history speculative
# D2H opt-in (TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H).
#
# Mirrors run_multirun_pytorch.sh (5 runs, 16 requests each, --concurrency 1,
# --streaming, beam=10) and run_multirun_prefill_ab.sh, but toggles the
# beam-history speculative-D2H opt-in via positional arg:
#   on  -> TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=1
#   off -> TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H unset (default)
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_multirun_beam_d2h_ab.sh on  <out_dir>
#   bash nvbugs_5615248/trtllm_bench/run_multirun_beam_d2h_ab.sh off <out_dir>
#
# Output layout (per run i in {1..NUM_RUNS}; i=1 has no numeric suffix), kept
# identical to run_multirun_pytorch.sh so aggregate_runs.py works unchanged:
#   <OUT>/report_pytorch[$i].json
#   <OUT>/request_pytorch[$i].json
#   <OUT>/output_pytorch[$i].json
#   <OUT>/run_pytorch[$i].log

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <on|off> <output_dir>" >&2
    exit 2
fi

MODE="$1"
OUT_DIR="$2"
NUM_RUNS="${NUM_RUNS:-5}"

case "${MODE}" in
    on)  export TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=1 ;;
    off) unset  TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H ;;
    *) echo "ERROR: first arg must be 'on' or 'off' (got '${MODE}')" >&2; exit 2 ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
DATASET="${WORKDIR}/dataset_isl100_osl20.jsonl"
CONFIG="${WORKDIR}/pytorch.yaml"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset not found at ${DATASET}." >&2
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

echo "[beam-D2H A/B pyt] mode=${MODE} TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=${TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H:-<unset>}" \
    | tee "${OUT_DIR}/env.txt"

run_one () {
    local suffix="$1"

    local report_json="${OUT_DIR}/report_pytorch${suffix}.json"
    local output_json="${OUT_DIR}/output_pytorch${suffix}.json"
    local request_json="${OUT_DIR}/request_pytorch${suffix}.json"
    local log_file="${OUT_DIR}/run_pytorch${suffix}.log"

    echo "=================================================================="
    echo "[beam-D2H A/B pyt] mode=${MODE} run #${suffix:-1} -> ${report_json}"
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

run_one ""
for ((i = 2; i <= NUM_RUNS; i++)); do
    run_one "${i}"
done

echo
echo "Done. ${NUM_RUNS} runs (mode=${MODE}) written under: ${OUT_DIR}"
