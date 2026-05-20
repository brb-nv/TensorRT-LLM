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
#
# PyTorch leg of the NVBug 5615248 TRT-vs-PyT repro.
#
# Runs trtllm-bench in throughput mode against TinyLlama with beam=10, 16
# requests per run, --concurrency 1, --streaming. Reads ALL the relevant
# knobs (including the two feature flags under test:
# enable_early_first_token_response and enable_speculative_beam_history_d2h)
# from pytorch_repro.yaml so the run is fully described by the config in
# this directory.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_multirun_pytorch_repro.sh <output_dir>
#
# Optional env vars:
#   NUM_RUNS  default: 5
#   MODEL     default: /home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
#             MUST be a local directory containing the HF model
#             (config.json + safetensors); HF repo ids are NOT accepted
#             by trtllm-bench's --model_path (click parses it as a Path,
#             which the HF Hub validator rejects). See repro.md for the
#             huggingface-cli download recipe.
#
# Output layout (per run i in {1..NUM_RUNS}; i=1 has no numeric suffix):
#   <OUT>/report_pytorch[$i].json   - aggregate report
#   <OUT>/request_pytorch[$i].json  - per-request timestamps + latencies (ns)
#   <OUT>/output_pytorch[$i].json   - generated tokens
#   <OUT>/run_pytorch[$i].log       - full stdout/stderr
#   <OUT>/env.txt                   - per-leg env summary

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <output_dir>" >&2
    exit 2
fi

OUT_DIR="$1"
NUM_RUNS="${NUM_RUNS:-5}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
DATASET="${WORKDIR}/dataset_isl100_osl20_repro.jsonl"
CONFIG="${WORKDIR}/pytorch_repro.yaml"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: dataset not found at ${DATASET}." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: pytorch_repro.yaml not found at ${CONFIG}." >&2
    exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
    cat >&2 <<EOF
ERROR: MODEL='${MODEL}' is not a local directory.
       trtllm-bench needs MODEL to point at a local HuggingFace model
       directory containing config.json + safetensors. HF repo ids are
       NOT accepted (click parses --model_path as pathlib.Path, which
       HF Hub's validator rejects).

       Either point MODEL at an existing local copy, or download it once:
         huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
             --local-dir \$PWD/nvbugs_5615248/tinyllama_hf
         export MODEL=\$PWD/nvbugs_5615248/tinyllama_hf
EOF
    exit 1
fi
if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# Capture the config that was actually used so post-hoc analysis is unambiguous.
{
    echo "[multi-run pyt repro] MODEL=${MODEL}"
    echo "[multi-run pyt repro] CONFIG=${CONFIG}"
    echo "[multi-run pyt repro] DATASET=${DATASET}"
    echo "[multi-run pyt repro] NUM_RUNS=${NUM_RUNS}"
    echo "[multi-run pyt repro] feature flags (from YAML):"
    grep -E '^(enable_early_first_token_response|enable_speculative_beam_history_d2h):' \
        "${CONFIG}" || true
} | tee "${OUT_DIR}/env.txt"

run_one () {
    local suffix="$1"  # "" for run 1, "2".."NUM_RUNS" for the rest

    local report_json="${OUT_DIR}/report_pytorch${suffix}.json"
    local output_json="${OUT_DIR}/output_pytorch${suffix}.json"
    local request_json="${OUT_DIR}/request_pytorch${suffix}.json"
    local log_file="${OUT_DIR}/run_pytorch${suffix}.log"

    echo "=================================================================="
    echo "[multi-run pyt repro] run #${suffix:-1} -> ${report_json}"
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
echo "Done. ${NUM_RUNS} PyT (feature) runs written under: ${OUT_DIR}"
