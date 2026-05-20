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
# Driver for the NVBug 5615248 TRT-vs-PyT comparison repro.
#
# Two legs (no commit swap; HEAD must contain the two TorchLlmArgs feature
# flags under test):
#
#   1. baseline_trt : TRT backend against the prebuilt TinyLlama engine
#                     at nvbugs_5615248/tinyllama_trt_engine.
#   2. feature_pyt  : PyTorch backend with both opt-ins on
#                     (enable_early_first_token_response=true and
#                      enable_speculative_beam_history_d2h=true,
#                      baked into pytorch_repro.yaml).
#
# Then aggregates `feature_pyt vs baseline_trt` with aggregate_runs_repro.py.
# Orientation: TRT is the baseline; PyT is the experiment. With this sign:
#
#   Δmedian = median(PyT) - median(TRT)
#     > 0  --  PyT slower than TRT  (residual gap)
#     < 0  --  PyT faster than TRT  (feature win)
#
# Because aggregate_runs_repro.py auto-detects the backend tag from the
# request_<backend>{,2..N}.json filenames, cross-backend aggregation goes
# through an ephemeral "shim" dir whose PyT artifacts are symlinked under
# request_trt*.json names so a single --backend tag works.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh [out_root]
#
# Optional env vars:
#   NUM_RUNS    default: 5  (forwarded to inner launchers)
#   MODEL       default: /home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
#               MUST be a local directory containing the HF model
#               (config.json + safetensors); HF repo ids are NOT accepted
#               by trtllm-bench's --model_path. See repro.md.
#   ENGINE_DIR  default: nvbugs_5615248/tinyllama_trt_engine
#   SKIP_TRT=1  re-aggregate only; expects ${OUT_ROOT}/baseline_trt/.done.
#   SKIP_PYT=1  re-aggregate only; expects ${OUT_ROOT}/feature_pyt/.done.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"

NUM_RUNS="${NUM_RUNS:-5}"
MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0}"
ENGINE_DIR="${ENGINE_DIR:-${REPO_DIR}/nvbugs_5615248/tinyllama_trt_engine}"

OUT_ROOT="${1:-${WORKDIR}/trt_vs_pyt_repro_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"
# Normalize to absolute so the cross-backend shim symlinks (which embed
# the target path) resolve correctly regardless of the user's CWD or how
# they spelled OUT_ROOT on the CLI.
OUT_ROOT="$(cd "${OUT_ROOT}" && pwd)"
DRIVER_LOG="${OUT_ROOT}/driver.log"

log() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "${DRIVER_LOG}"; }

# ---- Pre-flight ----------------------------------------------------------

cd "${REPO_DIR}"

if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

for f in run_multirun_pytorch_repro.sh run_multirun_trt_repro.sh \
         aggregate_runs_repro.py pytorch_repro.yaml trt_repro.yaml \
         dataset_isl100_osl20_repro.jsonl; do
    if [[ ! -f "${WORKDIR}/${f}" ]]; then
        echo "ERROR: required file missing: ${WORKDIR}/${f}" >&2
        exit 1
    fi
done

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
if [[ ! -d "${ENGINE_DIR}" ]]; then
    echo "ERROR: TRT engine dir not found at ${ENGINE_DIR}." >&2
    echo "       Build it first via:" >&2
    echo "         bash nvbugs_5615248/trtllm_bench/build_trt_engine_repro.sh" >&2
    exit 1
fi

TRT_DIR="${OUT_ROOT}/baseline_trt"
PYT_DIR="${OUT_ROOT}/feature_pyt"

log "OUT_ROOT=${OUT_ROOT}"
log "MODEL=${MODEL}"
log "ENGINE_DIR=${ENGINE_DIR}"
log "NUM_RUNS=${NUM_RUNS}"
log "HEAD=$(git -C "${REPO_DIR}" log -1 --oneline 2>/dev/null || echo unknown)"

# ---- Leg 1: TRT baseline -------------------------------------------------

if [[ -n "${SKIP_TRT:-}" ]]; then
    log "=== Leg 1/2: SKIP_TRT=1, skipping TRT phase."
    if [[ ! -f "${TRT_DIR}/.done" ]]; then
        echo "ERROR: SKIP_TRT=1 but ${TRT_DIR}/.done is missing." >&2
        exit 1
    fi
elif [[ -f "${TRT_DIR}/.done" ]]; then
    log "=== Leg 1/2: ${TRT_DIR}/.done present, skipping."
else
    log "=== Leg 1/2: TRT baseline -> ${TRT_DIR}"
    mkdir -p "${TRT_DIR}"
    NUM_RUNS="${NUM_RUNS}" MODEL="${MODEL}" ENGINE_DIR="${ENGINE_DIR}" \
        bash "${WORKDIR}/run_multirun_trt_repro.sh" "${TRT_DIR}" \
        2>&1 | tee -a "${DRIVER_LOG}"
    echo "${ENGINE_DIR}" >"${TRT_DIR}/.engine_dir"
    touch "${TRT_DIR}/.done"
fi

# ---- Leg 2: PyT feature ---------------------------------------------------

if [[ -n "${SKIP_PYT:-}" ]]; then
    log "=== Leg 2/2: SKIP_PYT=1, skipping PyT phase."
    if [[ ! -f "${PYT_DIR}/.done" ]]; then
        echo "ERROR: SKIP_PYT=1 but ${PYT_DIR}/.done is missing." >&2
        exit 1
    fi
elif [[ -f "${PYT_DIR}/.done" ]]; then
    log "=== Leg 2/2: ${PYT_DIR}/.done present, skipping."
else
    log "=== Leg 2/2: PyT feature -> ${PYT_DIR}"
    mkdir -p "${PYT_DIR}"
    NUM_RUNS="${NUM_RUNS}" MODEL="${MODEL}" \
        bash "${WORKDIR}/run_multirun_pytorch_repro.sh" "${PYT_DIR}" \
        2>&1 | tee -a "${DRIVER_LOG}"
    touch "${PYT_DIR}/.done"
fi

# ---- Aggregate (cross-backend shim) --------------------------------------
#
# aggregate_runs_repro.py picks ONE --backend tag, so we cannot directly
# compare request_trt*.json against request_pytorch*.json. Symlink the PyT
# artifacts into request_trt*.json names inside an ephemeral dir, then
# aggregate with --backend trt against the real TRT dir.

SHIM_ROOT="${OUT_ROOT}/.cross_backend_shim"
rm -rf "${SHIM_ROOT}"
mkdir -p "${SHIM_ROOT}/feature_pyt"

shim_pyt_as_trt () {
    local pyt_dir="$1" shim="$2"
    # Run 1 (unsuffixed).
    for ext in request output report; do
        local src="${pyt_dir}/${ext}_pytorch.json"
        [[ -f "${src}" ]] && ln -sf "${src}" "${shim}/${ext}_trt.json"
    done
    # Runs 2..N (suffixed).
    for src in "${pyt_dir}"/request_pytorch[0-9].json; do
        [[ -f "${src}" ]] || continue
        local base
        base=$(basename "${src}")
        local idx="${base#request_pytorch}"
        idx="${idx%.json}"
        ln -sf "${pyt_dir}/request_pytorch${idx}.json" "${shim}/request_trt${idx}.json"
        [[ -f "${pyt_dir}/output_pytorch${idx}.json" ]] && \
            ln -sf "${pyt_dir}/output_pytorch${idx}.json" "${shim}/output_trt${idx}.json"
        [[ -f "${pyt_dir}/report_pytorch${idx}.json" ]] && \
            ln -sf "${pyt_dir}/report_pytorch${idx}.json" "${shim}/report_trt${idx}.json"
    done
}

shim_pyt_as_trt "${PYT_DIR}" "${SHIM_ROOT}/feature_pyt"

CMP_MD="${OUT_ROOT}/cmp_feature_pyt_vs_baseline_trt.md"
log "=== Aggregating: feature_pyt vs baseline_trt -> ${CMP_MD}"
python3 "${WORKDIR}/aggregate_runs_repro.py" \
    --backend trt \
    --baseline   "${TRT_DIR}"             --baseline-label   "baseline_trt" \
    --experiment "${SHIM_ROOT}/feature_pyt" --experiment-label "feature_pyt" \
    > "${CMP_MD}" 2>&1

rm -rf "${SHIM_ROOT}"

log "=== Done"
log "Per-leg directories:"
log "  ${TRT_DIR}"
log "  ${PYT_DIR}"
log "Comparison report:"
log "  ${CMP_MD}"
