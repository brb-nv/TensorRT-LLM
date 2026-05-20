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
# Build the TinyLlama-1.1B-Chat-v1.0 TensorRT engine consumed by the
# baseline_trt leg of the NVBug 5615248 TRT-vs-PyT repro.
#
# Geometry pinned to match pytorch_repro.yaml so the two backends do
# apples-to-apples work on the same workload:
#   - max_seq_len    = 129
#   - max_input_len  = 129
#   - max_num_tokens = 129
#   - max_batch_size = 1
#   - max_beam_width = 10
#   - paged KV cache + paged context FMHA enabled
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/build_trt_engine_repro.sh
#
# Optional env vars:
#   HF_MODEL    default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
#               (HF repo id or local path containing the HF model weights)
#   DTYPE       default: bfloat16
#   CKPT_DIR    default: nvbugs_5615248/tinyllama_trt_ckpt
#   ENGINE_DIR  default: nvbugs_5615248/tinyllama_trt_engine

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_MODEL="${HF_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DTYPE="${DTYPE:-bfloat16}"
CKPT_DIR="${CKPT_DIR:-${REPO_DIR}/nvbugs_5615248/tinyllama_trt_ckpt}"
ENGINE_DIR="${ENGINE_DIR:-${REPO_DIR}/nvbugs_5615248/tinyllama_trt_engine}"

CONVERT_PY="${REPO_DIR}/examples/models/core/llama/convert_checkpoint.py"

if [[ ! -f "${CONVERT_PY}" ]]; then
    echo "ERROR: missing ${CONVERT_PY}." >&2
    exit 1
fi
if ! command -v trtllm-build >/dev/null 2>&1; then
    echo "ERROR: trtllm-build not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

echo "[build] HF_MODEL=${HF_MODEL}"
echo "[build] DTYPE=${DTYPE}"
echo "[build] CKPT_DIR=${CKPT_DIR}"
echo "[build] ENGINE_DIR=${ENGINE_DIR}"

# Step 1: HF -> TRT-LLM checkpoint conversion.
if [[ -f "${CKPT_DIR}/config.json" ]]; then
    echo "[build] ${CKPT_DIR}/config.json present, skipping checkpoint conversion."
else
    echo "[build] Converting HF -> TRT-LLM checkpoint..."
    mkdir -p "${CKPT_DIR}"
    python3 "${CONVERT_PY}" \
        --model_dir "${HF_MODEL}" \
        --output_dir "${CKPT_DIR}" \
        --dtype "${DTYPE}"
fi

# Step 2: TRT-LLM checkpoint -> TRT engine, geometry matched to pytorch_repro.yaml.
if [[ -f "${ENGINE_DIR}/rank0.engine" ]]; then
    echo "[build] ${ENGINE_DIR}/rank0.engine present, skipping engine build."
else
    echo "[build] Building TRT engine..."
    mkdir -p "${ENGINE_DIR}"
    trtllm-build \
        --checkpoint_dir "${CKPT_DIR}" \
        --output_dir "${ENGINE_DIR}" \
        --max_seq_len 129 \
        --max_input_len 129 \
        --max_num_tokens 129 \
        --max_batch_size 1 \
        --max_beam_width 10 \
        --gpt_attention_plugin auto \
        --paged_kv_cache enable \
        --use_paged_context_fmha enable
fi

echo
echo "[build] Done."
echo "[build] Engine: ${ENGINE_DIR}/rank0.engine"
echo "[build] Now run:"
echo "  bash nvbugs_5615248/trtllm_bench/run_trt_vs_pyt_compare_repro.sh"
