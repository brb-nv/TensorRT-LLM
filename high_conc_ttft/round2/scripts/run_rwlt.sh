#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the RWLT client against an already-running disagg trtllm-serve proxy on
# the same node (no sflow / sbatch / srun).
#
# Usage:
#   scripts/run_rwlt.sh <label> [base_url] [rwlt_config_basename]
#     label                output dir name under rwlt-results/ (e.g. c160)
#     base_url             default http://localhost:8000/v1
#     rwlt_config_basename default rwlt_c160
#
# Env vars:
#   AA_REPO       path to the artificial-analysis repo (has the rwlt package)
#   MODEL_API_ID  request body "model" field; default openai/gpt-oss-120b
#                 (must match trtllm-serve --served_model_name)
#   SEED          override seed in the RWLT config
#   CONCS         override concurrencies, e.g. "160" or "80,160"
set -euo pipefail

cd "$(dirname "$0")/.."
ROUND_ROOT="$(pwd)"

LABEL="${1:?usage: run_rwlt.sh <label> [base_url] [rwlt_config_basename]}"
BASE_URL="${2:-http://localhost:8000/v1}"
CONFIG_NAME="${3:-rwlt_c160}"

AA_REPO="${AA_REPO:-/home/scratch.bbuddharaju_gpu/artificial-analysis}"
MODEL_API_ID="${MODEL_API_ID:-openai/gpt-oss-120b}"

SRC_CONFIG="${ROUND_ROOT}/configs/${CONFIG_NAME}.yaml"
RESULTS_DIR="${ROUND_ROOT}/rwlt-results/${LABEL}"
mkdir -p "${RESULTS_DIR}"

[[ -f "${SRC_CONFIG}" ]] || { echo "ERROR: ${SRC_CONFIG} not found" >&2; exit 1; }

EXTRA_ARGS=()
[[ -n "${SEED:-}" ]] && EXTRA_ARGS+=(--seed "${SEED}")
[[ -n "${CONCS:-}" ]] && EXTRA_ARGS+=(--concurrencies "${CONCS}")

echo "RWLT label   : ${LABEL}"
echo "AA_REPO      : ${AA_REPO}"
echo "base_url     : ${BASE_URL}"
echo "model api id : ${MODEL_API_ID}"
echo "rwlt config  : ${SRC_CONFIG}"
echo "results dir  : ${RESULTS_DIR}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "extra args   : ${EXTRA_ARGS[*]}"

# `uv run` resolves the rwlt pyproject deps into ${AA_REPO}/.venv on first call.
cd "${AA_REPO}"
uv run --project "${AA_REPO}" -- \
  python3 -m rwlt.run \
    --config "${SRC_CONFIG}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_API_ID}" \
    --api-key not-required \
    --results-dir "${RESULTS_DIR}" \
    --request-log-path "${RESULTS_DIR}/rwlt_requests.jsonl" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/${LABEL}.rwlt.log"

echo "Wrote:"
ls -la "${RESULTS_DIR}"
