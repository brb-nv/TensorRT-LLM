#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SINGLE-COMMAND end-to-end session. Launches the 3ctx+2gen disagg setup,
# starts the crash-proof /perf_metrics watcher, runs the RWLT client, then
# ALWAYS tears everything down (watcher + servers) on exit, Ctrl-C, or client
# failure -- so the next session starts on a cold engine and no /perf_metrics
# data is lost even if the prefill workers crash mid-run.
#
# Usage:
#   scripts/run_session.sh [label] [rwlt_config_basename]
#     label                 output dir basename under rwlt-results/ (default: worst250)
#     rwlt_config_basename  default: rwlt_worst250
#
# Per-run logs and the final /perf_metrics drain land in rwlt-results/<label>/.
# The final drain (in stop_disagg.sh) captures all records in ONE file.
#
# Optional crash insurance: set PERF_WATCH=1 to also snapshot /perf_metrics
# periodically during the run. Off by default -- the destructive periodic
# reads fragment records across files; only worth it if the workers are
# expected to crash mid-run (then the final drain would get nothing).
#
# Env vars forwarded: MODEL, EAGLE_CKPT, SERVED_MODEL_NAME, CTX_GPUS,
# GEN_GPU_GROUPS, AA_REPO, MODEL_API_ID, SEED, CONCS,
# PERF_WATCH (default 0), PERF_WATCH_INTERVAL (default 30s).
set -uo pipefail

cd "$(dirname "$0")/.."
ROUND_ROOT="$(pwd)"

LABEL="${1:-worst250}"
RWLT_CONFIG="${2:-rwlt_worst250}"
PERF_WATCH="${PERF_WATCH:-0}"
PERF_WATCH_INTERVAL="${PERF_WATCH_INTERVAL:-30}"

RESULTS_DIR="${ROUND_ROOT}/rwlt-results/${LABEL}"
mkdir -p "${RESULTS_DIR}"
# Put server logs + drained perf metrics alongside the client results.
export LOG_DIR="${RESULTS_DIR}"
WATCH_PID=""

teardown() {
  echo
  if [[ -n "${WATCH_PID}" ]] && kill -0 "${WATCH_PID}" 2>/dev/null; then
    echo "[run_session] stopping perf_metrics watcher (PID ${WATCH_PID}) ..."
    kill "${WATCH_PID}" 2>/dev/null || true
  fi
  echo "[run_session] tearing down disagg servers ..."
  LOG_DIR="${RESULTS_DIR}" "${ROUND_ROOT}/scripts/stop_disagg.sh" || true
  echo "[run_session] teardown complete."
}
trap teardown EXIT INT TERM

echo "[run_session] launching 3ctx+2gen disagg (logs -> ${RESULTS_DIR}) ..."
"${ROUND_ROOT}/scripts/launch_disagg_3ctx2gen.sh"

if [[ "${PERF_WATCH}" == "1" ]]; then
  echo "[run_session] PERF_WATCH=1: starting periodic /perf_metrics watcher (${PERF_WATCH_INTERVAL}s) ..."
  "${ROUND_ROOT}/scripts/watch_perf_metrics.sh" "${RESULTS_DIR}" "${PERF_WATCH_INTERVAL}" \
    > "${RESULTS_DIR}/perf_watch.log" 2>&1 &
  WATCH_PID=$!
  echo "[run_session] watcher PID ${WATCH_PID}"
else
  echo "[run_session] perf_metrics watcher disabled (single final drain at teardown). Set PERF_WATCH=1 to enable."
fi

echo "[run_session] running RWLT client (label=${LABEL}, config=${RWLT_CONFIG}) ..."
"${ROUND_ROOT}/scripts/run_rwlt.sh" "${LABEL}" "http://localhost:8000/v1" "${RWLT_CONFIG}"

echo "[run_session] client finished. (watcher stop + drain + teardown run on exit)"
