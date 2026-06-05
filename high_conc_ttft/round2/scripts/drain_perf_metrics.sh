#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Snapshot the disagg proxy /perf_metrics endpoint to disk. Reading
# /perf_metrics is DESTRUCTIVE (it pops the per-worker deques while the proxy
# joins them), so this must run exactly once per session, after the client
# finishes and before teardown. The proxy drain pulls ctx + gen worker
# metrics as a side effect, so we only hit the proxy (one source of truth).
#
# Usage: scripts/drain_perf_metrics.sh [out_dir]
# out_dir resolution: arg -> DRAIN_OUT_DIR -> LOG_DIR -> logs/pids/last_log_dir
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROUND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="${ROUND_ROOT}/logs/pids"
LAST_LOG_DIR_FILE="${PID_DIR}/last_log_dir"

OUT_DIR="${1:-}"
if [[ -z "${OUT_DIR}" ]]; then
  if [[ -n "${DRAIN_OUT_DIR:-}" ]]; then OUT_DIR="${DRAIN_OUT_DIR}"
  elif [[ -n "${LOG_DIR:-}" ]]; then OUT_DIR="${LOG_DIR}"
  elif [[ -f "${LAST_LOG_DIR_FILE}" ]]; then
    OUT_DIR="$(cat "${LAST_LOG_DIR_FILE}")"
    echo "[drain] inheriting OUT_DIR=${OUT_DIR} from ${LAST_LOG_DIR_FILE}" >&2
  else
    OUT_DIR="${ROUND_ROOT}/logs"
    echo "WARNING: no out_dir/DRAIN_OUT_DIR/LOG_DIR/last_log_dir; using ${OUT_DIR}" >&2
  fi
fi
mkdir -p "${OUT_DIR}"

out_path="${OUT_DIR}/perf_metrics_proxy.json"
if ! curl -sS --max-time 180 "http://localhost:8000/perf_metrics" -o "${out_path}"; then
  echo "[drain] WARNING: proxy /perf_metrics curl failed" >&2
  echo "[]" > "${out_path}"
fi
[[ -s "${out_path}" ]] || echo "[]" > "${out_path}"
n=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d,list) else 1)' "${out_path}" 2>/dev/null || echo "?")
echo "[drain] proxy(8000): ${n} records -> ${out_path}"
