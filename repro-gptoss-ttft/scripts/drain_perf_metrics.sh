#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Snapshot the trtllm-serve /perf_metrics endpoint(s) to disk. Reading
# /perf_metrics is destructive (pops the worker-side deque, and the proxy
# also drains ctx/gen worker deques while joining), so this must run exactly
# once per session, after the client finishes and before teardown.
#
# Usage:
#   scripts/drain_perf_metrics.sh agg     [out_dir]
#   scripts/drain_perf_metrics.sh disagg  [out_dir]
#
# If <out_dir> is omitted, falls back (in order) to:
#   - DRAIN_OUT_DIR env var
#   - LOG_DIR env var
#   - logs/pids/last_log_dir (persisted by launch_disagg.sh / launch_agg.sh)
#   - the literal logs/ directory next to scripts/ (with a loud warning)
# This mirrors the resolution stop_disagg.sh uses so a bare invocation does
# the right thing for the most recent run.
#
# For agg, writes:   <out_dir>/perf_metrics.json
# For disagg, writes:
#   <out_dir>/perf_metrics_proxy.json   (joined ctx+gen view, primary input
#                                        for perf_metrics_breakdown.py)
#
# Note: in disagg mode the proxy's drain pulls from ctx and gen workers, so
# hitting the worker /perf_metrics endpoints AFTER the proxy would always
# return [] -- we deliberately skip them to keep one source of truth.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_DIR="${REPO_ROOT}/logs/pids"
LAST_LOG_DIR_FILE="${PID_DIR}/last_log_dir"

LAYOUT="${1:?usage: drain_perf_metrics.sh <agg|disagg> [out_dir]}"
OUT_DIR="${2:-}"

if [[ -z "${OUT_DIR}" ]]; then
  if [[ -n "${DRAIN_OUT_DIR:-}" ]]; then
    OUT_DIR="${DRAIN_OUT_DIR}"
  elif [[ -n "${LOG_DIR:-}" ]]; then
    OUT_DIR="${LOG_DIR}"
  elif [[ -f "${LAST_LOG_DIR_FILE}" ]]; then
    OUT_DIR="$(cat "${LAST_LOG_DIR_FILE}")"
    echo "[drain_perf_metrics] inheriting OUT_DIR=${OUT_DIR} from ${LAST_LOG_DIR_FILE}" >&2
  else
    OUT_DIR="${REPO_ROOT}/logs"
    echo "WARNING: drain_perf_metrics.sh has no out_dir / DRAIN_OUT_DIR / LOG_DIR / last_log_dir; falling back to ${OUT_DIR}" >&2
  fi
fi

mkdir -p "${OUT_DIR}"

drain_one() {
  local url="$1"
  local out_path="$2"
  local label="$3"
  # Long timeout: proxy may iterate over ctx/gen /perf_metrics serially.
  if ! curl -sS --max-time 120 "${url}" -o "${out_path}"; then
    echo "[drain_perf_metrics] WARNING: ${label} curl failed (${url})" >&2
    echo "[]" > "${out_path}"
    return 1
  fi
  if [[ ! -s "${out_path}" ]]; then
    echo "[]" > "${out_path}"
  fi
  local n
  n=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d, list) else 1)' "${out_path}" 2>/dev/null || echo "?")
  echo "[drain_perf_metrics] ${label}: ${n} records -> ${out_path}"
}

case "${LAYOUT}" in
  agg)
    drain_one "http://localhost:8000/perf_metrics" "${OUT_DIR}/perf_metrics.json" "agg(8000)"
    ;;
  disagg)
    # Proxy /perf_metrics drains both worker /perf_metrics endpoints as a
    # side-effect of joining them. Drain proxy only -- the joined output
    # already contains ctx_perf_metrics and gen_perf_metrics nested.
    drain_one "http://localhost:8000/perf_metrics" "${OUT_DIR}/perf_metrics_proxy.json" "proxy(8000)"
    ;;
  *)
    echo "ERROR: layout must be 'agg' or 'disagg', got ${LAYOUT}" >&2
    exit 2
    ;;
esac
