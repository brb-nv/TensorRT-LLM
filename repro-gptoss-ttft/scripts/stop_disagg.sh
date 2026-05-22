#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop the disagg trtllm-serve processes started by launch_disagg.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
REPRO_DIR="$(pwd)"
PID_DIR="logs/pids"

# Drain /perf_metrics BEFORE killing the proxy/workers. Reading the endpoint
# is destructive (pops the worker deques and joins them), so this must run
# exactly once per session before teardown. Without this, perf_metrics_proxy.
# json is never written for the run, even if the rwlt client ran successfully.
#
# Output directory resolution (first match wins):
#   1. DRAIN_OUT_DIR env var (explicit override; rarely needed)
#   2. LOG_DIR env var (caller passes same LOG_DIR they used for launch)
#   3. logs/pids/last_log_dir (persisted by launch_disagg.sh -- AUTO; this
#      is what makes the common case "just bash stop_disagg.sh" do the right
#      thing without needing to re-set LOG_DIR)
#   4. logs/ fallback (last-resort, also warned about loudly)
LAST_LOG_DIR_FILE="${PID_DIR}/last_log_dir"
if [[ -z "${DRAIN_OUT_DIR:-}" && -z "${LOG_DIR:-}" && -f "${LAST_LOG_DIR_FILE}" ]]; then
  LOG_DIR="$(cat "${LAST_LOG_DIR_FILE}")"
  echo "[stop_disagg] inheriting LOG_DIR=${LOG_DIR} from ${LAST_LOG_DIR_FILE}"
fi
DRAIN_OUT_DIR="${DRAIN_OUT_DIR:-${LOG_DIR:-${REPRO_DIR}/logs}}"
if [[ "${DRAIN_OUT_DIR}" == "${REPRO_DIR}/logs" || "${DRAIN_OUT_DIR}" == "logs" ]]; then
  echo "WARNING: DRAIN_OUT_DIR resolved to the default '${DRAIN_OUT_DIR}'. " \
       "If this run used a per-run LOG_DIR (e.g. rwlt-results/<run>/), the " \
       "drained perf_metrics will land in the wrong place. Pass LOG_DIR=... " \
       "or DRAIN_OUT_DIR=... to stop_disagg.sh, or rely on the auto-inherit " \
       "via logs/pids/last_log_dir (written by launch_disagg.sh)." >&2
fi
# CRITICAL: /perf_metrics is destructive (reading it pops the deque), so we
# MUST NOT hit it as a liveness probe -- doing so silently discards the
# session's records. Use /health (idempotent) for the liveness check, and
# call /perf_metrics exactly once via drain_perf_metrics.sh.
if curl -sf -o /dev/null --max-time 2 "http://localhost:8000/health" >/dev/null 2>&1; then
  echo "Draining /perf_metrics -> ${DRAIN_OUT_DIR}/perf_metrics_proxy.json"
  bash "${REPRO_DIR}/scripts/drain_perf_metrics.sh" disagg "${DRAIN_OUT_DIR}" || \
    echo "WARNING: drain_perf_metrics.sh exited non-zero (perf data may be empty)" >&2
else
  echo "Proxy at :8000 /health unreachable -- skipping /perf_metrics drain"
fi

for name in proxy gen ctx; do
  pidfile="${PID_DIR}/${name}.pid"
  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Killing ${name} (PID ${pid})"
      kill "${pid}" || true
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        if kill -0 "${pid}" 2>/dev/null; then sleep 1; else break; fi
      done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "Force-killing ${name} (PID ${pid})"
        kill -9 "${pid}" || true
      fi
    else
      echo "${name}: PID ${pid} not running"
    fi
    rm -f "${pidfile}"
  else
    echo "${name}: no pidfile (already stopped?)"
  fi
done
