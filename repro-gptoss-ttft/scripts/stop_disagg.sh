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
# Output directory:
#   - DRAIN_OUT_DIR env var (preferred; set by callers that use per-run dirs)
#   - otherwise LOG_DIR from launch_disagg.sh (defaults to logs/)
#   - otherwise the literal "logs/" directory next to scripts/
DRAIN_OUT_DIR="${DRAIN_OUT_DIR:-${LOG_DIR:-${REPRO_DIR}/logs}}"
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
