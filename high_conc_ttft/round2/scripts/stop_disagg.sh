#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop the 3ctx+2gen disagg trtllm-serve processes started by
# launch_disagg_3ctx2gen.sh, draining /perf_metrics first.
#
# Teardown order:
#   1. drain /perf_metrics (once, before killing)
#   2. kill by pidfile (per-run LOG_DIR/pids, falling back to logs/pids)
#   3. fallback: if ports 8000-8005 are STILL bound, pkill -f trtllm-serve
#      (this is what prevents the stale-server-holds-the-port trap; the
#      pidfile PIDs can be unkillable from a different PID namespace or stale).
#      Disable the broad fallback with KILL_BY_NAME=0 if other unrelated
#      trtllm-serve processes must be preserved on this node.
set -uo pipefail

cd "$(dirname "$0")/.."
ROUND_ROOT="$(pwd)"

# Resolve the per-run pid dir. A bare `stop_disagg.sh` must still find the
# pidfiles a run wrote under rwlt-results/<label>/pids, so prefer LOG_DIR
# (explicit or inherited) over the default logs/pids.
DEFAULT_PID_DIR="${ROUND_ROOT}/logs/pids"
LAST_LOG_DIR_FILE="${DEFAULT_PID_DIR}/last_log_dir"
if [[ -z "${LOG_DIR:-}" && -f "${LAST_LOG_DIR_FILE}" ]]; then
  LOG_DIR="$(cat "${LAST_LOG_DIR_FILE}")"
  echo "[stop] inheriting LOG_DIR=${LOG_DIR} from ${LAST_LOG_DIR_FILE}"
fi
PID_DIR="${LOG_DIR:+${LOG_DIR}/pids}"
PID_DIR="${PID_DIR:-${DEFAULT_PID_DIR}}"
DRAIN_OUT_DIR="${DRAIN_OUT_DIR:-${LOG_DIR:-${ROUND_ROOT}/logs}}"

PORTS=(8000 8001 8002 8003 8004 8005)

port_in_use() {  # 0 = something is listening on $1
  (exec 3<>"/dev/tcp/localhost/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; } || return 1
}

# 1. Drain /perf_metrics (idempotent /health probe; drain exactly once).
if curl -sf -o /dev/null --max-time 2 "http://localhost:8000/health" >/dev/null 2>&1; then
  echo "[stop] draining /perf_metrics -> ${DRAIN_OUT_DIR}/perf_metrics_proxy.json"
  bash "${ROUND_ROOT}/scripts/drain_perf_metrics.sh" "${DRAIN_OUT_DIR}" || \
    echo "[stop] WARNING: drain_perf_metrics.sh exited non-zero" >&2
else
  echo "[stop] proxy :8000 /health unreachable -- skipping drain"
fi

# 2. Kill by pidfile (proxy first, then gen, then ctx).
if [[ -d "${PID_DIR}" ]]; then
  for pidfile in "${PID_DIR}/proxy.pid" "${PID_DIR}"/gen*.pid "${PID_DIR}"/ctx*.pid; do
    [[ -f "${pidfile}" ]] || continue
    name="$(basename "${pidfile}" .pid)"
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] killing ${name} (PID ${pid})"
      kill "${pid}" || true
      for _ in $(seq 1 15); do kill -0 "${pid}" 2>/dev/null && sleep 1 || break; done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "[stop] force-killing ${name} (PID ${pid})"; kill -9 "${pid}" || true
      fi
    else
      echo "[stop] ${name}: PID ${pid} not running (from this namespace)"
    fi
    rm -f "${pidfile}"
  done
else
  echo "[stop] no pid dir at ${PID_DIR}"
fi

# 3. Fallback: if any disagg port is still bound, the pidfile kill missed it
# (stale/orphaned/cross-namespace). Kill by process name.
still=()
for p in "${PORTS[@]}"; do port_in_use "${p}" && still+=("${p}"); done
if [[ ${#still[@]} -gt 0 ]]; then
  if [[ "${KILL_BY_NAME:-1}" == "1" ]]; then
    echo "[stop] ports still bound after pidfile kill: ${still[*]}"
    echo "[stop] fallback: pkill -9 -f trtllm-serve"
    pkill -9 -f "trtllm-serve" 2>/dev/null || true
    sleep 3
    still=()
    for p in "${PORTS[@]}"; do port_in_use "${p}" && still+=("${p}"); done
    if [[ ${#still[@]} -gt 0 ]]; then
      echo "[stop] WARNING: ports STILL bound: ${still[*]} -- manual cleanup needed" >&2
    else
      echo "[stop] all ports free after name-based kill"
    fi
  else
    echo "[stop] WARNING: ports still bound (${still[*]}) and KILL_BY_NAME=0 -- not killing by name" >&2
  fi
else
  echo "[stop] all disagg ports free"
fi
echo "[stop] done."
