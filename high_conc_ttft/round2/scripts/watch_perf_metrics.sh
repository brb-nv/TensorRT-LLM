#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Crash-proof /perf_metrics capture. The disagg proxy /perf_metrics read is
# DESTRUCTIVE (pops the worker deques), so periodically snapshotting it both
# (a) bounds memory growth and (b) guarantees we keep the pre-crash data even
# if the prefill workers die mid-run. Each snapshot holds the requests that
# completed since the previous snapshot; merge them in analysis.
#
# RESILIENCE: under high concurrency the proxy event loop can briefly block
# (the very bottleneck we're studying), so a single /health miss or curl
# timeout does NOT mean the proxy is dead. We therefore (1) probe the real
# endpoint (/perf_metrics) directly rather than gating on /health, (2) keep a
# generous timeout, and (3) only stop after MAX_FAILS *consecutive* failures
# (so transient stalls are tolerated; the watcher exits only if the proxy is
# truly gone for ~MAX_FAILS*INTERVAL seconds).
#
# Usage:
#   scripts/watch_perf_metrics.sh <out_dir> [interval_seconds]
# Env:
#   PERF_WATCH_TIMEOUT  per-request curl --max-time for /perf_metrics (def 90)
#   PERF_WATCH_MAX_FAILS consecutive failures before giving up (def 10)
set -uo pipefail

OUT_DIR="${1:?usage: watch_perf_metrics.sh <out_dir> [interval_seconds]}"
INTERVAL="${2:-30}"
TIMEOUT="${PERF_WATCH_TIMEOUT:-90}"
MAX_FAILS="${PERF_WATCH_MAX_FAILS:-10}"
SNAP_DIR="${OUT_DIR}/perf_snapshots"
mkdir -p "${SNAP_DIR}"

echo "[watch] snapshotting /perf_metrics every ${INTERVAL}s -> ${SNAP_DIR}/ " \
     "(timeout=${TIMEOUT}s, give up after ${MAX_FAILS} consecutive failures)"
i=0
fails=0
while true; do
  ts="$(date +%Y%m%d_%H%M%S)"
  out="${SNAP_DIR}/snap_$(printf '%04d' "${i}")_${ts}.json"
  # Probe the real endpoint directly. curl exit codes: 0 ok, 7 connection
  # refused (proxy likely down), 28 timeout (proxy busy -> tolerate).
  curl -sS --max-time "${TIMEOUT}" "http://localhost:8000/perf_metrics" -o "${out}" 2>/dev/null
  rc=$?
  if [[ ${rc} -eq 0 ]]; then
    [[ -s "${out}" ]] || echo "[]" > "${out}"
    n=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d) if isinstance(d,list) else 1)' "${out}" 2>/dev/null || echo "?")
    echo "[watch] snap ${i}: ${n} records -> ${out}"
    fails=0
  else
    rm -f "${out}"  # don't leave a truncated/partial file
    fails=$((fails + 1))
    case ${rc} in
      7)  reason="connection refused" ;;
      28) reason="timeout (proxy busy)" ;;
      *)  reason="curl rc=${rc}" ;;
    esac
    echo "[watch] snap ${i}: FAILED (${reason}); consecutive=${fails}/${MAX_FAILS}"
    if [[ ${fails} -ge ${MAX_FAILS} ]]; then
      echo "[watch] proxy unreachable for ${fails} consecutive probes -- stopping watcher"
      break
    fi
  fi
  i=$((i + 1))
  sleep "${INTERVAL}"
done
echo "[watch] done after ${i} probes."
