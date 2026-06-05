#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch a 3ctx (TP1) + 2gen (TP2) disaggregated trtllm-serve setup for
# gpt-oss-120b, reproducing the topology of NVBug 6266370 on a single 8-GPU
# B200 node. Spawns 6 background processes, each with its own log:
#
#   ctx0  CUDA_VISIBLE_DEVICES=0      port 8001  CONTEXT     (TP1)
#   ctx1  CUDA_VISIBLE_DEVICES=1      port 8002  CONTEXT     (TP1)
#   ctx2  CUDA_VISIBLE_DEVICES=2      port 8003  CONTEXT     (TP1)
#   gen0  CUDA_VISIBLE_DEVICES=3,4    port 8004  GENERATION  (TP2)
#   gen1  CUDA_VISIBLE_DEVICES=5,6    port 8005  GENERATION  (TP2)
#   proxy                            port 8000  (once all 5 workers healthy)
#   (GPU 7 is left idle by default.)
#
# Env vars (override before invoking):
#   MODEL              HF id or local path of the model checkpoint
#   EAGLE_CKPT         path/id of the Eagle3 draft checkpoint
#   SERVED_MODEL_NAME  stable API id the RWLT client targets
#   CTX_GPUS           space-separated GPU indices for ctx workers ("0 1 2")
#   GEN_GPU_GROUPS     space-separated CUDA_VISIBLE_DEVICES groups for gen
#                      workers ("3,4 5,6")
#   CTX_CONFIG_BASE / GEN_CONFIG_BASE / PROXY_CONFIG_BASE   config basenames
#   LOG_DIR            per-run log/output dir (default: round1/logs)
#
# Stop everything with scripts/stop_disagg.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
ROUND_ROOT="$(pwd)"

MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/gpt_oss/gpt-oss-120b}"
# Bug used nvidia/gpt-oss-120b-Eagle3-next; local released-name equivalent:
EAGLE_CKPT="${EAGLE_CKPT:-/home/scratch.simengl_sw_3/trt_repos/hf_models/nvidia/gpt-oss-120b-Eagle3-v3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-openai/gpt-oss-120b}"

read -r -a CTX_GPUS <<< "${CTX_GPUS:-0 1 2}"
read -r -a GEN_GPU_GROUPS <<< "${GEN_GPU_GROUPS:-3,4 5,6}"
CTX_PORTS=(8001 8002 8003)
GEN_PORTS=(8004 8005)

CTX_CONFIG_BASE="${CTX_CONFIG_BASE:-ctx_tp1}"
GEN_CONFIG_BASE="${GEN_CONFIG_BASE:-gen_tp2}"
PROXY_CONFIG_BASE="${PROXY_CONFIG_BASE:-proxy_3ctx2gen}"
CONFIG_DIR="${CONFIG_DIR:-configs}"
LOG_DIR="${LOG_DIR:-${ROUND_ROOT}/logs}"
PID_DIR="${LOG_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Persist LOG_DIR so stop_disagg.sh / drain_perf_metrics.sh default to the
# same per-run directory without re-passing the env var on teardown.
echo "${LOG_DIR}" > "${PID_DIR}/last_log_dir"

# Prevent any HF download attempt from workers / proxy.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Base dir for cache-transceiver KV-transfer timing CSVs. Each worker gets its
# OWN subdir (set inline at launch) so the per-rank send/recv CSVs don't collide
# across the 3 ctx + 2 gen workers (they all write rank_0_*.csv otherwise).
KVT_BASE="${KVT_BASE:-${LOG_DIR}/disagg_kvcache_time}"
mkdir -p "${KVT_BASE}"

if [[ ${#CTX_GPUS[@]} -ne 3 ]]; then
  echo "ERROR: expected 3 CTX_GPUS, got ${#CTX_GPUS[@]} (${CTX_GPUS[*]})" >&2; exit 2
fi
if [[ ${#GEN_GPU_GROUPS[@]} -ne 2 ]]; then
  echo "ERROR: expected 2 GEN_GPU_GROUPS, got ${#GEN_GPU_GROUPS[@]} (${GEN_GPU_GROUPS[*]})" >&2; exit 2
fi

# Fail-fast on stale servers. If a previous run's proxy/workers still hold
# 8000-8005, trtllm-serve's bind fails BUT wait_for_health would pass against
# the stale process -- silently running the client (and /perf_metrics) against
# the OLD deployment. Abort loudly so the user tears down first.
STALE=()
for p in 8000 8001 8002 8003 8004 8005; do
  if curl -sf -o /dev/null --max-time 2 "http://localhost:${p}/health" 2>/dev/null; then
    STALE+=("${p}")
  fi
done
if [[ ${#STALE[@]} -gt 0 ]]; then
  echo "ERROR: ports already serving (stale run?): ${STALE[*]}" >&2
  echo "       Tear down first:  scripts/stop_disagg.sh   (then re-check with" >&2
  echo "       'pkill -f trtllm-serve' and nvidia-smi if needed), then relaunch." >&2
  exit 3
fi

resolve_config() {
  local base="$1"
  local src="${ROUND_ROOT}/${CONFIG_DIR}/${base}.yaml"
  local out="${LOG_DIR}/.runtime_${base}.yaml"
  if [[ ! -f "${src}" ]]; then
    echo "ERROR: ${src} not found" >&2
    return 1
  fi
  sed "s|EAGLE_CKPT_PLACEHOLDER|${EAGLE_CKPT}|g" "${src}" > "${out}"
  echo "${out}"
}

wait_for_health() {
  local url="$1" label="$2" timeout="${3:-1800}" start
  start=$(date +%s)
  echo "Waiting for ${label} (${url}) ..."
  while true; do
    if curl -sf -o /dev/null "${url}"; then
      echo "${label} ready after $(( $(date +%s) - start ))s"; return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      echo "ERROR: ${label} not healthy within ${timeout}s" >&2; return 1
    fi
    sleep 5
  done
}

CTX_RUN_CONFIG="$(resolve_config "${CTX_CONFIG_BASE}")"
GEN_RUN_CONFIG="$(resolve_config "${GEN_CONFIG_BASE}")"
PROXY_CONFIG="${ROUND_ROOT}/${CONFIG_DIR}/${PROXY_CONFIG_BASE}.yaml"

echo "model      -> ${MODEL}"
echo "eagle ckpt -> ${EAGLE_CKPT}"
echo "ctx config -> ${CTX_RUN_CONFIG} (GPUs ${CTX_GPUS[*]})"
echo "gen config -> ${GEN_RUN_CONFIG} (GPU groups ${GEN_GPU_GROUPS[*]})"
echo "proxy      -> ${PROXY_CONFIG}"
echo "log dir    -> ${LOG_DIR}"

# --- context workers (TP1) ---
for i in 0 1 2; do
  gpu="${CTX_GPUS[$i]}"; port="${CTX_PORTS[$i]}"
  kvt="${KVT_BASE}/ctx${i}"; mkdir -p "${kvt}"
  CUDA_VISIBLE_DEVICES="${gpu}" TRTLLM_KVCACHE_TIME_OUTPUT_PATH="${kvt}" \
    nohup trtllm-serve "${MODEL}" \
    --host localhost --port "${port}" --backend pytorch --server_role CONTEXT \
    --served_model_name "${SERVED_MODEL_NAME}" \
    --extra_llm_api_options "${CTX_RUN_CONFIG}" \
    > "${LOG_DIR}/ctx${i}.log" 2>&1 &
  echo $! > "${PID_DIR}/ctx${i}.pid"
  echo "ctx${i} PID $(cat "${PID_DIR}/ctx${i}.pid") (GPU ${gpu}, port ${port}, kvt ${kvt})"
done

# --- generation workers (TP2) ---
for i in 0 1; do
  grp="${GEN_GPU_GROUPS[$i]}"; port="${GEN_PORTS[$i]}"
  kvt="${KVT_BASE}/gen${i}"; mkdir -p "${kvt}"
  CUDA_VISIBLE_DEVICES="${grp}" TRTLLM_KVCACHE_TIME_OUTPUT_PATH="${kvt}" \
    nohup trtllm-serve "${MODEL}" \
    --host localhost --port "${port}" --backend pytorch --server_role GENERATION \
    --served_model_name "${SERVED_MODEL_NAME}" \
    --extra_llm_api_options "${GEN_RUN_CONFIG}" \
    > "${LOG_DIR}/gen${i}.log" 2>&1 &
  echo $! > "${PID_DIR}/gen${i}.pid"
  echo "gen${i} PID $(cat "${PID_DIR}/gen${i}.pid") (GPUs ${grp}, port ${port}, kvt ${kvt})"
done

for i in 0 1 2; do wait_for_health "http://localhost:${CTX_PORTS[$i]}/health" "ctx${i}"; done
for i in 0 1; do wait_for_health "http://localhost:${GEN_PORTS[$i]}/health" "gen${i}"; done

nohup trtllm-serve disaggregated -c "${PROXY_CONFIG}" \
  > "${LOG_DIR}/proxy.log" 2>&1 &
echo $! > "${PID_DIR}/proxy.pid"
echo "proxy PID $(cat "${PID_DIR}/proxy.pid")"

wait_for_health "http://localhost:8000/health" "disagg proxy"
echo "All 5 workers + proxy healthy. Disagg endpoint -> http://localhost:8000"
