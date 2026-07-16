#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Steady-state nsys profile of MiniMax-M3-NVFP4 for ISL=8192 / OSL=1 at a
# single concurrency, via trtllm-bench (PyTorch backend).
#
# Only the steady-state engine iterations are recorded: trtllm-bench runs
# --warmup requests first, then the CUDA profiler is toggled ON/OFF around a
# window of engine iterations (TLLM_PROFILE_START_STOP=A-B) that nsys records
# via `-c cudaProfilerApi`. Everything outside [A, B] -- model load, CUDA-graph
# capture, warmup, and the ramp/tail -- is excluded, so the trace is small and
# contains only steady-state prefill+decode work.
#
# Multi-GPU note: MiniMax-M3-NVFP4 needs tp=ep=4 (>=4 Blackwell GPUs, >=140 GB
# each). trtllm-bench self-spawns the tp workers; `--trace-fork-before-exec`
# makes nsys follow them into one combined trace. This mirrors the documented
# multi-GPU recipe in docs/source/developer-guide/perf-analysis.md.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash minimax_m3_fp4_nsys/run_nsys.sh [CONCURRENCY] [OUT_DIR]
#
# Args:
#   CONCURRENCY   in-flight request cap (default 1).
#   OUT_DIR       output dir (default minimax_m3_fp4_nsys/out/c<CONCURRENCY>).
#
# Common env overrides:
#   MODEL           HF model dir (default MiniMax-M3-NVFP4 under llm-models).
#   TP / EP         tensor / expert parallel size (default 4 / 4).
#   ISL / OSL       input / output length (default 8192 / 1).
#   MAX_SEQ_LEN     engine max seq len (default ISL+128).
#   MAX_BATCH_SIZE  engine max batch size (default 32).
#   MAX_NUM_TOKENS  engine max num tokens (default 8192).
#   WARMUP          trtllm-bench warmup requests (default 5).
#   NUM_REQUESTS    request supply (default 32; must exceed the profile
#                   window's stop iteration). With stop-shutdown the run is torn
#                   down at the window boundary, so only the first several are
#                   actually processed -- this is just enough supply to get
#                   there.
#   PROFILE_ITERS   steady-state engine-iteration window "A-B" recorded by
#                   nsys (default 12-24).
#   CAPTURE_END     nsys --capture-range-end (default "stop-shutdown": tear the
#                   run down as soon as the window closes, so it never runs the
#                   full request set and never risks a post-window stall). Use
#                   "stop" to keep running after the window (slower, riskier).
#   CUDA_GRAPH_TRACE  nsys --cuda-graph-trace value (default unset = don't pass
#                   the flag). Set to "node" to expand kernels inside CUDA
#                   graphs -- WARNING: CUPTI node tracing can deadlock with
#                   NCCL-in-graph (MoE EP all-to-all); leave unset unless needed.

set -euo pipefail

CONCURRENCY="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${2:-${SCRIPT_DIR}/out/c${CONCURRENCY}}"

MODEL="${MODEL:-/home/scratch.trt_llm_data_ci/llm-models/MiniMax-M3-NVFP4}"
TP="${TP:-4}"
EP="${EP:-4}"
ISL="${ISL:-8192}"
OSL="${OSL:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$((ISL + 128))}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
WARMUP="${WARMUP:-5}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
PROFILE_ITERS="${PROFILE_ITERS:-12-24}"
CAPTURE_END="${CAPTURE_END:-stop-shutdown}"
CUDA_GRAPH_TRACE="${CUDA_GRAPH_TRACE:-}"
# nsys --wait: "primary" finalizes the report when the trtllm-bench process
# exits WITHOUT blocking on the re-parented MPI worker processes (the ranks are
# self-spawned children). The profiled data is already flushed at the
# cudaProfilerStop boundary, so this avoids a teardown hang on multi-GPU runs.
NSYS_WAIT="${NSYS_WAIT:-primary}"

CONFIG="${SCRIPT_DIR}/config.yaml"
DATASET="${DATASET:-${SCRIPT_DIR}/dataset_isl${ISL}_osl${OSL}.jsonl}"

# ---- Pre-flight ----------------------------------------------------------
for tool in nsys trtllm-bench python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "ERROR: ${tool} not on PATH (are you inside the TRT-LLM container?)." >&2
        exit 1
    fi
done
if [[ ! -d "${MODEL}" ]]; then
    echo "ERROR: MODEL='${MODEL}' is not a local directory." >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found at ${CONFIG}." >&2
    exit 1
fi

# The profile window must sit inside the run: iterations advance roughly one
# per ISL=MAX_NUM_TOKENS prefill, so #requests must exceed the window stop.
PROFILE_STOP="${PROFILE_ITERS##*-}"
if (( NUM_REQUESTS <= PROFILE_STOP )); then
    echo "ERROR: NUM_REQUESTS=${NUM_REQUESTS} must exceed PROFILE_ITERS stop=${PROFILE_STOP}." >&2
    echo "       Raise NUM_REQUESTS or lower PROFILE_ITERS." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

# ---- Dataset (generate once, reuse) --------------------------------------
if [[ ! -f "${DATASET}" ]]; then
    echo "[run_nsys] generating dataset -> ${DATASET}"
    python3 "${SCRIPT_DIR}/gen_dataset.py" \
        --model "${MODEL}" \
        --isl "${ISL}" --osl "${OSL}" \
        --num-requests "${NUM_REQUESTS}" \
        --output "${DATASET}"
fi

REPORT="${OUT_DIR}/minimax_m3_fp4_c${CONCURRENCY}"
RUN_LOG="${OUT_DIR}/run_c${CONCURRENCY}.log"

echo "=================================================================="
echo "[run_nsys] MiniMax-M3-NVFP4  concurrency=${CONCURRENCY}"
echo "  model:          ${MODEL}"
echo "  tp/ep:          ${TP}/${EP}"
echo "  isl/osl:        ${ISL}/${OSL}"
echo "  max_seq_len:    ${MAX_SEQ_LEN}   max_num_tokens: ${MAX_NUM_TOKENS}"
echo "  warmup/reqs:    ${WARMUP}/${NUM_REQUESTS}"
echo "  profile iters:  ${PROFILE_ITERS} (steady-state window)"
echo "  capture end:    ${CAPTURE_END}"
echo "  cuda-graph-trace: ${CUDA_GRAPH_TRACE:-<off>}"
echo "  report:         ${REPORT}.nsys-rep"
echo "=================================================================="

# nsys flags:
#   -c cudaProfilerApi + TLLM_PROFILE_START_STOP : record only the steady-state
#       engine-iteration window [A, B]; the worker calls cudaProfilerStart/Stop.
#   --capture-range-end=stop-shutdown : tear the run down the moment the window
#       closes. This keeps the run short (it never processes the full request
#       set) and avoids the post-window stall that trips the 300s hang detector.
#   -t cuda,nvtx,python-gil : kernels + engine NVTX ranges + GIL contention.
#   --cuda-graph-trace : OFF by default; CUPTI node tracing can deadlock with
#       NCCL collectives captured in CUDA graphs. Opt in via CUDA_GRAPH_TRACE.
#   --trace-fork-before-exec=true : follow the tp worker processes.
#   -e ... : propagate profiling env vars to the (forked) worker processes.
NSYS_EXTRA_FLAGS=()
if [[ -n "${CUDA_GRAPH_TRACE}" ]]; then
    NSYS_EXTRA_FLAGS+=("--cuda-graph-trace=${CUDA_GRAPH_TRACE}")
fi

# With stop-shutdown, nsys terminates trtllm-bench at the window boundary, so
# a non-zero pipeline exit is EXPECTED. Don't let it abort the script / sweep;
# success is judged by the .nsys-rep existing below.
set +e
TLLM_PROFILE_START_STOP="${PROFILE_ITERS}" nsys profile \
    --output "${REPORT}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,python-gil \
    --capture-range=cudaProfilerApi \
    --capture-range-end="${CAPTURE_END}" \
    --trace-fork-before-exec=true \
    --wait="${NSYS_WAIT}" \
    --stats=false \
    "${NSYS_EXTRA_FLAGS[@]}" \
    -e "TLLM_PROFILE_START_STOP=${PROFILE_ITERS},TLLM_LLMAPI_ENABLE_NVTX=1" \
    -- \
    trtllm-bench \
        --model "${MODEL}" \
        --model_path "${MODEL}" \
        throughput \
        --backend pytorch \
        --config "${CONFIG}" \
        --dataset "${DATASET}" \
        --tp "${TP}" \
        --ep "${EP}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --max_num_tokens "${MAX_NUM_TOKENS}" \
        --concurrency "${CONCURRENCY}" \
        --warmup "${WARMUP}" \
        --num_requests "${NUM_REQUESTS}" \
        --report_json "${OUT_DIR}/report_c${CONCURRENCY}.json" \
    2>&1 | tee "${RUN_LOG}"
set -e

echo
if [[ -f "${REPORT}.nsys-rep" ]]; then
    echo "[run_nsys] Done. Report: ${REPORT}.nsys-rep"
    echo "[run_nsys] (A non-zero exit above is expected: --capture-range-end="
    echo "            ${CAPTURE_END} tears the run down at the window boundary.)"
    echo "[run_nsys] Summarize NVTX ranges via:"
    echo "    nsys stats --report nvtx_sum --format table ${REPORT}.nsys-rep"
else
    echo "[run_nsys] ERROR: ${REPORT}.nsys-rep was not produced. Check ${RUN_LOG}." >&2
    echo "[run_nsys] If the log shows the profile window was never reached, raise" >&2
    echo "           NUM_REQUESTS or lower PROFILE_ITERS." >&2
    exit 1
fi

# ---- Piecewise CUDA graph verification -----------------------------------
# Capture side (from the startup/warmup logs): these confirm graphs were
# actually captured (and warn if a bucket was skipped for exceeding the
# engine's reachable num_tokens ceiling).
echo
echo "[verify] Piecewise CUDA graph capture (from ${RUN_LOG}):"
if grep -qi "piecewise CUDA graph warmup" "${RUN_LOG}"; then
    grep -iE "Running piecewise CUDA graph warmup|Run piecewise CUDA graph warmup for num tokens" \
        "${RUN_LOG}" | sed 's/^/    /' || true
    if grep -qi "Skipping piecewise CUDA graph capture" "${RUN_LOG}"; then
        echo "    WARNING: a capture bucket was skipped (see below) -- that num_tokens"
        echo "             will run EAGER, not as a graph:"
        grep -i "Skipping piecewise CUDA graph capture" "${RUN_LOG}" | sed 's/^/      /'
    fi
else
    echo "    NOT FOUND -- piecewise capture did not run. Check that config.yaml"
    echo "    has torch_compile_config.enable_piecewise_cuda_graph: true and that"
    echo "    torch.compile is enabled."
fi
# Replay side (from the trace): the steady-state window is post-warmup, so any
# cudaGraphLaunch here is a graph *replay*. A count > 0 means piecewise (and/or
# the decode graph) is replaying during steady state.
echo
echo "[verify] Piecewise/CUDA graph replay (from the trace) -- run:"
echo "    nsys stats --report cuda_api_sum --format table ${REPORT}.nsys-rep | grep -iE 'Name|cudaGraphLaunch'"
echo "  cudaGraphLaunch count > 0 => graphs are replaying in the steady-state window."
echo "  For an A/B baseline, comment out torch_compile_config in config.yaml and"
echo "  re-run: the context region should then show many cudaLaunchKernel and no"
echo "  cudaGraphLaunch for the prefill pieces."
