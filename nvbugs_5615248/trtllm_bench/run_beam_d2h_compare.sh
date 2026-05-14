#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Driver: benchmark the beam-history speculative-D2H opt-in across two
# commits on this branch, with three configurations:
#
#   1. baseline_<short>     : checkout BASELINE_SHA, run multirun (pre-PR code).
#   2. feature_off_<short>  : checkout FEATURE_SHA, run A/B with mode=off
#                             (default behavior; should match baseline).
#   3. feature_on_<short>   : checkout FEATURE_SHA, run A/B with mode=on
#                             (TRTLLM_ENABLE_BEAM_SEARCH_SPECULATIVE_D2H=1).
#
# Then aggregates with aggregate_runs.py:
#   feature_off vs baseline    -> regression sanity check
#   feature_on  vs baseline    -> headline win vs pre-PR
#   feature_on  vs feature_off -> headline win on the same commit
#
# Usage (inside the TRT-LLM container, from the repo root):
#   bash nvbugs_5615248/trtllm_bench/run_beam_d2h_compare.sh [out_root]
#
# Optional env var overrides:
#   BASELINE_SHA  default: f03cb1ce6b327171be8a0ed9ceed64a078294aff
#   FEATURE_SHA   default: 5f55cce559518f0e43da53f9d01a29a4fed5fcc8
#   NUM_RUNS      default: 5 (forwarded to inner launchers)
#   MODEL         default: TinyLlama-1.1B-Chat-v1.0 (see inner launchers)
#
# Notes:
# * Untracked dirs (this script, aggregate_runs.py, *.yaml, *.jsonl, the
#   per-run output dirs) survive git checkout, so the bench infra stays
#   put across commit swaps.
# * The driver refuses to run with modified tracked files in the working
#   tree, since `git checkout <sha>` would silently overwrite them.
# * On EXIT, the original ref (branch name or detached SHA) is restored.
# * Python bytecode is invalidated by source mtime (set by `git checkout`),
#   so no manual __pycache__ scrubbing is needed.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"

BASELINE_SHA="${BASELINE_SHA:-f03cb1ce6b327171be8a0ed9ceed64a078294aff}"
FEATURE_SHA="${FEATURE_SHA:-5f55cce559518f0e43da53f9d01a29a4fed5fcc8}"
NUM_RUNS="${NUM_RUNS:-5}"

OUT_ROOT="${1:-${WORKDIR}/beam_d2h_compare_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"
DRIVER_LOG="${OUT_ROOT}/driver.log"

log() { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "${DRIVER_LOG}"; }

# ---- Pre-flight checks ----------------------------------------------------

cd "${REPO_DIR}"

if ! command -v trtllm-bench >/dev/null 2>&1; then
    echo "ERROR: trtllm-bench not on PATH (are you inside the TRT-LLM container?)." >&2
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: not inside a git work tree (REPO_DIR=${REPO_DIR})." >&2
    exit 1
fi

# Working tree must be clean of modified tracked files so `git checkout`
# does not lose user edits. Untracked files are fine.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree has modified tracked files. Commit or stash before running:" >&2
    git status -s >&2
    exit 1
fi

for sha in "${BASELINE_SHA}" "${FEATURE_SHA}"; do
    if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
        echo "ERROR: SHA not reachable in this repo: ${sha}" >&2
        exit 1
    fi
done

for f in run_multirun_pytorch.sh run_multirun_beam_d2h_ab.sh aggregate_runs.py \
         pytorch.yaml dataset_isl100_osl20.jsonl; do
    if [[ ! -f "${WORKDIR}/${f}" ]]; then
        echo "ERROR: required file missing: ${WORKDIR}/${f}" >&2
        exit 1
    fi
done

# ---- Save and restore HEAD ------------------------------------------------

ORIG_REF="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
log "Original ref: ${ORIG_REF}"

restore_head() {
    log "Restoring original ref: ${ORIG_REF}"
    git checkout --quiet "${ORIG_REF}" || {
        echo "WARNING: failed to restore ${ORIG_REF}; current HEAD: $(git rev-parse HEAD)" >&2
    }
}
trap restore_head EXIT

# ---- Helpers --------------------------------------------------------------

short() { git rev-parse --short=10 "$1"; }

BASELINE_SHORT="$(short "${BASELINE_SHA}")"
FEATURE_SHORT="$(short "${FEATURE_SHA}")"

BASELINE_DIR="${OUT_ROOT}/baseline_${BASELINE_SHORT}"
FEATURE_OFF_DIR="${OUT_ROOT}/feature_off_${FEATURE_SHORT}"
FEATURE_ON_DIR="${OUT_ROOT}/feature_on_${FEATURE_SHORT}"

log "OUT_ROOT=${OUT_ROOT}"
log "BASELINE_SHA=${BASELINE_SHA} (${BASELINE_SHORT})"
log "FEATURE_SHA=${FEATURE_SHA} (${FEATURE_SHORT})"
log "NUM_RUNS=${NUM_RUNS}"

run_at_sha() {
    local sha="$1"; shift
    log "Checking out ${sha}"
    git checkout --quiet "${sha}"
    log "HEAD now at: $(git log -1 --oneline)"
    NUM_RUNS="${NUM_RUNS}" "$@"
}

# ---- Leg 1: baseline ------------------------------------------------------

log "=== Leg 1/3: baseline at ${BASELINE_SHORT} -> ${BASELINE_DIR}"
run_at_sha "${BASELINE_SHA}" \
    bash "${WORKDIR}/run_multirun_pytorch.sh" "${BASELINE_DIR}" \
    2>&1 | tee -a "${DRIVER_LOG}"

# ---- Leg 2: feature, env=off (default) -----------------------------------

log "=== Leg 2/3: feature@${FEATURE_SHORT}, mode=off -> ${FEATURE_OFF_DIR}"
run_at_sha "${FEATURE_SHA}" \
    bash "${WORKDIR}/run_multirun_beam_d2h_ab.sh" off "${FEATURE_OFF_DIR}" \
    2>&1 | tee -a "${DRIVER_LOG}"

# ---- Leg 3: feature, env=on ----------------------------------------------

log "=== Leg 3/3: feature@${FEATURE_SHORT}, mode=on -> ${FEATURE_ON_DIR}"
run_at_sha "${FEATURE_SHA}" \
    bash "${WORKDIR}/run_multirun_beam_d2h_ab.sh" on "${FEATURE_ON_DIR}" \
    2>&1 | tee -a "${DRIVER_LOG}"

# ---- Aggregate ------------------------------------------------------------

log "=== Aggregating results"

aggregate() {
    local base_dir="$1" base_label="$2"
    local exp_dir="$3"  exp_label="$4"
    local out_md="$5"

    log "  ${exp_label} vs ${base_label} -> ${out_md}"
    python3 "${WORKDIR}/aggregate_runs.py" \
        --baseline "${base_dir}" --baseline-label "${base_label}" \
        --experiment "${exp_dir}" --experiment-label "${exp_label}" \
        > "${out_md}" 2>&1
}

aggregate \
    "${BASELINE_DIR}"   "baseline_${BASELINE_SHORT}" \
    "${FEATURE_OFF_DIR}" "feature_off_${FEATURE_SHORT}" \
    "${OUT_ROOT}/cmp_feature_off_vs_baseline.md"

aggregate \
    "${BASELINE_DIR}"  "baseline_${BASELINE_SHORT}" \
    "${FEATURE_ON_DIR}" "feature_on_${FEATURE_SHORT}" \
    "${OUT_ROOT}/cmp_feature_on_vs_baseline.md"

aggregate \
    "${FEATURE_OFF_DIR}" "feature_off_${FEATURE_SHORT}" \
    "${FEATURE_ON_DIR}"  "feature_on_${FEATURE_SHORT}" \
    "${OUT_ROOT}/cmp_feature_on_vs_feature_off.md"

log "=== Done"
log "Per-leg directories:"
log "  ${BASELINE_DIR}"
log "  ${FEATURE_OFF_DIR}"
log "  ${FEATURE_ON_DIR}"
log "Comparison reports:"
log "  ${OUT_ROOT}/cmp_feature_off_vs_baseline.md   (regression sanity)"
log "  ${OUT_ROOT}/cmp_feature_on_vs_baseline.md    (headline)"
log "  ${OUT_ROOT}/cmp_feature_on_vs_feature_off.md (headline, same SHA)"
