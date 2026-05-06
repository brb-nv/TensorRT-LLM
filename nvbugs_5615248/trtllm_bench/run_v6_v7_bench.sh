#!/usr/bin/env bash
# Two-ref bench driver for the v6+v7 beam-search-host-overhead PR.
#
# Compares two HEAD-relative refs by toggling the working tree between them
# and running ``run_multirun_pytorch.sh`` against each. Designed to be run
# from the repo root, inside the TRT-LLM container, on a single GPU.
#
#   baseline = HEAD~2   = piecewise CUDA-graph capture commit (fb0acdde05)
#   feature  = HEAD     = piecewise + v6 + v7
#
# v6 and v7 only modify ``tensorrt_llm/_torch/pyexecutor/sampler.py`` (pure
# Python), so no rebuild is required when toggling refs - the Python entry
# point ``trtllm-bench`` re-imports the module on every launch.
#
# Output layout (per ref, ${OUT_ROOT}/<label>/):
#   report_pytorch{,2..N}.json
#   request_pytorch{,2..N}.json
#   output_pytorch{,2..N}.json
#   run_pytorch{,2..N}.log
#
# After both refs finish, aggregates with aggregate_runs.py and prints
# pooled-per-request Welch's t-test for TTFT / E2E / ITL.
#
# Usage::
#
#   bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh
#
# Override the output root (default: nvbugs_5615248/trtllm_bench/v6_v7_validation)::
#
#   OUT_ROOT=/some/scratch/dir bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh
#
# Override the runs-per-ref count (default: NUM_RUNS=5, matching v3..v7)::
#
#   NUM_RUNS=3 bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh
#
# Skip an already-complete ref::
#
#   SKIP_BASELINE=1 bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh
#   SKIP_FEATURE=1  bash nvbugs_5615248/trtllm_bench/run_v6_v7_bench.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${REPO_DIR}/nvbugs_5615248/trtllm_bench"
OUT_ROOT="${OUT_ROOT:-${WORKDIR}/v6_v7_validation}"
NUM_RUNS="${NUM_RUNS:-5}"

BASELINE_REF="${BASELINE_REF:-HEAD~2}"
FEATURE_REF="${FEATURE_REF:-HEAD}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline_piecewise}"
FEATURE_LABEL="${FEATURE_LABEL:-feature_piecewise_v6_v7}"

cd "${REPO_DIR}"

# -----------------------------------------------------------------------------
# Pre-flight: make sure the user is inside the container, that the bench harness
# is on disk, and that the branch HEAD is the expected three-commit shape.
# -----------------------------------------------------------------------------
require () {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' not on PATH (are you inside the TRT-LLM container?)" >&2
        exit 1
    fi
}
require trtllm-bench
require git
require python3

for f in run_multirun_pytorch.sh aggregate_runs.py pytorch.yaml dataset_isl100_osl20.jsonl; do
    if [[ ! -e "${WORKDIR}/${f}" ]]; then
        echo "ERROR: missing ${WORKDIR}/${f}" >&2
        exit 1
    fi
done

# Resolve refs to immutable SHAs *now* so a later checkout can't move our
# notion of "baseline" or "feature".
BASELINE_SHA=$(git rev-parse "${BASELINE_REF}^{commit}")
FEATURE_SHA=$(git rev-parse "${FEATURE_REF}^{commit}")
ORIGINAL_HEAD=$(git rev-parse --abbrev-ref HEAD)
if [[ "${ORIGINAL_HEAD}" == "HEAD" ]]; then
    ORIGINAL_HEAD=$(git rev-parse HEAD)  # detached
fi

# Sanity-check the branch shape: HEAD..HEAD~2 should touch only sampler.py.
TOUCHED=$(git diff --name-only "${BASELINE_SHA}" "${FEATURE_SHA}" | sort -u)
EXPECTED=$'tensorrt_llm/_torch/pyexecutor/sampler.py'
if [[ "${TOUCHED}" != "${EXPECTED}" ]]; then
    echo "WARNING: baseline..feature touches files other than sampler.py:" >&2
    echo "${TOUCHED}" >&2
    echo "Continuing anyway, but the toggle-without-rebuild assumption may be unsafe." >&2
fi

# Refuse to run with a dirty working tree on tracked files (untracked is fine -
# nvbugs_5615248/ is expected to be untracked on this branch).
if ! git diff --quiet --exit-code; then
    echo "ERROR: working tree has unstaged changes on tracked files. Commit or stash first." >&2
    exit 1
fi
if ! git diff --cached --quiet --exit-code; then
    echo "ERROR: working tree has staged changes. Commit or stash first." >&2
    exit 1
fi

# Restore original HEAD on any exit (success, error, ctrl-c).
restore_head () {
    local rc=$?
    if [[ -n "${ORIGINAL_HEAD}" ]]; then
        echo "[v6_v7_bench] restoring original HEAD: ${ORIGINAL_HEAD}"
        git checkout --quiet "${ORIGINAL_HEAD}" || true
    fi
    exit "${rc}"
}
trap restore_head EXIT

# -----------------------------------------------------------------------------
# One-ref bench helper. Toggles working tree to ${ref}, runs the existing
# multi-run launcher into ${out}, then leaves ${ref} checked out (the trap
# restores the original HEAD at the very end).
# -----------------------------------------------------------------------------
bench_ref () {
    local ref="$1"
    local label="$2"
    local out="${OUT_ROOT}/${label}"

    if [[ -f "${out}/.done" ]]; then
        echo "[v6_v7_bench] ${label}: ${out}/.done present, skipping."
        return 0
    fi

    echo "=================================================================="
    echo "[v6_v7_bench] ${label}: checking out ${ref}"
    echo "=================================================================="
    git checkout --quiet "${ref}"

    mkdir -p "${out}"

    NUM_RUNS="${NUM_RUNS}" bash "${WORKDIR}/run_multirun_pytorch.sh" "${out}"

    # Stamp a .done marker so re-running the driver auto-skips this ref.
    git rev-parse HEAD >"${out}/.head_sha"
    echo "${label}" >"${out}/.label"
    touch "${out}/.done"
}

mkdir -p "${OUT_ROOT}"

# -----------------------------------------------------------------------------
# Baseline: piecewise commit (HEAD~2 by default).
# -----------------------------------------------------------------------------
if [[ -n "${SKIP_BASELINE:-}" ]]; then
    echo "[v6_v7_bench] SKIP_BASELINE=1, skipping baseline."
else
    bench_ref "${BASELINE_SHA}" "${BASELINE_LABEL}"
fi

# -----------------------------------------------------------------------------
# Feature: piecewise + v6 + v7 (HEAD by default).
# -----------------------------------------------------------------------------
if [[ -n "${SKIP_FEATURE:-}" ]]; then
    echo "[v6_v7_bench] SKIP_FEATURE=1, skipping feature."
else
    bench_ref "${FEATURE_SHA}" "${FEATURE_LABEL}"
fi

# -----------------------------------------------------------------------------
# Aggregate.
# -----------------------------------------------------------------------------
echo
echo "=================================================================="
echo "[v6_v7_bench] aggregate ${FEATURE_LABEL} vs ${BASELINE_LABEL}"
echo "=================================================================="
python3 "${WORKDIR}/aggregate_runs.py" \
    --backend pytorch \
    --baseline "${OUT_ROOT}/${BASELINE_LABEL}" \
    --baseline-label "${BASELINE_LABEL}" \
    --experiment "${OUT_ROOT}/${FEATURE_LABEL}" \
    --experiment-label "${FEATURE_LABEL}"

echo
echo "Done. Outputs under: ${OUT_ROOT}/"
echo "  ${OUT_ROOT}/${BASELINE_LABEL}/  (HEAD@$(cat "${OUT_ROOT}/${BASELINE_LABEL}/.head_sha" 2>/dev/null || echo unknown))"
echo "  ${OUT_ROOT}/${FEATURE_LABEL}/   (HEAD@$(cat "${OUT_ROOT}/${FEATURE_LABEL}/.head_sha" 2>/dev/null || echo unknown))"
