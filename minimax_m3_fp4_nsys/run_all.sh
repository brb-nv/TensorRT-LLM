#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep run_nsys.sh over one or more concurrencies for the MiniMax-M3-NVFP4
# ISL=8192 / OSL=1 steady-state profiling workload. One nsys report per
# concurrency is written under out/c<N>/.
#
# Usage (inside the TRT-LLM container, from the repo root):
#   # Default: concurrency 1 only.
#   bash minimax_m3_fp4_nsys/run_all.sh
#
#   # Full sweep (target concurrencies):
#   bash minimax_m3_fp4_nsys/run_all.sh 1 2 4 8 16
#
# All env overrides accepted by run_nsys.sh (MODEL, TP, EP, PROFILE_ITERS, ...)
# are honored here too.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to concurrency 1 when no concurrencies are passed.
CONCURRENCIES=("$@")
if [[ ${#CONCURRENCIES[@]} -eq 0 ]]; then
    CONCURRENCIES=(1)
fi

echo "[run_all] concurrencies: ${CONCURRENCIES[*]}"
for c in "${CONCURRENCIES[@]}"; do
    echo
    echo "###################################################################"
    echo "[run_all] concurrency=${c}"
    echo "###################################################################"
    bash "${SCRIPT_DIR}/run_nsys.sh" "${c}"
done

echo
echo "[run_all] Done. Reports under ${SCRIPT_DIR}/out/c<N>/."
