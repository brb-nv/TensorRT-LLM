#!/bin/bash

# Low Latency Benchmark script for TP sweep in gen-only mode - MiniMax M3
# Sweeps decode width x concurrency in the low-concurrency (high
# interactivity) zone.
#
# Model: MiniMax M3 NVFP4 (128 experts, 60 layers, ISL=8192, OSL=1024)
#
# Constraints:
#   - Concurrency: 1, 2, 4, 8, 16, 32
#   - TEP mode only (no AttnDP)
#   - PP = 1, CP = 1 (TP/EP only)
#   - prefill fixed at TP=1 (ctx worker config untouched)
#   - 4 <= num_gen_gpus <= 32
#
# Naming convention:
#   tep_N = TEP mode with N GPUs (AttnDP=false)

set -e

# Working directory
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${WORK_DIR}/config.yaml"

# Directory to save configs for review
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIGS_DIR="${WORK_DIR}/saved_configs/${TIMESTAMP}_tp_low_latency_m3"

# Client-side over-subscription factor: requests in flight = concurrency * this.
# Keep at 1: in gen_only mode with attention_dp=false submit.py caps the gen
# request queue at max_batch_size, so anything above 1 makes the extra requests
# arrive as a second wave that pollutes the latency measurement.
CONCURRENCY_MULTIPLIER="${CONCURRENCY_MULTIPLIER:-1}"

# =============================================================================
# Format: "num_gpus,concurrency,isl,osl,gen_pp,gen_tp,gen_cp,gen_moe_ep,attn_dp"
# attn_dp: 0=false (TEP), 1=true (DEP)
# concurrency drives the decode batch: gen max_batch_size = concurrency
# requests in flight = concurrency * CONCURRENCY_MULTIPLIER
# PP and CP are always 1 for this script (TP x EP only)
# =============================================================================

# =============================================================================
# CONCURRENCY 1 COMBINATIONS - Low latency single-request baseline
# Separated from the main arrays for targeted testing.
# =============================================================================
CONCURRENCY_1_COMBINATIONS=(
    # "4,1,8192,1024,1,4,1,4,0"     # tep_4
    # "8,1,8192,1024,1,8,1,8,0"     # tep_8
    # "16,1,8192,1024,1,16,1,16,0"  # tep_16
    "32,1,8192,1024,1,32,1,32,0"  # tep_32
)

# =============================================================================
# 4 GPU COMBINATIONS - TP = EP = 4 (1 node)
# =============================================================================
GEN4_COMBINATIONS=(
    # "4,2,8192,1024,1,4,1,4,0"
    # "4,4,8192,1024,1,4,1,4,0"
    # "4,8,8192,1024,1,4,1,4,0"
    # "4,16,8192,1024,1,4,1,4,0"
    # "4,32,8192,1024,1,4,1,4,0"
)

# =============================================================================
# 8 GPU COMBINATIONS - TP = EP = 8 (2 nodes)
# =============================================================================
GEN8_COMBINATIONS=(
    # "8,2,8192,1024,1,8,1,8,0"
    # "8,4,8192,1024,1,8,1,8,0"
    # "8,8,8192,1024,1,8,1,8,0"
    # "8,16,8192,1024,1,8,1,8,0"
    # "8,32,8192,1024,1,8,1,8,0"
)

# =============================================================================
# 16 GPU COMBINATIONS - TP = EP = 16 (4 nodes)
# =============================================================================
GEN16_COMBINATIONS=(
    # "16,2,8192,1024,1,16,1,16,0"
    # "16,4,8192,1024,1,16,1,16,0"
    # "16,8,8192,1024,1,16,1,16,0"
    # "16,16,8192,1024,1,16,1,16,0"
    # "16,32,8192,1024,1,16,1,16,0"
)

# =============================================================================
# 32 GPU COMBINATIONS - TP = EP = 32 (8 nodes)
# =============================================================================
GEN32_COMBINATIONS=(
    "32,2,8192,1024,1,32,1,32,0"
    "32,4,8192,1024,1,32,1,32,0"
    "32,8,8192,1024,1,32,1,32,0"
    "32,16,8192,1024,1,32,1,32,0"
    "32,32,8192,1024,1,32,1,32,0"
)

# Function to save config for review
save_config() {
    local num_gpus=$1
    local target_concurrency=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8
    local attn_dp=$9

    mkdir -p "$CONFIGS_DIR"

    local mode_str=$( [ "$attn_dp" -eq 1 ] && echo "dep" || echo "tep" )
    local config_name="${mode_str}_${num_gpus}_c${target_concurrency}_isl${isl}_osl${osl}_pp${pp}_tp${tp}_cp${cp}_ep${ep}.yaml"
    local save_path="${CONFIGS_DIR}/${config_name}"

    cp "$CONFIG_FILE" "$save_path"
    echo "Saved config to: $save_path"
}

# Function to update config.yaml using sed
update_config() {
    local num_gpus=$1
    local target_concurrency=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8
    local attn_dp=$9

    # Calculate derived values
    local concurrency=$((target_concurrency * CONCURRENCY_MULTIPLIER))
    local max_seq_len=$((isl + osl + 512))  # isl + osl + buffer for special tokens
    local attn_dp_bool=$( [ "$attn_dp" -eq 1 ] && echo "true" || echo "false" )
    local mode_str=$( [ "$attn_dp" -eq 1 ] && echo "DEP" || echo "TEP" )
    # Without AttnDP the instance shares one batch, so the decode batch is the
    # target concurrency regardless of width
    local worker_max_batch_size=$target_concurrency
    if [ "$worker_max_batch_size" -lt 1 ]; then
        worker_max_batch_size=1
    fi
    # Pure decode: every scheduled request contributes exactly one token per
    # step, so the token budget is the batch size
    local worker_max_num_tokens=$worker_max_batch_size

    echo "=========================================="
    echo "Updating config with:"
    echo "  Mode: ${mode_str}_${num_gpus} (${mode_str} mode with ${num_gpus} GPUs)"
    echo "  NUM_GPUS=$num_gpus, PP=$pp, TP=$tp, CP=$cp, EP=$ep"
    echo "  ISL=$isl, OSL=$osl"
    echo "  target concurrency=$target_concurrency, requests in flight=$concurrency (x${CONCURRENCY_MULTIPLIER})"
    echo "  enable_attention_dp=$attn_dp_bool"
    echo "  max_seq_len=$max_seq_len (isl + osl + 512 = $isl + $osl + 512)"
    echo "  worker max_batch_size=$worker_max_batch_size (also cuda_graph_config.max_batch_size)"
    echo "  worker max_num_tokens=$worker_max_num_tokens"
    echo "=========================================="

    # Update benchmark section
    sed -i "s/input_length: [0-9]*/input_length: $isl/" "$CONFIG_FILE"
    sed -i "s/output_length: [0-9]*/output_length: $osl/" "$CONFIG_FILE"
    sed -i "s/concurrency_list: \"[0-9]*\"/concurrency_list: \"$concurrency\"/" "$CONFIG_FILE"

    # Update gen worker config. Both max_batch_size occurrences in the gen block
    # (worker and cuda_graph_config) are set to the micro-batch size.
    sed -i "/worker_config:/,/ctx:/ {
        s/tensor_parallel_size: [0-9]*/tensor_parallel_size: $tp/
        s/pipeline_parallel_size: [0-9]*/pipeline_parallel_size: $pp/
        s/moe_expert_parallel_size: [0-9]*/moe_expert_parallel_size: $ep/
        s/context_parallel_size: [0-9]*/context_parallel_size: $cp/
        s/max_batch_size: [0-9]*/max_batch_size: $worker_max_batch_size/
        s/max_num_tokens: [0-9]*/max_num_tokens: $worker_max_num_tokens/
        s/max_seq_len: [0-9]*/max_seq_len: $max_seq_len/
    }" "$CONFIG_FILE"

    # Update enable_attention_dp in gen section
    sed -i "/^  gen:/,/^  ctx:/ {
        s/enable_attention_dp: [a-z]*/enable_attention_dp: $attn_dp_bool/
    }" "$CONFIG_FILE"

    echo "Config updated successfully"
}

submit_job() {
    echo "Submitting job..."
    cd "$WORK_DIR"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        python3 submit.py -c config.yaml --dry-run
    else
        python3 submit.py -c config.yaml
    fi
    echo "Job submitted"
}

# =============================================================================
# Main execution
# =============================================================================

MODE="${1:-all}"

usage() {
    echo "Usage: $0 [gen4|gen8|gen16|gen32|all|concurrency1]"
    echo ""
    echo "Low Latency TP sweep for gen-only mode (TEP only, no AttnDP)"
    echo "Model: MiniMax M3 NVFP4 (128 experts, 60 layers, ISL=8192, OSL=1024)"
    echo "Constraints: 4 <= num_gpus <= 32, PP=1, CP=1, prefill TP=1"
    echo "Concurrency: 1, 2, 4, 8, 16, 32"
    echo ""
    echo "Options:"
    echo "  gen4         Run 4 GPU combinations (concurrency >= 2)"
    echo "  gen8         Run 8 GPU combinations (concurrency >= 2)"
    echo "  gen16        Run 16 GPU combinations (concurrency >= 2)"
    echo "  gen32        Run 32 GPU combinations (concurrency >= 2)"
    echo "  all          Run all combinations (default, includes concurrency=1)"
    echo "  concurrency1 Run only concurrency=1 combinations (low latency baseline)"
    echo ""
    echo "Set DRY_RUN=1 to generate configs without submitting."
    echo ""
    echo "Naming convention:"
    echo "  tep_N  - TEP mode with N GPUs (AttnDP=false)"
    echo ""
    exit 1
}

case "$MODE" in
    gen4)
        COMBINATIONS=("${GEN4_COMBINATIONS[@]}")
        MODE_DESC="4 GPU TP"
        ;;
    gen8)
        COMBINATIONS=("${GEN8_COMBINATIONS[@]}")
        MODE_DESC="8 GPU TP"
        ;;
    gen16)
        COMBINATIONS=("${GEN16_COMBINATIONS[@]}")
        MODE_DESC="16 GPU TP"
        ;;
    gen32)
        COMBINATIONS=("${GEN32_COMBINATIONS[@]}")
        MODE_DESC="32 GPU TP"
        ;;
    all)
        COMBINATIONS=("${CONCURRENCY_1_COMBINATIONS[@]}" "${GEN4_COMBINATIONS[@]}" "${GEN8_COMBINATIONS[@]}" "${GEN16_COMBINATIONS[@]}" "${GEN32_COMBINATIONS[@]}")
        MODE_DESC="All TP Low Latency M3 (concurrency 1-32)"
        ;;
    concurrency1)
        COMBINATIONS=("${CONCURRENCY_1_COMBINATIONS[@]}")
        MODE_DESC="Concurrency=1 Low Latency Baseline"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        usage
        ;;
esac

cd "$WORK_DIR"

total_combinations=${#COMBINATIONS[@]}
current=0

echo "============================================"
echo "Starting $MODE_DESC benchmark with $total_combinations combinations"
echo "Model: MiniMax M3 NVFP4 (ISL=8192, OSL=1024)"
echo "Constraints: 4 <= num_gpus <= 32, PP=1, CP=1, TEP only"
echo "Mode: gen_only (low latency focus)"
echo "============================================"

for combo in "${COMBINATIONS[@]}"; do
    current=$((current + 1))

    IFS=',' read -r num_gpus target_concurrency isl osl gen_pp gen_tp gen_cp gen_moe_ep attn_dp <<< "$combo"

    mode_str=$( [ "$attn_dp" -eq 1 ] && echo "dep" || echo "tep" )
    echo ""
    echo "============================================"
    echo "[$MODE_DESC] Processing combination $current/$total_combinations"
    echo "  Experiment: ${mode_str}_${num_gpus}_c${target_concurrency}"
    echo "  Config: GPUs=$num_gpus, concurrency=$target_concurrency, ISL=$isl, OSL=$osl"
    echo "  Parallelism: PP=$gen_pp, TP=$gen_tp, CP=$gen_cp, EP=$gen_moe_ep, AttnDP=$attn_dp"
    echo "============================================"

    update_config "$num_gpus" "$target_concurrency" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep" "$attn_dp"

    save_config "$num_gpus" "$target_concurrency" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep" "$attn_dp"

    submit_job

    sleep 2
done

echo ""
echo "============================================"
echo "$MODE_DESC benchmark complete! Submitted $total_combinations jobs"
echo "Configs saved to: $CONFIGS_DIR"
echo "============================================"
