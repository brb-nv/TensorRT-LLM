#!/bin/bash

# Sweep script for benchmarking different combinations of TP, CP, EP, ISL, batch_size
# This script updates gen worker's config in config.yaml and submits jobs.

set -e

# Working directory
WORK_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark"
CONFIG_FILE="${WORK_DIR}/config.yaml"

# Directory to save configs for review
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIGS_DIR="${WORK_DIR}/saved_configs/${TIMESTAMP}"

# =============================================================================
# DEFINE SWEEP PARAMETERS - Modify these arrays as needed
# =============================================================================
NUM_GPUS=16                       # Total number of GPUs (TP = NUM_GPUS / CP)
CP_VALUES=(1 8 16)                    # Context Parallel sizes
ISL_VALUES=(1048576)              # Input Sequence Lengths
BATCH_SIZE_VALUES=(32 64)          # Batch sizes

# =============================================================================
# Helper functions
# =============================================================================

save_config() {
    local tp=$1
    local cp=$2
    local ep=$3
    local isl=$4
    local batch_size=$5

    # Create configs directory if it doesn't exist
    mkdir -p "$CONFIGS_DIR"

    # Generate descriptive filename
    local config_name="tp${tp}_cp${cp}_ep${ep}_isl${isl}_bs${batch_size}.yaml"
    local save_path="${CONFIGS_DIR}/${config_name}"

    cp "$CONFIG_FILE" "$save_path"
    echo "Saved config to: $save_path"
}

# Function to update config.yaml using sed
update_config() {
    local tp=$1
    local cp=$2
    local ep=$3
    local isl=$4
    local batch_size=$5

    # Calculate derived values
    local concurrency=$(( batch_size * 8 < 64 ? batch_size * 8 : batch_size * 2 ))
    local max_seq_len=$((isl + 2560))  # osl is 1024.

    echo "=========================================="
    echo "Updating config with:"
    echo "  NUM_GPUS=$NUM_GPUS, TP=$tp, CP=$cp, EP=$ep"
    echo "  ISL=$isl, OSL=1024, batch_size=$batch_size"
    echo "  concurrency=$concurrency, max_seq_len=$max_seq_len"
    echo "=========================================="

    # Update benchmark section
    sed -i "s/input_length: [0-9]*/input_length: $isl/" "$CONFIG_FILE"
    sed -i "s/concurrency_list: \"[0-9]*\"/concurrency_list: \"$concurrency\"/" "$CONFIG_FILE"

    # Update gen worker config
    sed -i "/worker_config:/,/ctx:/ {
        s/tensor_parallel_size: [0-9]*/tensor_parallel_size: $tp/
        s/moe_expert_parallel_size: [0-9]*/moe_expert_parallel_size: $ep/
        s/context_parallel_size: [0-9]*/context_parallel_size: $cp/
        s/max_batch_size: [0-9]*/max_batch_size: $batch_size/
        s/max_num_tokens: [0-9]*/max_num_tokens: $((batch_size * 32))/
        s/max_seq_len: [0-9]*/max_seq_len: $max_seq_len/
    }" "$CONFIG_FILE"

    # Update cuda_graph_config.batch_sizes (multi-line YAML list format)
    sed -i "/batch_sizes:$/,/^[^-]/ {
        s/^\( *- \)[0-9][0-9]*/\1$batch_size/
    }" "$CONFIG_FILE"

    echo "Config updated successfully"
}

submit_job() {
    echo "Submitting job..."
    cd "$WORK_DIR"
    python3 submit.py -c config.yaml
    echo "Job submitted"
}

# =============================================================================
# Main execution
# =============================================================================

# Navigate to work directory
cd "$WORK_DIR"

# Counter for tracking progress
total_combinations=$((${#CP_VALUES[@]} * ${#ISL_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]}))
current=0

echo "============================================"
echo "Starting sweep with $total_combinations combinations"
echo "============================================"

# Iterate through all combinations
for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
    for cp in "${CP_VALUES[@]}"; do
        # Calculate TP from NUM_GPUS and CP
        tp=$((NUM_GPUS / cp))
        
        # Calculate EP as min(NUM_GPUS, 8)
        ep=$((NUM_GPUS < 8 ? NUM_GPUS : 8))
        
        for isl in "${ISL_VALUES[@]}"; do
            current=$((current + 1))
            echo ""
            echo "============================================"
            echo "Processing combination $current/$total_combinations"
            echo "============================================"

            # Update config with current parameters
            update_config "$tp" "$cp" "$ep" "$isl" "$batch_size"

            # Save config for later review
            save_config "$tp" "$cp" "$ep" "$isl" "$batch_size"

            # Submit the job
            submit_job

            # Optional: Add delay between submissions to avoid overwhelming the scheduler
            sleep 5
        done
    done
done

echo ""
echo "============================================"
echo "Sweep complete! Submitted $total_combinations jobs"
echo "Configs saved to: $CONFIGS_DIR"
echo "============================================"
