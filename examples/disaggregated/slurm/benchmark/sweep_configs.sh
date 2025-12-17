#!/bin/bash

# Sweep script for benchmarking different combinations of TP, CP, EP, ISL, batch_size
# This script updates gen worker's config in config.yaml and submits jobs.

set -e

# Working directory
WORK_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark"
CONFIG_FILE="${WORK_DIR}/config.yaml"
CONFIG_BACKUP="${WORK_DIR}/config.yaml.backup"

# =============================================================================
# DEFINE SWEEP PARAMETERS - Modify these arrays as needed
# =============================================================================
NUM_GPUS=8                       # Total number of GPUs (TP = NUM_GPUS / CP)
CP_VALUES=(1 2 4 8)                    # Context Parallel sizes
ISL_VALUES=(2**15 2**16 2**17 2**18 2**19 2**20)              # Input Sequence Lengths
BATCH_SIZE_VALUES=(1 2 4 8)          # Batch sizes

# =============================================================================
# Helper functions
# =============================================================================

backup_config() {
    cp "$CONFIG_FILE" "$CONFIG_BACKUP"
    echo "Backed up config to $CONFIG_BACKUP"
}

restore_config() {
    if [ -f "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        echo "Restored config from backup"
    fi
}

# Function to update config.yaml using yq
update_config() {
    local tp=$1
    local cp=$2
    local ep=$3
    local isl=$4
    local batch_size=$5

    # Calculate derived values
    local concurrency=$(( batch_size * 4 < 64 ? batch_size * 4 : 64 ))
    local max_seq_len=$((isl + 2560))  # osl is 1024.

    echo "=========================================="
    echo "Updating config with:"
    echo "  NUM_GPUS=$NUM_GPUS, TP=$tp (NUM_GPUS/CP), CP=$cp, EP=$ep"
    echo "  ISL=$isl, batch_size=$batch_size"
    echo "  concurrency=$concurrency, max_seq_len=$max_seq_len"
    echo "=========================================="

    # Check if yq is available
    if command -v yq &> /dev/null; then
        # Use yq for YAML manipulation
        yq -i ".benchmark.input_length = $isl" "$CONFIG_FILE"
        yq -i ".benchmark.concurrency_list = \"$concurrency\"" "$CONFIG_FILE"

        yq -i ".worker_config.gen.tensor_parallel_size = $tp" "$CONFIG_FILE"
        yq -i ".worker_config.gen.context_parallel_size = $cp" "$CONFIG_FILE"
        yq -i ".worker_config.gen.moe_expert_parallel_size = $ep" "$CONFIG_FILE"
        yq -i ".worker_config.gen.max_batch_size = $batch_size" "$CONFIG_FILE"
        yq -i ".worker_config.gen.max_num_tokens = $((batch_size * 3))" "$CONFIG_FILE"
        yq -i ".worker_config.gen.max_seq_len = $max_seq_len" "$CONFIG_FILE"

        # Update cuda_graph_config.batch_sizes to include sizes from 1 to batch_size
        yq -i ".worker_config.gen.cuda_graph_config.batch_sizes = [$batch_size]" "$CONFIG_FILE"

    else
        # Fallback to sed-based approach
        echo "Warning: yq not found, using sed-based approach"
        
        # Update benchmark section
        sed -i "s/input_length: [0-9]*/input_length: $isl/" "$CONFIG_FILE"
        sed -i "s/concurrency_list: \"[0-9]*\"/concurrency_list: \"$concurrency\"/" "$CONFIG_FILE"

        # Update gen worker config
        sed -i "/worker_config:/,/ctx:/ {
            s/tensor_parallel_size: [0-9]*/tensor_parallel_size: $tp/
            s/moe_expert_parallel_size: [0-9]*/moe_expert_parallel_size: $ep/
            s/context_parallel_size: [0-9]*/context_parallel_size: $cp/
            s/max_batch_size: [0-9]*/max_batch_size: $batch_size/
            s/max_num_tokens: [0-9]*/max_num_tokens: $((batch_size * 3))/
            s/max_seq_len: [0-9]*/max_seq_len: $max_seq_len/
        }" "$CONFIG_FILE"
    fi

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

# Trap to restore config on exit/error
trap restore_config EXIT

# Navigate to work directory
cd "$WORK_DIR"

# Backup original config
backup_config

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

            # Restore original config before each update
            restore_config

            # Update config with current parameters
            update_config "$tp" "$cp" "$ep" "$isl" "$batch_size"

            # Submit the job
            submit_job

            # Optional: Add delay between submissions to avoid overwhelming the scheduler
            sleep 2
        done
    done
done

echo ""
echo "============================================"
echo "Sweep complete! Submitted $total_combinations jobs"
echo "============================================"

