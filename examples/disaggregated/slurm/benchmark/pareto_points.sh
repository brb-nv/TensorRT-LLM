#!/bin/bash

# Benchmark script for specific parameter combinations
# This script updates gen worker's config in config.yaml and submits jobs.

set -e

# Working directory
WORK_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark"
CONFIG_FILE="${WORK_DIR}/config.yaml"

# Directory to save configs for review
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIGS_DIR="${WORK_DIR}/saved_configs/${TIMESTAMP}"

# =============================================================================
# DEFINE SPECIFIC COMBINATIONS - Modify this array as needed
# Format: "num_gpus,batch_size,isl,osl,gen_pp,gen_tp,gen_cp,gen_moe_ep"
# =============================================================================
COMBINATIONS=(
    # num_gpus, batch_size, isl, osl, gen_pp, gen_tp, gen_cp, gen_moe_ep
    # ck1_ctp16tp4
    "64,1,131072,8192,1,4,16,1"
    # ck2_ctp16tp2pp2
    "64,2,131072,8192,2,2,16,1"
    # ck4_ctp32pp2
    "64,4,131072,8192,2,1,32,1"
    # ck8_cep16pp4
    "64,8,131072,8192,4,1,16,16"
    # ck16_cep16pp4
    "64,16,131072,8192,4,1,16,16"
    # ck32_cep8pp8
    "64,32,131072,8192,8,1,8,8"
    # ck4_ctp4ep16
    "64,4,131072,8192,1,4,16,16"
    # ck8_ctp4ep16 - unsure abt this.
    "64,8,131072,8192,1,4,16,16"
    # ck224_ctp2cep32
    "64,224,131072,8192,1,1,64,32"
    # ck8_ctp2ep32
    "64,8,131072,8192,1,32,2,32"
)

# =============================================================================
# Helper functions
# =============================================================================

save_config() {
    local num_gpus=$1
    local batch_size=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8

    # Create configs directory if it doesn't exist
    mkdir -p "$CONFIGS_DIR"

    # Generate descriptive filename
    local config_name="gpus${num_gpus}_bs${batch_size}_isl${isl}_osl${osl}_pp${pp}_tp${tp}_cp${cp}_ep${ep}.yaml"
    local save_path="${CONFIGS_DIR}/${config_name}"

    cp "$CONFIG_FILE" "$save_path"
    echo "Saved config to: $save_path"
}

# Function to determine moe_backend based on EP for GB200 with NVFP4
# Reference: deployment-guide-for-deepseek-r1-on-trtllm.md
#   GB200 EP<=8, NVFP4 -> CUTLASS or TRTLLM
#   GB200 EP>8, NVFP4 -> WIDEEP
get_moe_backend() {
    local ep=$1
    if [ "$ep" -le 8 ]; then
        echo "CUTLASS"
    else
        echo "WIDEEP"
    fi
}

# Function to generate pp_partition YAML block for 61 layers
# Distributes layers as evenly as possible: first (pp-1) ranks get ceil(61/pp), last gets remainder
generate_pp_partition() {
    local pp=$1
    local total_layers=61
    
    if [ "$pp" -le 1 ]; then
        return  # No output for pp <= 1
    fi
    
    local layers_per_rank=$(( (total_layers + pp - 1) / pp ))
    local last_rank_layers=$(( total_layers - (pp - 1) * layers_per_rank ))
    
    echo "    pp_partition:"
    for ((i = 1; i < pp; i++)); do
        echo "    - $layers_per_rank"
    done
    echo "    - $last_rank_layers"
}

# Function to update config.yaml using sed
update_config() {
    local num_gpus=$1
    local batch_size=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8

    # Calculate derived values
    local concurrency=$(( batch_size * 8 < 64 ? batch_size * 8 : batch_size * 2 ))
    local max_seq_len=$((isl + osl + 512))  # buffer for special tokens
    local moe_backend=$(get_moe_backend "$ep")

    echo "=========================================="
    echo "Updating config with:"
    echo "  NUM_GPUS=$num_gpus, PP=$pp, TP=$tp, CP=$cp, EP=$ep"
    echo "  ISL=$isl, OSL=$osl, batch_size=$batch_size"
    echo "  concurrency=$concurrency, max_seq_len=$max_seq_len"
    echo "  moe_backend=$moe_backend (auto-selected for GB200 NVFP4)"
    if [ "$pp" -gt 1 ]; then
        local layers_per_rank=$(( (61 + pp - 1) / pp ))
        local last_rank_layers=$(( 61 - (pp - 1) * layers_per_rank ))
        echo "  pp_partition: $((pp - 1))x${layers_per_rank} + 1x${last_rank_layers} = 61 layers"
    fi
    echo "=========================================="

    # Update benchmark section
    sed -i "s/input_length: [0-9]*/input_length: $isl/" "$CONFIG_FILE"
    sed -i "s/output_length: [0-9]*/output_length: $osl/" "$CONFIG_FILE"
    sed -i "s/concurrency_list: \"[0-9]*\"/concurrency_list: \"$concurrency\"/" "$CONFIG_FILE"

    # Update gen worker config
    sed -i "/worker_config:/,/ctx:/ {
        s/tensor_parallel_size: [0-9]*/tensor_parallel_size: $tp/
        s/pipeline_parallel_size: [0-9]*/pipeline_parallel_size: $pp/
        s/moe_expert_parallel_size: [0-9]*/moe_expert_parallel_size: $ep/
        s/context_parallel_size: [0-9]*/context_parallel_size: $cp/
        s/max_batch_size: [0-9]*/max_batch_size: $batch_size/
        s/max_num_tokens: [0-9]*/max_num_tokens: $((batch_size * 32))/
        s/max_seq_len: [0-9]*/max_seq_len: $max_seq_len/
    }" "$CONFIG_FILE"

    # Update moe_config backend in gen section only (avoid matching ctx's moe_config or cache_transceiver_config.backend)
    sed -i "/^  gen:/,/^  ctx:/ {
        /moe_config:/,/cache_transceiver_config:/ {
            s/backend: [A-Z]*/backend: $moe_backend/
        }
    }" "$CONFIG_FILE"

    # Update cuda_graph_config.batch_sizes (multi-line YAML list format)
    sed -i "/batch_sizes:$/,/^[^-]/ {
        s/^\( *- \)[0-9][0-9]*/\1$batch_size/
    }" "$CONFIG_FILE"

    # Handle pp_partition: remove existing and add new if pp > 1
    # First, remove any existing pp_partition block in gen section (matches pp_partition: followed by lines starting with "    - ")
    sed -i "/^  gen:/,/^  ctx:/ {
        /^    pp_partition:$/,/^    [^-]/ {
            /^    pp_partition:$/d
            /^    - [0-9]/d
        }
    }" "$CONFIG_FILE"

    # Add pp_partition if pp > 1 (insert after num_postprocess_workers line in gen section)
    if [ "$pp" -gt 1 ]; then
        local pp_partition_block
        pp_partition_block=$(generate_pp_partition "$pp")
        # Use awk to insert after num_postprocess_workers in gen section
        awk -v pp_block="$pp_partition_block" '
            /^  gen:/ { in_gen=1 }
            /^  ctx:/ { in_gen=0 }
            { print }
            in_gen && /num_postprocess_workers:/ { print pp_block }
        ' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
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

# Navigate to work directory
cd "$WORK_DIR"

# Counter for tracking progress
total_combinations=${#COMBINATIONS[@]}
current=0

echo "============================================"
echo "Starting benchmark with $total_combinations specific combinations"
echo "============================================"

# Iterate through specific combinations
for combo in "${COMBINATIONS[@]}"; do
    current=$((current + 1))
    
    # Parse the combination (comma-separated values)
    IFS=',' read -r num_gpus batch_size isl osl gen_pp gen_tp gen_cp gen_moe_ep <<< "$combo"
    
    echo ""
    echo "============================================"
    echo "Processing combination $current/$total_combinations"
    echo "  Config: GPUs=$num_gpus, BS=$batch_size, ISL=$isl, OSL=$osl, PP=$gen_pp, TP=$gen_tp, CP=$gen_cp, EP=$gen_moe_ep"
    echo "============================================"

    # Update config with current parameters
    update_config "$num_gpus" "$batch_size" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep"

    # Save config for later review
    save_config "$num_gpus" "$batch_size" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep"

    # Submit the job
    submit_job

    # Optional: Add delay between submissions to avoid overwhelming the scheduler
    sleep 5
done

echo ""
echo "============================================"
echo "Benchmark complete! Submitted $total_combinations jobs"
echo "Configs saved to: $CONFIGS_DIR"
echo "============================================"
