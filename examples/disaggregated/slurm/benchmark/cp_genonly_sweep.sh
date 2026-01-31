#!/bin/bash

# Benchmark script for Context Parallelism (CP) sweep in gen-only mode
# This script updates gen worker's config in config.yaml and submits jobs.
#
# Global Batch Size:
#   global_batch_size = concurrency (the batch_size field in COMBINATIONS)
#
# Naming convention:
#   tep_N = TEP mode with N GPUs (AttnDP=false, max CP for low batch)
#   dep_N = DEP mode with N GPUs (AttnDP=true, higher TP for high batch)

set -e

# Working directory
WORK_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark"
CONFIG_FILE="${WORK_DIR}/config.yaml"

# Directory to save configs for review
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIGS_DIR="${WORK_DIR}/saved_configs/${TIMESTAMP}_cp_sweep"

# =============================================================================
# Format: "num_gpus,global_batch_size,isl,osl,gen_pp,gen_tp,gen_cp,gen_moe_ep,attn_dp"
# attn_dp: 0=false (TEP), 1=true (DEP)
# global_batch_size = concurrency
# =============================================================================

# =============================================================================
# 2 GPU COMBINATIONS - PP x TP x CP = 2
# =============================================================================
GEN2_COMBINATIONS=(
    # tep_2: TEP mode (PP=1, TP=1, CP=2) - concurrency 1,2,4,8
    "2,1,131072,8192,1,1,2,2,0"
    "2,2,131072,8192,1,1,2,2,0"
    "2,4,131072,8192,1,1,2,2,0"
    "2,8,131072,8192,1,1,2,2,0"
    # dep_2: DEP mode (PP=1, TP=2, CP=1) - concurrency 16
    "2,16,131072,8192,1,2,1,2,1"
)

# =============================================================================
# 4 GPU COMBINATIONS - PP x TP x CP = 4
# =============================================================================
GEN4_COMBINATIONS=(
    # tep_4: TEP mode (PP=1, TP=1, CP=4) - concurrency 1,2,4
    "4,1,131072,8192,1,1,4,4,0"
    "4,2,131072,8192,1,1,4,4,0"
    "4,4,131072,8192,1,1,4,4,0"
    # dep_4: DEP mode (PP=1, TP=2, CP=2) - concurrency 8,16,32,64
    "4,8,131072,8192,1,2,2,4,1"
    "4,16,131072,8192,1,2,2,4,1"
    "4,32,131072,8192,1,2,2,4,1"
    "4,64,131072,8192,1,2,2,4,1"
)

# =============================================================================
# 8 GPU COMBINATIONS - PP x TP x CP = 8
# =============================================================================
GEN8_COMBINATIONS=(
    # tep_8: TEP mode (PP=1, TP=1, CP=8) - concurrency 1,2,4,8
    "8,1,131072,8192,1,1,8,8,0"
    "8,2,131072,8192,1,1,8,8,0"
    "8,4,131072,8192,1,1,8,8,0"
    "8,8,131072,8192,1,1,8,8,0"
    # dep_8: DEP mode (PP=1, TP=4, CP=2) - concurrency 16,32,64,128,256
    "8,16,131072,8192,1,4,2,8,1"
    "8,32,131072,8192,1,4,2,8,1"
    "8,64,131072,8192,1,4,2,8,1"
    "8,128,131072,8192,1,4,2,8,1"
    "8,256,131072,8192,1,4,2,8,1"
)

# =============================================================================
# 16 GPU COMBINATIONS - PP x TP x CP = 16 (DEP only for high batch)
# =============================================================================
GEN16_COMBINATIONS=(
    # dep_16: DEP mode (PP=1, TP=8, CP=2) - concurrency 16,32,64,128,256,512
    "16,16,131072,8192,1,8,2,16,1"
    "16,32,131072,8192,1,8,2,16,1"
    "16,64,131072,8192,1,8,2,16,1"
    "16,128,131072,8192,1,8,2,16,1"
    "16,256,131072,8192,1,8,2,16,1"
    "16,512,131072,8192,1,8,2,16,1"
)

# =============================================================================
# 32 GPU COMBINATIONS - PP x TP x CP = 32 (DEP only for high batch)
# =============================================================================
GEN32_COMBINATIONS=(
    # dep_32: DEP mode (PP=1, TP=16, CP=2) - concurrency 32,64,128,256,512,1024
    "32,32,131072,8192,1,16,2,32,1"
    "32,64,131072,8192,1,16,2,32,1"
    "32,128,131072,8192,1,16,2,32,1"
    "32,256,131072,8192,1,16,2,32,1"
    "32,512,131072,8192,1,16,2,32,1"
    "32,1024,131072,8192,1,16,2,32,1"
)

# =============================================================================
# Helper functions
# =============================================================================

save_config() {
    local num_gpus=$1
    local global_batch_size=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8
    local attn_dp=$9

    # Create configs directory if it doesn't exist
    mkdir -p "$CONFIGS_DIR"

    # Generate descriptive filename
    local mode_str=$( [ "$attn_dp" -eq 1 ] && echo "dep" || echo "tep" )
    local config_name="${mode_str}_${num_gpus}_gbs${global_batch_size}_isl${isl}_osl${osl}_pp${pp}_tp${tp}_cp${cp}_ep${ep}.yaml"
    local save_path="${CONFIGS_DIR}/${config_name}"

    cp "$CONFIG_FILE" "$save_path"
    echo "Saved config to: $save_path"
}

# Function to determine moe_backend
get_moe_backend() {
    echo "CUTEDSL"
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
    local global_batch_size=$2  # global_batch_size = concurrency
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8
    local attn_dp=$9

    # Calculate derived values
    # global_batch_size = concurrency (directly from experiments)
    local concurrency=$global_batch_size
    local max_seq_len=$((isl + osl + 512))  # buffer for special tokens
    local moe_backend=$(get_moe_backend "$ep")
    local attn_dp_bool=$( [ "$attn_dp" -eq 1 ] && echo "true" || echo "false" )
    local mode_str=$( [ "$attn_dp" -eq 1 ] && echo "DEP" || echo "TEP" )
    # Worker max_batch_size calculation:
    # - max_batch_size in config = micro-batch size (per forward pass)
    # - Runtime computes: max_num_sequences = max_batch_size * pp_size (total in-flight capacity)
    # - With AttnDP: max_batch_size is per-rank, so micro-batch = global_batch_size / (tp * pp)
    # - Without AttnDP: micro-batch = global_batch_size / pp
    local worker_max_batch_size
    local micro_batch_size
    if [ "$attn_dp" -eq 1 ]; then
        # With AttnDP: per-rank micro-batch = global_batch_size / (tp * pp)
        # Runtime will compute max_num_sequences = micro_batch * pp = global_batch_size / tp (per-rank total)
        worker_max_batch_size=$(( global_batch_size / (tp * pp) ))
        micro_batch_size=$worker_max_batch_size
    else
        # Without AttnDP: micro-batch = global_batch_size / pp
        # Runtime will compute max_num_sequences = micro_batch * pp = global_batch_size (total)
        worker_max_batch_size=$(( global_batch_size / pp ))
        micro_batch_size=$worker_max_batch_size
    fi

    # Ensure minimum batch size of 1
    if [ "$worker_max_batch_size" -lt 1 ]; then
        worker_max_batch_size=1
        micro_batch_size=1
    fi

    echo "=========================================="
    echo "Updating config with:"
    echo "  Mode: ${mode_str}_${num_gpus} (${mode_str} mode with ${num_gpus} GPUs)"
    echo "  NUM_GPUS=$num_gpus, PP=$pp, TP=$tp, CP=$cp, EP=$ep"
    echo "  ISL=$isl, OSL=$osl"
    echo "  global_batch_size=$global_batch_size (= concurrency)"
    echo "  enable_attention_dp=$attn_dp_bool"
    echo "  concurrency=$concurrency, max_seq_len=$max_seq_len"
    echo "  moe_backend=$moe_backend (auto-selected for GB200 NVFP4)"
    if [ "$attn_dp" -eq 1 ]; then
        echo "  worker max_batch_size=$worker_max_batch_size (micro-batch = global_batch_size / (tp * pp) with AttnDP)"
        echo "  max_num_sequences per rank=$(( worker_max_batch_size * pp )) (= global_batch_size / tp)"
    else
        echo "  worker max_batch_size=$worker_max_batch_size (micro-batch = global_batch_size / pp)"
        echo "  max_num_sequences=$(( worker_max_batch_size * pp )) (= global_batch_size)"
    fi
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
    # Note: max_batch_size is the micro-batch size; runtime computes max_num_sequences = max_batch_size * pp
    sed -i "/worker_config:/,/ctx:/ {
        s/tensor_parallel_size: [0-9]*/tensor_parallel_size: $tp/
        s/pipeline_parallel_size: [0-9]*/pipeline_parallel_size: $pp/
        s/moe_expert_parallel_size: [0-9]*/moe_expert_parallel_size: $ep/
        s/context_parallel_size: [0-9]*/context_parallel_size: $cp/
        s/max_batch_size: [0-9]*/max_batch_size: $worker_max_batch_size/
        s/max_num_tokens: [0-9]*/max_num_tokens: $((worker_max_batch_size * 2))/
        s/max_seq_len: [0-9]*/max_seq_len: $max_seq_len/
    }" "$CONFIG_FILE"

    # Update enable_attention_dp in gen section
    sed -i "/^  gen:/,/^  ctx:/ {
        s/enable_attention_dp: [a-z]*/enable_attention_dp: $attn_dp_bool/
    }" "$CONFIG_FILE"

    # Update moe_config backend in gen section only (avoid matching ctx's moe_config or cache_transceiver_config.backend)
    sed -i "/^  gen:/,/^  ctx:/ {
        /moe_config:/,/cache_transceiver_config:/ {
            s/backend: [A-Z]*/backend: $moe_backend/
        }
    }" "$CONFIG_FILE"

    # Update cuda_graph_config.max_batch_size with micro_batch_size
    # With PP and/or AttnDP, each forward pass only sees micro_batch_size requests
    sed -i "/^  gen:/,/^  ctx:/ {
        /cuda_graph_config:/,/print_iter_log:/ {
            s/max_batch_size: [0-9]*/max_batch_size: $micro_batch_size/
        }
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

# Parse command line arguments
MODE="${1:-all}"  # Default to all if not specified

usage() {
    echo "Usage: $0 [gen2|gen4|gen8|gen16|gen32|all]"
    echo ""
    echo "Context Parallelism (CP) sweep for gen-only mode"
    echo "global_batch_size = concurrency"
    echo ""
    echo "Options:"
    echo "  gen2      Run 2 GPU instance combinations only (5 experiments)"
    echo "  gen4      Run 4 GPU instance combinations only (7 experiments)"
    echo "  gen8      Run 8 GPU instance combinations only (9 experiments)"
    echo "  gen16     Run 16 GPU instance combinations only (6 experiments)"
    echo "  gen32     Run 32 GPU instance combinations only (6 experiments)"
    echo "  all       Run all combinations (default, 33 experiments)"
    echo ""
    echo "Naming convention:"
    echo "  tep_N  - TEP mode with N GPUs (AttnDP=false, max CP for low batch)"
    echo "  dep_N  - DEP mode with N GPUs (AttnDP=true, higher TP for high batch)"
    echo ""
    exit 1
}

# Select the appropriate combinations array based on mode
case "$MODE" in
    gen2)
        COMBINATIONS=("${GEN2_COMBINATIONS[@]}")
        MODE_DESC="2 GPU CP"
        ;;
    gen4)
        COMBINATIONS=("${GEN4_COMBINATIONS[@]}")
        MODE_DESC="4 GPU CP"
        ;;
    gen8)
        COMBINATIONS=("${GEN8_COMBINATIONS[@]}")
        MODE_DESC="8 GPU CP"
        ;;
    gen16)
        COMBINATIONS=("${GEN16_COMBINATIONS[@]}")
        MODE_DESC="16 GPU CP"
        ;;
    gen32)
        COMBINATIONS=("${GEN32_COMBINATIONS[@]}")
        MODE_DESC="32 GPU CP"
        ;;
    all)
        COMBINATIONS=("${GEN2_COMBINATIONS[@]}" "${GEN4_COMBINATIONS[@]}" "${GEN8_COMBINATIONS[@]}" "${GEN16_COMBINATIONS[@]}" "${GEN32_COMBINATIONS[@]}")
        MODE_DESC="All CP"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        usage
        ;;
esac

# Navigate to work directory
cd "$WORK_DIR"

# Counter for tracking progress
total_combinations=${#COMBINATIONS[@]}
current=0

echo "============================================"
echo "Starting $MODE_DESC benchmark with $total_combinations combinations"
echo "Segment limit: 72 GPUs"
echo "Mode: gen_only_no_context"
echo "Note: global_batch_size = concurrency"
echo "============================================"

# Iterate through specific combinations
for combo in "${COMBINATIONS[@]}"; do
    current=$((current + 1))
    
    # Parse the combination (comma-separated values)
    IFS=',' read -r num_gpus global_batch_size isl osl gen_pp gen_tp gen_cp gen_moe_ep attn_dp <<< "$combo"
    
    mode_str=$( [ "$attn_dp" -eq 1 ] && echo "dep" || echo "tep" )
    echo ""
    echo "============================================"
    echo "[$MODE_DESC] Processing combination $current/$total_combinations"
    echo "  Experiment: ${mode_str}_${num_gpus}"
    echo "  Config: GPUs=$num_gpus, global_batch_size=$global_batch_size, ISL=$isl, OSL=$osl"
    echo "  Parallelism: PP=$gen_pp, TP=$gen_tp, CP=$gen_cp, EP=$gen_moe_ep"
    echo "============================================"

    # Update config with current parameters
    update_config "$num_gpus" "$global_batch_size" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep" "$attn_dp"

    # Save config for later review
    save_config "$num_gpus" "$global_batch_size" "$isl" "$osl" "$gen_pp" "$gen_tp" "$gen_cp" "$gen_moe_ep" "$attn_dp"

    # Submit the job
    submit_job

    # Optional: Add delay between submissions to avoid overwhelming the scheduler
    sleep 5
done

echo ""
echo "============================================"
echo "$MODE_DESC benchmark complete! Submitted $total_combinations jobs"
echo "Configs saved to: $CONFIGS_DIR"
echo "============================================"
