#!/bin/bash

# Cross-product benchmark configuration generator
# Generates all valid (PP, TP, CP, EP) combinations for systematic exploration
#
# Constraints:
#   1. PP × TP × CP = 64
#   2. EP = TP × CP
#   3. CP >= 2, TP >= 2 (practical minimums)
#   4. AttnDP = false for BS <= 32, true for BS >= 64
#
# Total experiments: 95

set -e

# Working directory
WORK_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark"
CONFIG_FILE="${WORK_DIR}/config.yaml"

# Directory to save configs and logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CROSSPROD_DIR="${WORK_DIR}/${TIMESTAMP}_trtllm_crossprod"
CONFIGS_DIR="${CROSSPROD_DIR}/saved_configs"

# Fixed parameters
NUM_GPUS=64
ISL=131072
OSL=8192

# Batch sizes to test (limited to <= 3072)
BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512 1024 1792 2048 3072)

# =============================================================================
# Valid (TP, CP) pairs for each PP value
# Constraints: PP × TP × CP = 64, TP >= 2, CP >= 2
# =============================================================================

# PP=1: TP×CP=64 -> (32,2), (16,4), (8,8), (4,16), (2,32)
PP1_CONFIGS=("32,2" "16,4" "8,8" "4,16" "2,32")

# PP=2: TP×CP=32 -> (16,2), (8,4), (4,8), (2,16)
PP2_CONFIGS=("16,2" "8,4" "4,8" "2,16")

# PP=4: TP×CP=16 -> (8,2), (4,4), (2,8)
PP4_CONFIGS=("8,2" "4,4" "2,8")

# PP=8: TP×CP=8 -> (4,2), (2,4)
PP8_CONFIGS=("4,2" "2,4")

# =============================================================================
# Helper functions (same as pareto_points.sh)
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
    local attn_dp=$9

    mkdir -p "$CONFIGS_DIR"

    local attn_dp_str=$( [ "$attn_dp" -eq 1 ] && echo "attnDP" || echo "noAttnDP" )
    local config_name="gpus${num_gpus}_bs${batch_size}_isl${isl}_osl${osl}_pp${pp}_tp${tp}_cp${cp}_ep${ep}_${attn_dp_str}.yaml"
    local save_path="${CONFIGS_DIR}/${config_name}"

    cp "$CONFIG_FILE" "$save_path"
    echo "Saved config to: $save_path"
}

get_moe_backend() {
    echo "CUTLASS"
}

generate_pp_partition() {
    local pp=$1
    local total_layers=61
    
    if [ "$pp" -le 1 ]; then
        return
    fi
    
    local layers_per_rank=$(( (total_layers + pp - 1) / pp ))
    local last_rank_layers=$(( total_layers - (pp - 1) * layers_per_rank ))
    
    echo "    pp_partition:"
    for ((i = 1; i < pp; i++)); do
        echo "    - $layers_per_rank"
    done
    echo "    - $last_rank_layers"
}

update_config() {
    local num_gpus=$1
    local batch_size=$2
    local isl=$3
    local osl=$4
    local pp=$5
    local tp=$6
    local cp=$7
    local ep=$8
    local attn_dp=$9

    local concurrency=$(( batch_size * 8 < 64 ? batch_size * 8 : batch_size * 2 ))
    local max_seq_len=$((isl + osl + 512))
    local moe_backend=$(get_moe_backend "$ep")
    local attn_dp_bool=$( [ "$attn_dp" -eq 1 ] && echo "true" || echo "false" )
    
    local worker_max_batch_size
    local micro_batch_size
    if [ "$attn_dp" -eq 1 ]; then
        worker_max_batch_size=$(( batch_size / (tp * pp) ))
        micro_batch_size=$worker_max_batch_size
    else
        worker_max_batch_size=$(( batch_size / pp ))
        micro_batch_size=$worker_max_batch_size
    fi

    echo "=========================================="
    echo "Updating config with:"
    echo "  NUM_GPUS=$num_gpus, PP=$pp, TP=$tp, CP=$cp, EP=$ep"
    echo "  ISL=$isl, OSL=$osl, batch_size=$batch_size"
    echo "  enable_attention_dp=$attn_dp_bool"
    echo "  concurrency=$concurrency, max_seq_len=$max_seq_len"
    echo "  moe_backend=$moe_backend"
    if [ "$attn_dp" -eq 1 ]; then
        echo "  worker max_batch_size=$worker_max_batch_size (micro-batch = batch_size / (tp * pp) with AttnDP)"
        echo "  max_num_sequences per rank=$(( worker_max_batch_size * pp )) (= batch_size / tp)"
    else
        echo "  worker max_batch_size=$worker_max_batch_size (micro-batch = batch_size / pp)"
        echo "  max_num_sequences=$(( worker_max_batch_size * pp )) (= batch_size)"
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

    # Update moe_config backend in gen section
    sed -i "/^  gen:/,/^  ctx:/ {
        /moe_config:/,/cache_transceiver_config:/ {
            s/backend: [A-Z]*/backend: $moe_backend/
        }
    }" "$CONFIG_FILE"

    # Update cuda_graph_config.max_batch_size
    sed -i "/^  gen:/,/^  ctx:/ {
        /cuda_graph_config:/,/print_iter_log:/ {
            s/max_batch_size: [0-9]*/max_batch_size: $micro_batch_size/
        }
    }" "$CONFIG_FILE"

    # Handle pp_partition: remove existing and add new if pp > 1
    sed -i "/^  gen:/,/^  ctx:/ {
        /^    pp_partition:$/,/^    [^-]/ {
            /^    pp_partition:$/d
            /^    - [0-9]/d
        }
    }" "$CONFIG_FILE"

    if [ "$pp" -gt 1 ]; then
        local pp_partition_block
        pp_partition_block=$(generate_pp_partition "$pp")
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
    local log_dir=$1
    echo "Submitting job..."
    echo "Log directory: $log_dir"
    cd "$WORK_DIR"
    python3 submit.py -c config.yaml --log-dir "$log_dir"
    echo "Job submitted"
}

# =============================================================================
# Generate log directory path for an experiment
# Format: <CROSSPROD_DIR>/bs<batch_size>/pp<pp>_tp<tp>_cp<cp>_ep<ep>_<attn_dp_str>
# =============================================================================
get_log_dir() {
    local batch_size=$1
    local pp=$2
    local tp=$3
    local cp=$4
    local ep=$5
    local attn_dp=$6
    
    local attn_dp_str=$( [ "$attn_dp" -eq 1 ] && echo "attnDP" || echo "noAttnDP" )
    local bs_dir="${CROSSPROD_DIR}/bs${batch_size}"
    local exp_name="pp${pp}_tp${tp}_cp${cp}_ep${ep}_${attn_dp_str}"
    
    echo "${bs_dir}/${exp_name}"
}

# =============================================================================
# Get PP values to test for a given batch size
# =============================================================================
get_pp_values() {
    local bs=$1
    
    case $bs in
        1|2)
            echo "1"
            ;;
        4|8)
            echo "1 2"
            ;;
        16)
            echo "2 4"
            ;;
        32)
            echo "2 4 8"
            ;;
        64|128|256|512|1024|1792)
            echo "1"
            ;;
        2048)
            echo "1 2"
            ;;
        3072)
            echo "1 2 4"
            ;;
        *)
            echo "1"  # Default
            ;;
    esac
}

# =============================================================================
# Get (TP, CP) configs for a given PP value
# =============================================================================
get_tp_cp_configs() {
    local pp=$1
    
    case $pp in
        1)
            echo "${PP1_CONFIGS[@]}"
            ;;
        2)
            echo "${PP2_CONFIGS[@]}"
            ;;
        4)
            echo "${PP4_CONFIGS[@]}"
            ;;
        8)
            echo "${PP8_CONFIGS[@]}"
            ;;
        *)
            echo ""
            ;;
    esac
}

# =============================================================================
# Determine AttnDP setting based on batch size
# =============================================================================
get_attn_dp() {
    local bs=$1
    if [ "$bs" -le 32 ]; then
        echo "0"  # false
    else
        echo "1"  # true
    fi
}

# =============================================================================
# Count total experiments (dry-run mode)
# =============================================================================
count_experiments() {
    local total=0
    
    for bs in "${BATCH_SIZES[@]}"; do
        local pp_values
        pp_values=$(get_pp_values "$bs")
        
        for pp in $pp_values; do
            local configs
            configs=$(get_tp_cp_configs "$pp")
            
            for config in $configs; do
                total=$((total + 1))
            done
        done
    done
    
    echo "$total"
}

# =============================================================================
# Generate all combinations
# =============================================================================
generate_combinations() {
    local combinations=()
    
    for bs in "${BATCH_SIZES[@]}"; do
        local attn_dp
        attn_dp=$(get_attn_dp "$bs")
        
        local pp_values
        pp_values=$(get_pp_values "$bs")
        
        for pp in $pp_values; do
            local configs
            configs=$(get_tp_cp_configs "$pp")
            
            for config in $configs; do
                IFS=',' read -r tp cp <<< "$config"
                local ep=$((tp * cp))  # EP = TP × CP
                
                # Format: num_gpus,batch_size,isl,osl,pp,tp,cp,ep,attn_dp
                combinations+=("${NUM_GPUS},${bs},${ISL},${OSL},${pp},${tp},${cp},${ep},${attn_dp}")
            done
        done
    done
    
    printf '%s\n' "${combinations[@]}"
}

# =============================================================================
# Main execution
# =============================================================================

usage() {
    echo "Usage: $0 [run|count|list]"
    echo ""
    echo "Options:"
    echo "  run     Submit all experiments (default)"
    echo "  count   Count total experiments without submitting"
    echo "  list    List all configurations without submitting"
    echo ""
    exit 1
}

MODE="${1:-run}"

case "$MODE" in
    count)
        total=$(count_experiments)
        echo "Total experiments: $total"
        exit 0
        ;;
    list)
        echo "Cross-product configurations (PP×TP×CP=64, EP=TP×CP):"
        echo "============================================"
        echo "Format: GPUs,BS,ISL,OSL,PP,TP,CP,EP,AttnDP"
        echo "============================================"
        generate_combinations
        echo "============================================"
        total=$(count_experiments)
        echo "Total: $total experiments"
        exit 0
        ;;
    run)
        # Continue to main execution
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

# Generate all combinations
readarray -t COMBINATIONS < <(generate_combinations)

total_combinations=${#COMBINATIONS[@]}
current=0

echo "============================================"
echo "Cross-Product Benchmark Generator"
echo "Constraints: PP×TP×CP=64, EP=TP×CP, TP>=2, CP>=2"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Total experiments: $total_combinations"
echo "Output directory: $CROSSPROD_DIR"
echo "============================================"

# Create the main output directory
mkdir -p "$CROSSPROD_DIR"
mkdir -p "$CONFIGS_DIR"

# Iterate through all combinations
for combo in "${COMBINATIONS[@]}"; do
    current=$((current + 1))
    
    IFS=',' read -r num_gpus batch_size isl osl pp tp cp ep attn_dp <<< "$combo"
    
    attn_dp_str=$( [ "$attn_dp" -eq 1 ] && echo "true" || echo "false" )
    echo ""
    echo "============================================"
    echo "[CrossProd] Processing experiment $current/$total_combinations"
    echo "  Config: GPUs=$num_gpus, BS=$batch_size, ISL=$isl, OSL=$osl"
    echo "          PP=$pp, TP=$tp, CP=$cp, EP=$ep, AttnDP=$attn_dp_str"
    echo "============================================"

    # Generate log directory path
    log_dir=$(get_log_dir "$batch_size" "$pp" "$tp" "$cp" "$ep" "$attn_dp")

    # Update config with current parameters
    update_config "$num_gpus" "$batch_size" "$isl" "$osl" "$pp" "$tp" "$cp" "$ep" "$attn_dp"

    # Save config for later review
    save_config "$num_gpus" "$batch_size" "$isl" "$osl" "$pp" "$tp" "$cp" "$ep" "$attn_dp"

    # Submit the job with log directory
    submit_job "$log_dir"

    # Delay between submissions
    sleep 5
done

echo ""
echo "============================================"
echo "Cross-Product benchmark complete!"
echo "Submitted $total_combinations experiments"
echo "Results directory: $CROSSPROD_DIR"
echo "  - Logs organized by batch size: bs1/, bs2/, ..., bs3072/"
echo "  - Saved configs: $CONFIGS_DIR"
echo "============================================"
