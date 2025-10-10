#!/bin/bash

ctx_tp_size=8
ctx_pp_size=1
ctx_cp_size=1
ctx_chunked_prefill=false
gen_tp_size=8
gen_pp_size=1
gen_cp_size=2

partition=batch
account=coreai_horizon_dilations
job_name=coreai_horizon_dilations-helix_benchmark_test_ctxtp${ctx_tp_size}cp${ctx_cp_size}$(if [ "${ctx_chunked_prefill}" = "true" ]; then echo "chunked"; fi)_gentp${gen_tp_size}cp${gen_cp_size}
container_image=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/containers/tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568.sqsh
# e.g. /mnt/data:/mnt/data
mounts=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations:/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations
# Path to disaggr_torch.slurm
workdir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/TensorRT-LLM/examples/disaggregated/slurm/benchmark/
# Path to the model checkpoint
model_dir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data/models/DeepSeek-R1/DeepSeek-R1-FP4
# Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used
repo_dir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/TensorRT-LLM
# Path to the data directory
data_dir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data

ntasks_per_node=4 # 4 GPUs per GB200 node

batch=1
isl=262144
osl=1024
concurrency=8
multi_round=1
streaming=true
benchmark_mode=e2e
build_wheel=true
cuda_architectures="100a-real"
ctx_max_tokens=$((batch * (isl + 10)))
gen_max_tokens=$((batch * (isl + osl + 10)))
# note: this also works for 32 tokens/block
tokens_per_block=64
transceiver_factor=2
transceiver_blocks=$(((ctx_max_tokens * transceiver_factor + tokens_per_block - 1) / tokens_per_block))
cache_transceiver_max_num_tokens=$((tokens_per_block * transceiver_blocks))

ctx_nodes_num=$(((ctx_tp_size * ctx_pp_size * ctx_cp_size + ntasks_per_node - 1) / ntasks_per_node))
gen_nodes_num=$(((gen_tp_size * gen_pp_size * gen_cp_size + ntasks_per_node - 1) / ntasks_per_node))
total_node_num=$((ctx_nodes_num + gen_nodes_num))
ntasks=$((total_node_num * ntasks_per_node))

args=(
    # Context - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    1 $ctx_tp_size $ctx_pp_size $ctx_cp_size $batch $ctx_max_tokens false "0.25"
    # Generation - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    1 $gen_tp_size $gen_pp_size $gen_cp_size $batch $gen_max_tokens false "0.25"
    # Other arguments - [eplb_num_slots, mtp_size]
    0 0
    # Benchmarking arguments
    $concurrency
    $isl
    $osl
    $multi_round
    $streaming
    # User specific arguments
    $container_image
    $mounts
    $workdir
    $model_dir
    $benchmark_mode
    $repo_dir
    $build_wheel
    $cuda_architectures
    $data_dir
    $cache_transceiver_max_num_tokens
)

# This command starts a job with 8 nodes, 32 GPUs in total.
# The server will include 4 context workers with DEP4, and 1 generation worker with DEP8.
# `--segment` makes sure that all nodes are in the same NVLink domain
sbatch --nodes=${total_node_num} \
    --ntasks=${ntasks} \
    --ntasks-per-node=${ntasks_per_node} \
    --partition=${partition} \
    --account=${account} \
    --job-name=${job_name} \
    --gres=gpu:${ntasks_per_node} \
    --segment=${total_node_num} \
    ${workdir}/disaggr_torch.slurm "${args[@]}"
