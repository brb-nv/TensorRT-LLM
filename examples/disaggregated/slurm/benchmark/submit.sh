#!/bin/bash

partition="" # slurm partition
account="" # slurm account
job_name="" # slurm job name
container_image="" # /path/to/image.sqsh
mounts=""  # e.g. /mnt/data:/mnt/data
workdir=""  # Path to this directory
model_dir=""  # Path to the model checkpoint
repo_dir=""  # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used

ntasks_per_node=4 # 4 GPUs per GB200 node
total_node_num=2
ntasks=$((total_node_num * ntasks_per_node))

concurrency=8
isl=1024
osl=1024
multi_round=10
streaming=true
benchmark_mode=e2e
build_wheel=true

args=(
    1 4 1 1 4 4480 false "0.75"   # Context - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    1 2 1 2 1024 1024 false "0.8" # Generation - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    0 0                        # Other arguments
    $concurrency               # Benchmarking arguments
    $isl
    $osl
    $multi_round
    $streaming
    $container_image           # User specific arguments
    $mounts
    $workdir
    $model_dir
    $benchmark_mode
    $repo_dir
    $build_wheel
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
