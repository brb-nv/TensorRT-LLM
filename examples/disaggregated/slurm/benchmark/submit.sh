#!/bin/bash

partition=batch
account=coreai_horizon_dilations
job_name=helix-benchmark-test-ctxtp4-gentp2cp2
container_image=urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.08-py3-aarch64-ubuntu24.04-trt10.13.2.6-skip-tritondevel-202509112230-7568
mounts=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations:/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations  # e.g. /mnt/data:/mnt/data
workdir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/bbuddharaju/TensorRT-LLM/examples/disaggregated/slurm/benchmark/  # Path to disaggr_torch.slurm
model_dir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data/models/DeepSeek-V3-Lite/fp8  # Path to the model checkpoint
repo_dir=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/bbuddharaju/TensorRT-LLM  # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used

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
