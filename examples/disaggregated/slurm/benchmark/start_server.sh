#!/bin/bash
set -u
set -e
set -x

num_ctx_servers=$1
num_gen_servers=$2
work_dir=$3
script_dir=$4

export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1

python3 ${script_dir}/gen_server_config.py \
    --num_ctx_servers ${num_ctx_servers} \
    --num_gen_servers ${num_gen_servers} \
    --work_dir ${work_dir}
echo "server config generated to ${work_dir}/server_config.yaml"

trtllm-serve disaggregated -c ${work_dir}/server_config.yaml -t 1800 -r 1200
