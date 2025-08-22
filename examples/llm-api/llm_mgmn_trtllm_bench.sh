#!/bin/bash
#SBATCH -A coreai_comparch_trtllm
#SBATCH -p batch
#SBATCH -t 01:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH -o logs/trtllm-bench.out
#SBATCH -e logs/trtllm-bench.err
#SBATCH -J trtllm-bench

### :title Run trtllm-bench with pytorch backend on Slurm
### :order 1
### :section Slurm

# NOTE, this feature is experimental and may not work on all systems.
# The trtllm-llmapi-launch is a script that launches the LLM-API code on
# Slurm-like systems, and can support multi-node and multi-GPU setups.

# Note that, the number of MPI processes should be the same as the model world
# size. e.g. For tensor_parallel_size=16, you may use 2 nodes with 8 gpus for
# each, or 4 nodes with 4 gpus for each or other combinations.

# This docker image should have tensorrt_llm installed, or you need to install
# it in the task.

# The following variables are expected to be set in the environment:
# You can set them via --export in the srun/sbatch command.
#   CONTAINER_IMAGE: the docker image to use, you'd better install tensorrt_llm in it, or install it in the task.
#   MOUNT_DIR: the directory to mount in the container
#   MOUNT_DEST: the destination directory in the container
#   WORKDIR: the working directory in the container
#   SOURCE_ROOT: the path to the TensorRT-LLM source
#   PROLOGUE: the prologue to run before the script
#   LOCAL_MODEL: the local model directory to use, NOTE: downloading from HF is
#      not supported in Slurm mode, you need to download the model and put it in
#      the LOCAL_MODEL directory.

# Docker container configuration
export CONTAINER_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.06-py3-x86_64-ubuntu24.04-trt10.11.0.33-skip-tritondevel-202508110900-6715"
export MOUNT_DIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM"
export MOUNT_DEST="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM"
export WORKDIR="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM"

# TensorRT-LLM source and model paths
export SOURCE_ROOT="/lustre/fsw/coreai_comparch_trtllm/bbuddharaju/TensorRT-LLM"
export LOCAL_MODEL="/lustre/fsw/coreai_comparch_trtllm/llm_data/llm-models/llama-3.3-models/Llama-3.3-70B-Instruct"
export MODEL_NAME="Llama-3.3-70B-Instruct"

# Optional variables
export PROLOGUE="cd $WORKDIR && pip install -e . && echo 'TRTLLM installation complete!'"
export EXTRA_ARGS=""                           # Additional arguments for trtllm-bench

export prepare_dataset="$SOURCE_ROOT/benchmarks/cpp/prepare_dataset.py"
export data_path="$WORKDIR/token-norm-dist.txt"

echo "Preparing dataset..."
srun -l \
    -N 1 \
    -n 1 \
    --container-image=${CONTAINER_IMAGE} \
    --container-name="prepare-name" \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL \
    --mpi=pmix \
    bash -c "
        $PROLOGUE
        python3 $prepare_dataset \
            --tokenizer=$LOCAL_MODEL \
            --stdout token-norm-dist \
            --num-requests=100 \
            --input-mean=128 \
            --output-mean=128 \
            --input-stdev=0 \
            --output-stdev=0 > $data_path
    "

echo "Running benchmark..."
# Just launch trtllm-bench job with trtllm-llmapi-launch command.

srun -l \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${MOUNT_DIR}:${MOUNT_DEST} \
    --container-workdir=${WORKDIR} \
    --export=ALL,PYTHONPATH=${SOURCE_ROOT} \
    --mpi=pmix \
    bash -c "
        set -ex
        $PROLOGUE
        export PATH=$PATH:~/.local/bin

        # This is optional
        cat > /tmp/pytorch_extra_args.txt << EOF
print_iter_log: true
enable_attention_dp: false
EOF

        # launch the benchmark
        trtllm-llmapi-launch \
         trtllm-bench \
            --model $MODEL_NAME \
            --model_path $LOCAL_MODEL \
            throughput \
            --dataset $data_path \
            --backend pytorch \
            --tp 16 \
            --extra_llm_api_options /tmp/pytorch_extra_args.txt \
            $EXTRA_ARGS
    "
