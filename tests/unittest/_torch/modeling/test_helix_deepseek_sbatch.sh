#!/bin/bash
NODES=${1:-4}
gpus_per_node=4
gpus=$((NODES * gpus_per_node))
sbatch <<EOF
#!/bin/bash
#SBATCH --nodes=${NODES}
#SBATCH --partition=batch
#SBATCH --account=coreai_horizon_dilations
#SBATCH --time=01:00:00
#SBATCH --job-name="coreai_horizon_dilations-trtllm-helix-test-layer"
#SBATCH --comment=fact_off
#SBATCH --gres=gpu:${gpus_per_node}

cleanup_on_failure() {
    echo "Error: \$1"
    scancel \${SLURM_JOB_ID}
    exit 1
}

set -x

export WORKDIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/helix"
export CONTAINER_NAME="tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568"
export CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/containers/tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568.sqsh"
export CONTAINER_MOUNT="/lustre/:/lustre/"
logdir=\${WORKDIR}/slurm-\${SLURM_JOB_ID}-N${NODES}
mkdir -p \${logdir}
full_logdir=\${logdir}
trtllm_repo=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/TensorRT-LLM

echo "Starting container..."
if ! srun -l --container-image=\${CONTAINER_IMAGE} \
        --container-name=\${CONTAINER_NAME} \
        --container-mounts=\${CONTAINER_MOUNT} \
        --mpi=pmix \
        echo "Container up." &> \${full_logdir}/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check \${full_logdir}/container_launch.log"
fi

if [ -d "\${trtllm_repo}" ]; then
    echo "Installing TensorRT-LLM from \${trtllm_repo}..."
    TRT_LLM_GIT_COMMIT=\$(git -C \${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: \${TRT_LLM_GIT_COMMIT}"

    echo "Building TensorRT-LLM wheel on one node..."
    build_command="python3 \${trtllm_repo}/scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --use_ccache --cuda_architectures \"\${cuda_architectures}\""
    if ! srun --container-name=\${CONTAINER_NAME} \
        --container-mounts=\${CONTAINER_MOUNT} \
        --mpi=pmix --overlap -N 1 --ntasks-per-node=1 \
        bash -c "cd \${trtllm_repo} && \${build_command}" \
        &> \${full_logdir}/build.log; then
        cleanup_on_failure "TensorRT-LLM build failed. Check \${full_logdir}/build.log for details"
    fi

    echo "Installing TensorRT-LLM..."
    if ! srun --container-name=\${CONTAINER_NAME} \
        --container-mounts=\${CONTAINER_MOUNT} \
        --mpi=pmix --overlap -N \$SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd \${trtllm_repo} && pip install -e ." \
        &> \${full_logdir}/install.log; then
        cleanup_on_failure "TensorRT-LLM installation failed. Check \${full_logdir}/install.log for details"
    fi
    echo "TensorRT-LLM installation completed successfully"
fi

export TLLM_LOG_LEVEL="INFO" # DEBUG is verbose
export TRTLLM_ENABLE_PDL=1
export LLM_MODELS_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data/models
export LD_LIBRARY_PATH=/workspace/TensorRT-LLM/cpp/build/tensorrt_llm/executor/cache_transmission/ucx_utils:\${LD_LIBRARY_PATH}

echo "====== Baseline ========"
srun --mpi pmix -N ${NODES} --ntasks-per-node ${gpus_per_node} \
  --container-env=MASTER_ADDR,MASTER_PORT \
  --container-name=\${CONTAINER_NAME} \
  --container-mounts=\${CONTAINER_MOUNT} \
  python3 \${trtllm_repo}/tests/unittest/_torch/modeling/test_helix_deepseek.py \
    &> \${full_logdir}/benchmark.log 2>&1
EOF
