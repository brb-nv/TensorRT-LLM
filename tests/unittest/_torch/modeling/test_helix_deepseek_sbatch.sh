#!/bin/bash
TP=${1:-8}
KVP=${2:-1}
EP=${3:-2}
DENSE=${4:-0}
dense_arg=""
if (( DENSE == 1 )); then
  dense_arg="--dense"
fi
world_size=$((TP * KVP))
if (( world_size % EP != 0 )); then
  echo "World size $world_size must be a multiple of EP $EP"
  exit 1
fi
gpus_per_node=4
NODES=$(((world_size + gpus_per_node - 1) / gpus_per_node))
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
#SBATCH --segment=${NODES}
#SBATCH --ntasks=${gpus}
#SBATCH --ntasks-per-node=${gpus_per_node}

cleanup_on_failure() {
    echo "Error: \$1"
    scancel \${SLURM_JOB_ID}
    exit 1
}

set -x

export WORKDIR="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/helix"
export CONTAINER_NAME="tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568"
export CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/containers/tllm_pyt2508_py3_aarch64_trt10.13.2.6_202509112230_7568_build7e31a9b56d10e83fe1e654dbccc10e7e5bbad0f0.sqsh"
export CONTAINER_MOUNT="/lustre/:/lustre/"
logdir=\${WORKDIR}/slurm-\${SLURM_JOB_ID}
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

export TLLM_LOG_LEVEL="INFO" # DEBUG is verbose
export TRTLLM_ENABLE_PDL=1
export LLM_MODELS_ROOT=/lustre/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/mjoux/data/models
export LD_LIBRARY_PATH=/workspace/TensorRT-LLM/cpp/build/tensorrt_llm/executor/cache_transmission/ucx_utils:\${LD_LIBRARY_PATH}

echo "====== Baseline ========"
srun --mpi pmix -N ${NODES} --ntasks-per-node ${gpus_per_node} \
  --container-env=MASTER_ADDR,MASTER_PORT \
  --container-name=\${CONTAINER_NAME} \
  --container-mounts=\${CONTAINER_MOUNT} \
  python3 \${trtllm_repo}/tests/unittest/_torch/modeling/test_helix_deepseek.py --type v3 --tp ${TP} --kvp ${KVP} --ep ${EP} ${dense_arg} \
    &> \${full_logdir}/benchmark.log 2>&1
EOF
