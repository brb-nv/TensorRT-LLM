#!/bin/bash
# set these variables according to your cluster setup

export CONTAINER_IMAGE=""   # /path/to/image.sqsh
export CONTAINER_MOUNTS=""  # /path:/path
export WORK_DIR=""          # Path to this directory
export MODEL_DIR=""         # Path to the model checkpoint
export REPO_DIR=""          # Path to the repo to install TensorRT-LLM, if this is empty, the pre-installed version will be used
export DATA_DIR=""          # Path to the data directory
export SLURM_PARTITION=""   # slurm partition
export SLURM_ACCOUNT=""     # slurm account
export SLURM_JOB_NAME=""    # slurm job name
