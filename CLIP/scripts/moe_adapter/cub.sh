#!/bin/bash
#SBATCH --job-name=cl_laplace_kernel_moe_adapter_cub_10x20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_PATH="/shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/cub"
LOG_PATH="$BASE_PATH/logs"

mkdir -p "$BASE_PATH"
mkdir -p "$LOG_PATH"

cd "$HOME/LaplaceKernelInContinualLearning/CLIP"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

ALPHA_GMRF_VALUES=(0.0001 0.0005 0.001 0.002)

for ALPHA in "${ALPHA_GMRF_VALUES[@]}"; do
    echo "Running alpha_gmrf=$ALPHA"

    RUN_PATH="$BASE_PATH/alpha_${ALPHA}"
    mkdir -p "$RUN_PATH"

    python3 -u main.py \
        --config=./configs/moe_adapter/cub.yaml \
        --seed 1993 \
        --local_path="$RUN_PATH" \
        --alpha_gmrf="$ALPHA"

done