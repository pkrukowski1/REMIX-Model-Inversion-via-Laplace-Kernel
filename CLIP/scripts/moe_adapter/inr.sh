#!/bin/bash
#SBATCH --job-name=cl_laplace_kernel_moe_adapter_inr_10x20
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090
#SBATCH --array=0-6
#SBATCH --output=/shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/inr/logs/array_%A_task_%a.out

export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_PATH="/shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/inr"
LOG_PATH="$BASE_PATH/logs"

mkdir -p "$BASE_PATH"
mkdir -p "$LOG_PATH"

cd "$HOME/LaplaceKernelInContinualLearning/CLIP"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

# Dense logarithmic grid (7 values)
ALPHA_GMRF_VALUES=(0.0001 0.0005 0.001 0.002 0.005 0.01 0.05)

# Extract the specific value for THIS specific task using the SLURM array ID
ALPHA=${ALPHA_GMRF_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running SLURM Array Task ID: $SLURM_ARRAY_TASK_ID with alpha_gmrf=$ALPHA"

RUN_PATH="$BASE_PATH/alpha_${ALPHA}"
mkdir -p "$RUN_PATH"

python3 -u main.py \
    --config=./configs/moe_adapter/inr.yaml \
    --seed 1993 \
    --local_path="$RUN_PATH" \
    --alpha_gmrf="$ALPHA"