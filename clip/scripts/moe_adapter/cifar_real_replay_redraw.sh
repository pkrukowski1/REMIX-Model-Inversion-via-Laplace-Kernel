#!/bin/bash
#SBATCH --job-name=moe_adapter_cifar_real_replay_redraw
#SBATCH --qos=normal
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal/logs/%x_%A_%a.out

# MoE-Adapter (ViT-B/16) on CIFAR-100, 10 tasks of 10 classes, replaying real training samples instead of
# inverted ones (5 per class, as PMI); the samples of every seen class are redrawn after each task.
# One array task per seed. Submit from the CLIP code directory:
#   sbatch scripts/moe_adapter/cifar_real_replay_redraw.sh

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEEDS=(1993 1994 1995)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:?Submit with sbatch}]}
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal}"
RUN_PATH="$BASE_PATH/cifar/real_replay_fresh/seed_$SEED"

cd "$SLURM_SUBMIT_DIR"
[[ -f main.py ]] || { echo "Submit from the CLIP code directory"; exit 1; }

if [[ -f "$RUN_PATH/results.json" ]]; then
    echo "Already finished: $RUN_PATH"
    exit 0
fi
rm -rf "$RUN_PATH"
mkdir -p "$RUN_PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

echo "Seed $SEED -> $RUN_PATH"
python3 -u main.py \
    --config=./configs/moe_adapter/cf100.yaml \
    --seed "$SEED" \
    --prefix real_replay_fresh \
    --local_path="$RUN_PATH" \
    --upper_bound real_replay_fresh
