#!/bin/bash
#SBATCH --job-name=codaprompt_cifar
#SBATCH --qos=normal
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt/logs/%x_%A_%a.out

# CODA-Prompt (ViT-B/16) on CIFAR-100, 10 tasks, as published (rehearsal-free): one array task per seed.
# Create the log folder once, then submit from the CLIP code directory:
#   mkdir -p /raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt/logs
#   sbatch scripts/codaprompt/cifar.sh
# REMIX + CODA-Prompt: scripts/codaprompt/remix_tune.sh, then scripts/codaprompt/remix_final.sh.
# Finished runs (results.json present) are skipped.

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

read -r -a SEEDS <<< "${SEEDS:-1993 1994 1995}"
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt}"

IDX="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
if (( IDX >= ${#SEEDS[@]} )); then
    echo "Array index $IDX >= number of seeds ${#SEEDS[@]}; nothing to do."
    exit 0
fi
SEED="${SEEDS[$IDX]}"
RUN_PATH="$BASE_PATH/cifar/coda/seed_$SEED"
echo "[$IDX/${#SEEDS[@]}] CODA-Prompt, seed $SEED -> $RUN_PATH"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u main.py --config=./configs/codaprompt/cf100.yaml --seed $SEED --prefix coda --local_path=$RUN_PATH"
    exit 0
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f main.py ]] || { echo "Submit from the CLIP code directory"; exit 1; }
if [[ -f "$RUN_PATH/results.json" ]]; then
    echo "Already finished: $RUN_PATH"
    exit 0
fi
rm -rf "$RUN_PATH"
mkdir -p "$RUN_PATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

python3 -u main.py \
    --config=./configs/codaprompt/cf100.yaml \
    --seed "$SEED" \
    --prefix coda \
    --local_path="$RUN_PATH"

[[ -f "$RUN_PATH/results.json" ]] || { echo "Run ended without results.json"; exit 1; }
