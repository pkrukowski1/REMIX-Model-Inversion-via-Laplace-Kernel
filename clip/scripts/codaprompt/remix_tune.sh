#!/bin/bash
#SBATCH --job-name=codaprompt_remix_tune
#SBATCH --qos=normal
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt/logs/%x_%A_%a.out

# REMIX + CODA-Prompt (ViT-B/16) on CIFAR-100, 10 tasks: grid over alpha_stat (alpha_frob) on seed 1993,
# one array task per value (MoE-Adapter used 0.001 on CIFAR-100 and 0.1 on CUB).
# Create the log folder once, then submit from the CLIP code directory:
#   mkdir -p /raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt/logs
#   sbatch scripts/codaprompt/remix_tune.sh
# When all runs have finished: sbatch scripts/codaprompt/remix_final.sh (best value on seeds 1994 and 1995).
# Finished runs (results.json present) are skipped.

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

read -r -a ALPHAS <<< "${ALPHAS:-0.0001 0.001 0.01 0.1}"   # keep in sync with remix_final.sh
SEED="${TUNE_SEED:-1993}"
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt}"

IDX="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
if (( IDX >= ${#ALPHAS[@]} )); then
    echo "Array index $IDX >= number of values ${#ALPHAS[@]}; nothing to do."
    exit 0
fi
ALPHA="${ALPHAS[$IDX]}"
TAG="tune_remix_a$ALPHA"
RUN_PATH="$BASE_PATH/cifar/$TAG/seed_$SEED"
echo "[$IDX/${#ALPHAS[@]}] REMIX + CODA-Prompt, alpha_stat=$ALPHA, seed $SEED -> $RUN_PATH"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u main.py --config=./configs/codaprompt/cf100_inv.yaml --seed $SEED --prefix $TAG --local_path=$RUN_PATH --alpha_frob $ALPHA"
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
    --config=./configs/codaprompt/cf100_inv.yaml \
    --seed "$SEED" \
    --prefix "$TAG" \
    --local_path="$RUN_PATH" \
    --alpha_frob "$ALPHA"

[[ -f "$RUN_PATH/results.json" ]] || { echo "Run ended without results.json"; exit 1; }
