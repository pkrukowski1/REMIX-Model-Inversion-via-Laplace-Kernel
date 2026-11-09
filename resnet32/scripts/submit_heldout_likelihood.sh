#!/bin/bash
# Held-out likelihood analysis (ResNet-32 CIFAR-100 features) as one SLURM array: 3 backbones (final models of
# the PMI ablation runs, seeds 1-3) x 4 parts (per-channel features, flattened outputs of stages 1-3).
# Submit once those PMI runs have finished; nothing is submitted while a checkpoint is missing.
# Run from the resnet32 directory:
#
#   bash scripts/submit_heldout_likelihood.sh                    # 12 tasks
#   CONDA_ENV=laplace_kernel_in_cl bash scripts/submit_heldout_likelihood.sh   # extra args go to sbatch
#
# Each task gets the smallest A100 (MIG slice or whole card) with at least GPU_GB GB (default 8);
# GRES=gpu:<type>:1 overrides. Finished parts are skipped. Summarise with:
# python tools/collect_heldout_likelihood.py

set -eo pipefail

export BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation}"
export CKPT_TAG="${CKPT_TAG:-pmi}"
export CKPT_SEEDS="${CKPT_SEEDS:-1 2 3}"
export PARTS="${PARTS:-channel stage_1 stage_2 stage_3}"
export OUT_DIR="${OUT_DIR:-$BASE_PATH/heldout_likelihood}"
export DATA_PATH="${DATA_PATH:-/raid/NFS_SHARE/datasets}"
export CODE_DIR="${CODE_DIR:-$(pwd)}"
MAX_PARALLEL="${MAX_PARALLEL:-12}"
GPU_GB="${GPU_GB:-8}"

[[ -f "$CODE_DIR/tools/heldout_likelihood.py" ]] || { echo "Run this from the resnet32 directory (or set CODE_DIR)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/heldout_likelihood.sh"
source "$SCRIPT_DIR/gpu.sh"

missing=0
for s in $CKPT_SEEDS; do
    ckpt="$BASE_PATH/cifar100_10task/$CKPT_TAG/seed_$s/backbone_9.pkl"
    if [[ ! -f "$ckpt" ]]; then
        echo "Missing checkpoint: $ckpt"
        missing=1
    fi
done
if (( missing )); then
    echo "Submit again when those runs have finished."
    exit 1
fi

N_RUNS=$(COUNT_ONLY=1 bash "$JOB_SCRIPT")
(( N_RUNS > 0 )) || { echo "Nothing to submit"; exit 1; }
echo "Held-out likelihood: $N_RUNS tasks (max $MAX_PARALLEL at once)"
echo "  backbones: $CKPT_TAG, seeds $CKPT_SEEDS"
echo "  parts:     $PARTS"
echo "  results:   $OUT_DIR/parts/seed_<seed>_<part>.json"
GPU=$(gpu_options "$GPU_GB") || exit 1
read -r -a GPU_OPTS <<< "$GPU"
echo "  GPU:       $GPU (at least $GPU_GB GB)"

mkdir -p "$BASE_PATH/logs" "$OUT_DIR"
sbatch --export=ALL \
       --job-name=heldout_likelihood \
       --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
       --output="$BASE_PATH/logs/%x_%A_%a.out" \
       "${GPU_OPTS[@]}" \
       "$@" \
       "$JOB_SCRIPT"
