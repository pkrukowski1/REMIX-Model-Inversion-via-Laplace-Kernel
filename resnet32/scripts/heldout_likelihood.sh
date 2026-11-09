#!/bin/bash
#SBATCH --job-name=heldout_likelihood
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation/logs/%x_%A_%a.out
#
# Held-out likelihood of ResNet-32 CIFAR-100 features under the covariance families of the ablation
# (tools/heldout_likelihood.py). One array task = one (backbone seed, part); part = channel (every BatchNorm
# input) or stage_1 / stage_2 / stage_3 (flattened stage outputs). The backbones are the final models of the
# PMI ablation runs. Submit with scripts/submit_heldout_likelihood.sh, from the resnet32 directory.

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation}"
CKPT_TAG="${CKPT_TAG:-pmi}"
CKPT_SEEDS="${CKPT_SEEDS:-1 2 3}"
PARTS="${PARTS:-channel stage_1 stage_2 stage_3}"
OUT_DIR="${OUT_DIR:-$BASE_PATH/heldout_likelihood}"
DATA_PATH="${DATA_PATH:-/raid/NFS_SHARE/datasets}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

read -r -a SS <<< "$CKPT_SEEDS"
read -r -a PS <<< "$PARTS"
N_RUNS=$(( ${#SS[@]} * ${#PS[@]} ))
if [[ "${COUNT_ONLY:-0}" == "1" ]]; then
    echo "$N_RUNS"
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:?Run as a SLURM array job (see scripts/submit_heldout_likelihood.sh)}"
if (( IDX >= N_RUNS )); then
    echo "Array index $IDX >= number of runs $N_RUNS; nothing to do."
    exit 0
fi
SEED="${SS[$(( IDX / ${#PS[@]} ))]}"
PART="${PS[$(( IDX % ${#PS[@]} ))]}"
CKPT="$BASE_PATH/cifar100_10task/$CKPT_TAG/seed_$SEED/backbone_9.pkl"
echo "[$IDX/$N_RUNS] backbone seed $SEED, part $PART -> $OUT_DIR/parts/seed_${SEED}_${PART}.json"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u tools/heldout_likelihood.py --ckpt $CKPT --seed $SEED --parts $PART --data_path $DATA_PATH --out_dir $OUT_DIR"
    exit 0
fi

cd "$CODE_DIR"
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
mkdir -p "$OUT_DIR"
python3 -u tools/heldout_likelihood.py --ckpt "$CKPT" --seed "$SEED" --parts "$PART" \
    --data_path "$DATA_PATH" --out_dir "$OUT_DIR" "$@"
