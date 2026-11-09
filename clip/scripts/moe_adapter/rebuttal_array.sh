#!/bin/bash
#SBATCH --job-name=remix_rebuttal_moe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal/logs/%x_%A_%a.out
#
# One SLURM array task = one (dataset, method, seed) run of MoE-Adapter. Each experiment has its own submit
# script, which sizes the array (run from the CLIP code directory):
#   scripts/moe_adapter/submit_rebuttal.sh       PMI vs REMIX, 10 tasks, 5 seeds (std and significance tests)
#   scripts/moe_adapter/submit_upper_bounds.sh   joint training and real replay
#   scripts/moe_adapter/submit_task_splits.sh    PMI vs REMIX on CIFAR-100 with 5 and 20 tasks
#
# Methods:
#   pmi          alpha_frob = 0 (PMI baseline re-run in this codebase)
#   remix        alpha_frob = value fixed per dataset below (chosen before these runs; do not retune on test)
#   joint        upper bound: one task with all classes
#   real_replay  upper bound: same pipeline and memory budget, real samples instead of inverted ones
# The seed also sets the class order (shuffle: true), so PMI and REMIX with the same seed see the same tasks.
# INCREMENT=20 (or 5) runs another task split (init_cls = increment = INCREMENT) into <dataset>_inc<INCREMENT>.

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASETS="${DATASETS:-cifar cub}"   # ImageNet-R (inr) only if asked for explicitly
METHODS="${METHODS:-pmi remix}"
SEEDS="${SEEDS:-1993 1994 1995 1996 1997}"
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal}"
CODE_DIR="${CODE_DIR:-$HOME/LaplaceKernelInContinualLearning/CLIP}"
INCREMENT="${INCREMENT:-}"

declare -A CONFIG=(
    [cifar]=./configs/moe_adapter/cf100.yaml
    [cub]=./configs/moe_adapter/cub.yaml
    [inr]=./configs/moe_adapter/inr.yaml
)
# alpha_frob for REMIX. cifar/cub: values from Table 3 (selected on seed 1993 only).
# inr: fixed a priori (CUB's value) because the earlier ImageNet-R sweep used wrong class names.
declare -A REMIX_ALPHA=(
    [cifar]="${REMIX_ALPHA_CIFAR:-0.0001}"
    [cub]="${REMIX_ALPHA_CUB:-0.01}"
    [inr]="${REMIX_ALPHA_INR:-0.01}"
)

read -r -a DS <<< "$DATASETS"
read -r -a MS <<< "$METHODS"
read -r -a SS <<< "$SEEDS"
N_RUNS=$(( ${#DS[@]} * ${#MS[@]} * ${#SS[@]} ))

IDX="${SLURM_ARRAY_TASK_ID:?Run this as a SLURM array job (see submit_rebuttal.sh)}"
if (( IDX >= N_RUNS )); then
    echo "Array index $IDX >= number of runs $N_RUNS; nothing to do."
    exit 0
fi
SEED="${SS[$(( IDX % ${#SS[@]} ))]}"
METHOD="${MS[$(( (IDX / ${#SS[@]}) % ${#MS[@]} ))]}"
DATASET="${DS[$(( IDX / (${#SS[@]} * ${#MS[@]}) ))]}"

CFG="${CONFIG[$DATASET]:-}"
[[ -n "$CFG" ]] || { echo "Unknown dataset '$DATASET'"; exit 1; }

case "$METHOD" in
    pmi)
        TAG="pmi";                                  EXTRA=(--alpha_frob 0) ;;
    remix)
        TAG="remix_a${REMIX_ALPHA[$DATASET]}";      EXTRA=(--alpha_frob "${REMIX_ALPHA[$DATASET]}") ;;
    joint)
        TAG="joint";                                EXTRA=(--upper_bound joint) ;;
    real_replay)
        TAG="real_replay";                          EXTRA=(--upper_bound real_replay) ;;
    *)
        echo "Unknown method '$METHOD'"; exit 1 ;;
esac

SETTING="$DATASET"
SPLIT=()
if [[ -n "$INCREMENT" ]]; then
    if [[ "$METHOD" == joint ]]; then
        echo "The joint upper bound does not depend on the task split; skipping."
        exit 0
    fi
    SETTING="${DATASET}_inc${INCREMENT}"
    SPLIT=(--init_cls "$INCREMENT" --increment "$INCREMENT")
fi

RUN_PATH="$BASE_PATH/$SETTING/$TAG/seed_$SEED"
echo "[$IDX/$N_RUNS] dataset=$DATASET method=$METHOD seed=$SEED ${INCREMENT:+increment=$INCREMENT }-> $RUN_PATH"

# Check the index -> run mapping without running anything:
#   for i in $(seq 0 19); do DRY_RUN=1 SLURM_ARRAY_TASK_ID=$i bash scripts/moe_adapter/rebuttal_array.sh; done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u main.py --config=$CFG --seed $SEED --prefix $TAG --local_path=$RUN_PATH ${EXTRA[*]} ${SPLIT[*]}"
    exit 0
fi

if [[ -f "$RUN_PATH/results.json" ]]; then
    echo "Already finished ($RUN_PATH/results.json); skipping."
    exit 0
fi
# A leftover directory from a crashed/cancelled run must not be reused: the inversion code merges
# new feature statistics into any *_input_stats.pkl it finds in $RUN_PATH/buffer.
if [[ -d "$RUN_PATH" ]]; then
    case "$RUN_PATH" in
        "$BASE_PATH"/*/seed_*) echo "Removing incomplete run directory $RUN_PATH"; rm -rf -- "$RUN_PATH" ;;
        *) echo "Refusing to remove unexpected path $RUN_PATH"; exit 1 ;;
    esac
fi
mkdir -p "$RUN_PATH"

cd "$CODE_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

{
    echo "date: $(date -Is)"
    echo "host: $(hostname)  job: ${SLURM_ARRAY_JOB_ID:-?}_${IDX}"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "cmd: python3 -u main.py --config=$CFG --seed $SEED --prefix $TAG --local_path=$RUN_PATH ${EXTRA[*]} ${SPLIT[*]}"
} > "$RUN_PATH/run_info.txt"

python3 -u main.py \
    --config="$CFG" \
    --seed "$SEED" \
    --prefix "$TAG" \
    --local_path="$RUN_PATH" \
    "${EXTRA[@]}" \
    "${SPLIT[@]}"

[[ -f "$RUN_PATH/results.json" ]] || { echo "Run ended without results.json"; exit 1; }
