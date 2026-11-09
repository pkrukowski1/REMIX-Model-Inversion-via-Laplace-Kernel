#!/bin/bash
# MoE-Adapter upper bounds on CIFAR-100 and CUB, 3 seeds: joint training on all classes (one
# task) and real replay (same pipeline and memory budget, real samples instead of inverted ones).
# Run from the CLIP code directory:
#
#   bash scripts/moe_adapter/submit_upper_bounds.sh             # 2 datasets x 2 bounds x 3 seeds = 12 runs
#   SEEDS="1993 1994 1995 1996 1997" bash scripts/moe_adapter/submit_upper_bounds.sh   # 5 seeds, like PMI vs REMIX
#
# Any extra arguments are passed to sbatch. Finished runs (results.json present) are skipped.
# Summarise with: python tools/collect_rebuttal.py

set -eo pipefail

export DATASETS="${DATASETS:-cifar cub}"
export METHODS="${METHODS:-joint real_replay}"
export SEEDS="${SEEDS:-1993 1994 1995}"
export BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal}"
export CODE_DIR="${CODE_DIR:-$(pwd)}"
export INCREMENT=""   # task split of the configs (10 tasks)
MAX_PARALLEL="${MAX_PARALLEL:-12}"

read -r -a DS <<< "$DATASETS"
read -r -a MS <<< "$METHODS"
read -r -a SS <<< "$SEEDS"
N_RUNS=$(( ${#DS[@]} * ${#MS[@]} * ${#SS[@]} ))
(( N_RUNS > 0 )) || { echo "Nothing to submit"; exit 1; }

mkdir -p "$BASE_PATH/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Upper bounds: $N_RUNS runs (max $MAX_PARALLEL at once)"
echo "  datasets: $DATASETS"
echo "  bounds:   $METHODS"
echo "  seeds:    $SEEDS"
echo "  results:  $BASE_PATH/<dataset>/<bound>/seed_<seed>/results.json"

sbatch --export=ALL \
       --job-name=remix_upper_bounds \
       --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
       --output="$BASE_PATH/logs/%x_%A_%a.out" \
       "$@" \
       "$SCRIPT_DIR/rebuttal_array.sh"
