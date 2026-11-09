#!/bin/bash
# MoE-Adapter, PMI vs REMIX on CIFAR-100 and CUB (10 tasks, as in Table 2), 5 seeds: standard deviations
# and paired significance tests. Run from the CLIP code directory:
#
#   bash scripts/moe_adapter/submit_rebuttal.sh                 # 2 datasets x 2 methods x 5 seeds = 20 runs
#   DATASETS=cifar bash scripts/moe_adapter/submit_rebuttal.sh -p rtx4090_batch --qos=batch   # extra args go to sbatch
#
# The upper bounds and the other task splits have their own scripts (submit_upper_bounds.sh,
# submit_task_splits.sh). Finished runs (results.json present) are skipped, so the same command can be
# resubmitted after failures. Summarise with: python tools/collect_rebuttal.py

set -eo pipefail

export DATASETS="${DATASETS:-cifar cub}"
export METHODS="${METHODS:-pmi remix}"
export SEEDS="${SEEDS:-1993 1994 1995 1996 1997}"
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

echo "PMI vs REMIX (10 tasks): $N_RUNS runs (max $MAX_PARALLEL at once)"
echo "  datasets: $DATASETS"
echo "  methods:  $METHODS"
echo "  seeds:    $SEEDS"
echo "  results:  $BASE_PATH/<dataset>/<method>/seed_<seed>/results.json"

sbatch --export=ALL \
       --job-name=remix_significance \
       --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
       --output="$BASE_PATH/logs/%x_%A_%a.out" \
       "$@" \
       "$SCRIPT_DIR/rebuttal_array.sh"
