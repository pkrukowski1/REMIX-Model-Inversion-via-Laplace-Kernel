#!/bin/bash
# MoE-Adapter, PMI vs REMIX on CIFAR-100 with other task splits, 3 seeds: 5 tasks of 20 classes and 20 tasks
# of 5 classes (the 10-task runs come from submit_rebuttal.sh). One SLURM array per split.
# Run from the CLIP code directory:
#
#   bash scripts/moe_adapter/submit_task_splits.sh              # 2 splits x 2 methods x 3 seeds = 12 runs
#   SPLITS=20 bash scripts/moe_adapter/submit_task_splits.sh    # only the 5-task split
#
# Results go to <BASE_PATH>/cifar_inc<classes per task>/. Any extra arguments are passed to sbatch.
# Each run gets the smallest A100 (MIG slice or whole card) with at least GPU_GB GB (default 16);
# GRES=gpu:<type>:1 overrides.
# Finished runs are skipped. Summarise with: python tools/collect_rebuttal.py

set -eo pipefail

export DATASETS="${DATASETS:-cifar}"
export METHODS="${METHODS:-pmi remix}"
export SEEDS="${SEEDS:-1993 1994 1995}"
export BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal}"
export CODE_DIR="${CODE_DIR:-$(pwd)}"
SPLITS="${SPLITS:-20 5}"   # classes per task (init_cls = increment)
MAX_PARALLEL="${MAX_PARALLEL:-12}"
GPU_GB="${GPU_GB:-16}"

read -r -a DS <<< "$DATASETS"
read -r -a MS <<< "$METHODS"
read -r -a SS <<< "$SEEDS"
N_RUNS=$(( ${#DS[@]} * ${#MS[@]} * ${#SS[@]} ))
(( N_RUNS > 0 )) || { echo "Nothing to submit"; exit 1; }

mkdir -p "$BASE_PATH/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu.sh"
GPU=$(gpu_options "$GPU_GB") || exit 1
read -r -a GPU_OPTS <<< "$GPU"
echo "GPU per run: $GPU (at least $GPU_GB GB)"

for increment in $SPLITS; do
    echo "Split with $increment classes per task: $N_RUNS runs (max $MAX_PARALLEL at once)"
    echo "  datasets: $DATASETS   methods: $METHODS   seeds: $SEEDS"
    echo "  results:  $BASE_PATH/<dataset>_inc$increment/<method>/seed_<seed>/results.json"
    sbatch --export=ALL,INCREMENT="$increment" \
           --job-name="remix_split_inc$increment" \
           --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
           --output="$BASE_PATH/logs/%x_%A_%a.out" \
           "${GPU_OPTS[@]}" \
           "$@" \
           "$SCRIPT_DIR/rebuttal_array.sh"
done
