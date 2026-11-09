#!/bin/bash
# Covariance ablation (ResNet-32, CIFAR-100, 10 tasks), stage 2: the best value of each tuned variant (final
# accuracy on the first seed, tools/select_cov_config.py) on the remaining seeds. Submit after stage 1
# (scripts/submit_cov_ablation.sh) has finished; nothing is submitted while tuning runs are missing.
# Run from the resnet32 directory:
#
#   bash scripts/submit_cov_ablation_final.sh            # 4 tuned variants x 2 seeds = 8 runs
#   CONDA_ENV=laplace_kernel_in_cl bash scripts/submit_cov_ablation_final.sh   # extra args go to sbatch
#
# Use the same DATASETS / SEEDS / *_GRID values as in stage 1. Finished runs are skipped. Each run gets the
# smallest A100 (MIG slice or whole card) with at least GPU_GB GB (default 24); GRES=gpu:<type>:1 overrides.
# Summarise with: python tools/collect_cov_ablation.py --root "$BASE_PATH"

set -eo pipefail

export DATASETS="${DATASETS:-cifar100}"
export VARIANTS="${VARIANTS:-pmi diag lcm lcm_fixed lowrank blockdiag sparse shrinkage}"
export SEEDS="${SEEDS:-1 2 3}"
export LOWRANK_GRID="${LOWRANK_GRID:-2 4 8}"
export BLOCK_GRID="${BLOCK_GRID:-4 8 16}"
export SPARSE_GRID="${SPARSE_GRID:-1 2 4}"
export SHRINK_GRID="${SHRINK_GRID:-oas 0.1 0.3}"
export SELECT_METRIC="${SELECT_METRIC:-final}"
export BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation}"
export CODE_DIR="${CODE_DIR:-$(pwd)}"
MAX_PARALLEL="${MAX_PARALLEL:-14}"
GPU_GB="${GPU_GB:-24}"

[[ -f "$CODE_DIR/main_cov_ablation.py" ]] || { echo "Run this from the resnet32 directory (or set CODE_DIR)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/cov_ablation_10_task.sh"
source "$SCRIPT_DIR/gpu.sh"

N_RUNS=$(STAGE=final COUNT_ONLY=1 bash "$JOB_SCRIPT")
(( N_RUNS > 0 )) || { echo "No variant with a hyperparameter grid in VARIANTS"; exit 1; }
echo "Selected values (best $SELECT_METRIC accuracy on seed ${SEEDS%% *}):"
if ! STAGE=final SELECT_ONLY=1 bash "$JOB_SCRIPT"; then
    echo "Some tuning runs have not finished (listed above); submit again when stage 1 is done."
    exit 1
fi
echo "Stage 2 (final): $N_RUNS runs (max $MAX_PARALLEL at once)"
STAGE=final LIST_ONLY=1 bash "$JOB_SCRIPT"
if [[ "${LIST_ONLY:-0}" == "1" ]]; then
    exit 0
fi
GPU=$(gpu_options "$GPU_GB") || exit 1
read -r -a GPU_OPTS <<< "$GPU"
echo "GPU per run: $GPU (at least $GPU_GB GB)"

mkdir -p "$BASE_PATH/logs"
sbatch --export=ALL,STAGE=final \
       --job-name=cov_ablation_final \
       --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
       --output="$BASE_PATH/logs/%x_%A_%a.out" \
       "${GPU_OPTS[@]}" \
       "$@" \
       "$JOB_SCRIPT"
