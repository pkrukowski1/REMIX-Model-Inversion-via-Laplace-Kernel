#!/bin/bash
# Covariance ablation (ResNet-32, CIFAR-100, 10 tasks), stage 1: every value of each tuned variant's grid on
# the first seed, and pmi, diag, lcm and lcm_fixed on all seeds. When these runs have finished, submit
# stage 2 with scripts/submit_cov_ablation_final.sh. Run from the resnet32 directory:
#
#   bash scripts/submit_cov_ablation.sh                  # 4 x 3 tuning runs + 4 variants x 3 seeds = 24 runs
#   LIST_ONLY=1 bash scripts/submit_cov_ablation.sh      # print the runs, submit nothing
#   CONDA_ENV=laplace_kernel_in_cl bash scripts/submit_cov_ablation.sh   # extra args go to sbatch
#
# Each run gets the smallest A100 (MIG slice or whole card) with at least GPU_GB GB (default 24);
# GRES=gpu:<type>:1 overrides. Finished runs (accs.pkl present) are skipped, so the same command can be
# resubmitted after failures.

set -eo pipefail

export DATASETS="${DATASETS:-cifar100}"
export VARIANTS="${VARIANTS:-pmi diag lcm lcm_fixed lowrank blockdiag sparse shrinkage}"
export SEEDS="${SEEDS:-1 2 3}"
export LOWRANK_GRID="${LOWRANK_GRID:-2 4 8}"
export BLOCK_GRID="${BLOCK_GRID:-4 8 16}"
export SPARSE_GRID="${SPARSE_GRID:-1 2 4}"
export SHRINK_GRID="${SHRINK_GRID:-oas 0.1 0.3}"
export BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation}"
export CODE_DIR="${CODE_DIR:-$(pwd)}"
MAX_PARALLEL="${MAX_PARALLEL:-14}"
GPU_GB="${GPU_GB:-24}"

[[ -f "$CODE_DIR/main_cov_ablation.py" ]] || { echo "Run this from the resnet32 directory (or set CODE_DIR)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/cov_ablation_10_task.sh"
source "$SCRIPT_DIR/gpu.sh"

N_RUNS=$(STAGE=tune COUNT_ONLY=1 bash "$JOB_SCRIPT")
(( N_RUNS > 0 )) || { echo "Nothing to submit"; exit 1; }
echo "Stage 1 (tuning): $N_RUNS runs (max $MAX_PARALLEL at once)"
echo "  datasets: $DATASETS   seeds: $SEEDS (tuning on seed ${SEEDS%% *})   alpha_frob: ${ALPHA:-Table 3}"
echo "  grids:    lowrank {$LOWRANK_GRID}  blockdiag {$BLOCK_GRID}  sparse {$SPARSE_GRID}  shrinkage {$SHRINK_GRID}"
echo "  results:  $BASE_PATH/<dataset>_10task/<variant>/seed_<seed>/accs.pkl"
STAGE=tune LIST_ONLY=1 bash "$JOB_SCRIPT"
if [[ "${LIST_ONLY:-0}" == "1" ]]; then
    exit 0
fi
GPU=$(gpu_options "$GPU_GB") || exit 1
read -r -a GPU_OPTS <<< "$GPU"
echo "GPU per run: $GPU (at least $GPU_GB GB)"

mkdir -p "$BASE_PATH/logs"
sbatch --export=ALL,STAGE=tune \
       --job-name=cov_ablation_tune \
       --array=0-$(( N_RUNS - 1 ))%"$MAX_PARALLEL" \
       --output="$BASE_PATH/logs/%x_%A_%a.out" \
       "${GPU_OPTS[@]}" \
       "$@" \
       "$JOB_SCRIPT"
echo "When these runs have finished: bash scripts/submit_cov_ablation_final.sh"
