#!/bin/bash
#SBATCH --job-name=cov_ablation_10task
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation/logs/%x_%A_%a.out
#
# Covariance-model ablation, ResNet-32, CIFAR-100, 10 tasks. Same hyperparameters as
# scripts/task_cont_cifar100_10_task.sh; the only difference between variants is the correlation model
# inside the REMIX NLL term. Two stages, each with its own submit script (run from the resnet32 directory):
#
# STAGE=tune   scripts/submit_cov_ablation.sh: variants with a hyperparameter run each value of their grid
#              on the first seed; pmi, diag, lcm and lcm_fixed run all seeds.
# STAGE=final  scripts/submit_cov_ablation_final.sh, after stage 1: each tuned variant runs its best value
#              (final accuracy on the first seed, tools/select_cov_config.py) on the remaining seeds.
#
# Variants (parameters per layer for C = 64):
#   pmi        alpha_frob = 0, no correlation term                           0
#   diag       diagonal Gaussian NLL, same weight as REMIX                   0
#   lcm        REMIX (Laplace covariance model)                              192
#   lcm_fixed  REMIX with coordinates frozen (no learned 1D order)           128
#   lowrank    U U^T + diag, rank in LOWRANK_GRID                            192 / 320 / 576 (rank 2 / 4 / 8)
#   blockdiag  block-diagonal, block size in BLOCK_GRID                      96 / 224 / 480  (4 / 8 / 16)
#   sparse     graphical lasso, precision edges per channel in SPARSE_GRID   ~128 / ~192 / ~320 (1 / 2 / 4)
#   shrinkage  dense correlation, shrinkage in SHRINK_GRID                   2016 (oas / 0.1 / 0.3)
#   full       dense merged correlation, no shrinkage (not run by default)   2016
# Every non-PMI variant uses alpha_frob from Table 3 (0.5 for CIFAR-100, 0.25 for Tiny) unless ALPHA is set.

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

STAGE="${STAGE:-tune}"
DATASETS="${DATASETS:-cifar100}"
VARIANTS="${VARIANTS:-pmi diag lcm lcm_fixed lowrank blockdiag sparse shrinkage}"
SEEDS="${SEEDS:-1 2 3}"
LOWRANK_GRID="${LOWRANK_GRID:-2 4 8}"
BLOCK_GRID="${BLOCK_GRID:-4 8 16}"
SPARSE_GRID="${SPARSE_GRID:-1 2 4}"
SHRINK_GRID="${SHRINK_GRID:-oas 0.1 0.3}"
SELECT_METRIC="${SELECT_METRIC:-final}"
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

grid_of() {
    case "$1" in
        lowrank)   echo "$LOWRANK_GRID" ;;
        blockdiag) echo "$BLOCK_GRID" ;;
        sparse)    echo "$SPARSE_GRID" ;;
        shrinkage) echo "$SHRINK_GRID" ;;
    esac
}

read -r -a DS <<< "$DATASETS"
read -r -a VS <<< "$VARIANTS"
read -r -a SS <<< "$SEEDS"
TUNE_SEED="${SS[0]}"

# One entry per run: "dataset variant hyperparameter seed" ('-': none; 'best': selected when the job starts).
PLAN=()
for ds in "${DS[@]}"; do
    for v in "${VS[@]}"; do
        read -r -a GRID <<< "$(grid_of "$v")"
        case "$STAGE" in
            tune)
                if (( ${#GRID[@]} )); then
                    for h in "${GRID[@]}"; do PLAN+=("$ds $v $h $TUNE_SEED"); done
                else
                    for s in "${SS[@]}"; do PLAN+=("$ds $v - $s"); done
                fi ;;
            final)
                if (( ${#GRID[@]} )); then
                    for s in "${SS[@]:1}"; do PLAN+=("$ds $v best $s"); done
                fi ;;
            *) echo "STAGE must be 'tune' or 'final'"; exit 1 ;;
        esac
    done
done
N_RUNS=${#PLAN[@]}
if [[ "${COUNT_ONLY:-0}" == "1" ]]; then
    echo "$N_RUNS"
    exit 0
fi
if [[ "${LIST_ONLY:-0}" == "1" ]]; then
    for i in "${!PLAN[@]}"; do echo "    $i: ${PLAN[$i]}"; done
    exit 0
fi

dataset_settings() {   # settings of the 10-task scripts, and where the runs of this dataset go
    case "$1" in
        cifar100)
            dataset='seq_cifar100'
            data_path='/raid/NFS_SHARE/datasets'
            class_order_file='./class_order_cifar100.txt'
            lrs='1e-1,5e-2,0,0,1e-2'
            gen_batch_size=4000
            default_alpha=0.5 ;;
        tiny)
            dataset='seq_tinyimagenet'
            data_path='/raid/NFS_SHARE/datasets/tiny-imagenet-200'
            class_order_file='./class_order_tiny_imgenet.txt'
            lrs='1e-1,5e-2,0,0,5e-3'
            gen_batch_size=1000
            default_alpha=0.25 ;;
        *) echo "Unknown dataset '$1' (use cifar100 or tiny)"; exit 1 ;;
    esac
    ALPHA_RUN="${ALPHA:-$default_alpha}"
    SETTING_PATH="$BASE_PATH/${1}_10task"
}

tag_of() {   # run directory name of (variant, hyperparameter)
    case "$1" in
        pmi)                      echo "pmi" ;;
        lcm_fixed)                echo "lcm-fixed_a${ALPHA_RUN}" ;;
        shrinkage)                echo "shrinkage-${2}_a${ALPHA_RUN}" ;;
        lowrank|blockdiag|sparse) echo "${1}${2}_a${ALPHA_RUN}" ;;
        *)                        echo "${1}_a${ALPHA_RUN}" ;;
    esac
}

select_hparam() {   # best grid value of tuned variant $1 for the current dataset; fails while tuning runs are missing
    local grid cands=() h best
    read -r -a grid <<< "$(grid_of "$1")"
    for h in "${grid[@]}"; do cands+=("$(tag_of "$1" "$h")"); done
    best=$(python3 "$CODE_DIR/tools/select_cov_config.py" --root "$SETTING_PATH" --seed "$TUNE_SEED" \
               --metric "$SELECT_METRIC" --tags "${cands[@]}") || return 1
    echo "${grid[$best]}"
}

cd "$CODE_DIR"
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# SELECT_ONLY=1: print the value each tuned variant will use in the final stage; exit 1 if tuning runs are
# missing (scripts/submit_cov_ablation_final.sh calls this before submitting).
if [[ "${SELECT_ONLY:-0}" == "1" ]]; then
    status=0
    for ds in "${DS[@]}"; do
        dataset_settings "$ds"
        for v in "${VS[@]}"; do
            [[ -n "$(grid_of "$v")" ]] || continue
            if h=$(select_hparam "$v"); then
                echo "  $ds $v: $h"
            else
                status=1
            fi
        done
    done
    exit "$status"
fi

IDX="${SLURM_ARRAY_TASK_ID:?Run as a SLURM array job (see scripts/submit_cov_ablation.sh)}"
if (( IDX >= N_RUNS )); then
    echo "Array index $IDX >= number of runs $N_RUNS; nothing to do."
    exit 0
fi
read -r DATASET VARIANT HPARAM SEED <<< "${PLAN[$IDX]}"
dataset_settings "$DATASET"

if [[ "$HPARAM" == best ]]; then
    if ! HPARAM=$(select_hparam "$VARIANT"); then
        echo "Cannot select the $VARIANT configuration (tuning runs above are missing)."
        [[ "${DRY_RUN:-0}" == "1" ]] && exit 0
        exit 1
    fi
    echo "Selected $VARIANT $HPARAM (best $SELECT_METRIC accuracy on seed $TUNE_SEED among: $(grid_of "$VARIANT"))"
fi

# ---- variant -----------------------------------------------------------------------------------
case "$VARIANT" in
    pmi)                     COV=(--alpha_frob 0) ;;
    diag|lcm|lcm_fixed|full) COV=(--alpha_frob "$ALPHA_RUN" --cov_type "$VARIANT") ;;
    lowrank)                 COV=(--alpha_frob "$ALPHA_RUN" --cov_type lowrank --cov_rank "$HPARAM") ;;
    blockdiag)               COV=(--alpha_frob "$ALPHA_RUN" --cov_type blockdiag --cov_block "$HPARAM") ;;
    sparse)                  COV=(--alpha_frob "$ALPHA_RUN" --cov_type sparse --cov_edges_per_dim "$HPARAM") ;;
    shrinkage)               COV=(--alpha_frob "$ALPHA_RUN" --cov_type shrinkage --cov_shrinkage "$HPARAM") ;;
    *) echo "Unknown variant '$VARIANT'"; exit 1 ;;
esac
TAG="$(tag_of "$VARIANT" "$HPARAM")"
RUN_PATH="$SETTING_PATH/$TAG/seed_$SEED"
echo "[$STAGE $IDX/$N_RUNS] dataset=$DATASET variant=$VARIANT hparam=$HPARAM seed=$SEED -> $RUN_PATH"

ARGS=(
    --local_path="$RUN_PATH"
    --dataset="$dataset"
    --data_path="$data_path"
    --use_cuda=1
    --max_task=-1
    --save_ckpt=1
    --seed="$SEED"
    --init_split=0
    --tasks=10
    --class_order_file="$class_order_file"
    --batch_size=128
    --base_lr=0.1
    --lr_factor=0.1
    --lrs="$lrs"
    --momentum=0.9
    --weight_decay=0.0005
    --milestones='80,120'
    --finetuning_epochs=40
    --finetuning_lr=0.005
    --max_epochs=200
    --epochs='20,20,0,0,120'
    --lambda_ce=0.75
    --lambda_hkd=0.15
    --lambda_rkd=0.5
    --lambda_ft=1.5
    --lambda_fkd=0
    --ce_mode='local'
    --init_samples=3000
    --per_task_samples=2000
    --gen_batch_size="$gen_batch_size"
    --use_unslt=0
    --feat_type='cont'
    --act='leaky'
    --save_data=0
    --cont_lr=1e-2
    --cont_epoch=200
    --cont_blocks=2
    --cont_step=40
    --cont_rslt=0.5
    --cont_temp=1.0
    --inversion_lr=0.8
    --train_steps=50
    --tune_steps=160
    --tune_lr=0.4
    --alpha_pr=0.0
    --alpha_rf=0.25
    --rf_factor=20.0
    --layer_wise=1
    --layer_batch=4000
    --boost_factor=1
    --search_param=0
    --inv_milestones='80,120'
    --inv_lr_rate=0.5
    --inv_warmup=5
)
[[ "${SAVE_PSEUDO:-0}" == "1" ]] && ARGS+=(--save_pseudo_samples)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u main_cov_ablation.py ${COV[*]} --local_path=$RUN_PATH --dataset=$dataset --seed=$SEED --lrs=$lrs --gen_batch_size=$gen_batch_size ..."
    exit 0
fi

if [[ -f "$RUN_PATH/accs.pkl" ]]; then
    echo "Already finished ($RUN_PATH/accs.pkl); skipping."
    exit 0
fi
# Never reuse a half-finished run directory (stale buffers / stats would leak into the new run).
if [[ -d "$RUN_PATH" ]]; then
    case "$RUN_PATH" in
        "$BASE_PATH"/*/seed_*) echo "Removing incomplete run directory $RUN_PATH"; rm -rf -- "$RUN_PATH" ;;
        *) echo "Refusing to remove unexpected path $RUN_PATH"; exit 1 ;;
    esac
fi
mkdir -p "$RUN_PATH"

{
    echo "date: $(date -Is)"
    echo "host: $(hostname)  job: ${SLURM_ARRAY_JOB_ID:-?}_${IDX}  stage: $STAGE"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "cmd: python3 -u main_cov_ablation.py ${COV[*]} ${ARGS[*]}"
} > "$RUN_PATH/run_info.txt"

python3 -u main_cov_ablation.py "${COV[@]}" "${ARGS[@]}"

[[ -f "$RUN_PATH/accs.pkl" ]] || { echo "Run ended without accs.pkl"; exit 1; }
