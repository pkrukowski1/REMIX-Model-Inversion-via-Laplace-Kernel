#!/bin/bash
#SBATCH --job-name=codaprompt_remix_final
#SBATCH --qos=normal
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt/logs/%x_%A_%a.out

# REMIX + CODA-Prompt (ViT-B/16) on CIFAR-100, 10 tasks: the alpha_stat with the highest average incremental
# accuracy on seed 1993 in scripts/codaprompt/remix_tune.sh, run on seeds 1994 and 1995 (one array task per
# seed); the tuning run counts as seed 1993. Nothing runs while a tuning run is missing.
# From the CLIP code directory:
#   sbatch scripts/codaprompt/remix_final.sh
#   SUMMARY=1 bash scripts/codaprompt/remix_final.sh   # grid, selected value, CODA-Prompt vs REMIX + CODA-Prompt

set -eo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

read -r -a ALPHAS <<< "${ALPHAS:-0.0001 0.001 0.01 0.1}"   # keep in sync with remix_tune.sh
TUNE_SEED="${TUNE_SEED:-1993}"
read -r -a SEEDS <<< "${SEEDS:-1994 1995}"
METRIC="${SELECT_METRIC:-avg_acc}"
BASE_PATH="${BASE_PATH:-/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/codaprompt}"
[[ "$METRIC" == avg_acc || "$METRIC" == last_acc ]] || { echo "SELECT_METRIC must be avg_acc or last_acc"; exit 1; }

if ! BEST=$(python3 - "$BASE_PATH/cifar" "$TUNE_SEED" "$METRIC" "${ALPHAS[@]}" <<'EOF'
import json
import os
import sys

root, seed, metric, alphas = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
scores = {}
for a in alphas:
    path = os.path.join(root, 'tune_remix_a' + a, 'seed_' + seed, 'results.json')
    if os.path.exists(path):
        with open(path) as fr:
            res = json.load(fr)
        scores[a] = res[metric]
        sys.stderr.write('  alpha_stat={:<8} avg_acc={:.2f}  last_acc={:.2f}\n'.format(a, res['avg_acc'], res['last_acc']))
    else:
        sys.stderr.write('  alpha_stat={:<8} not finished\n'.format(a))
if len(scores) < len(alphas):
    sys.exit(1)
print(max(alphas, key=lambda a: scores[a]))
EOF
); then
    echo "Some tuning runs have not finished (listed above); submit again when scripts/codaprompt/remix_tune.sh is done."
    exit 1
fi
echo "Selected alpha_stat=$BEST (highest $METRIC on seed $TUNE_SEED)"
TAG="remix_a$BEST"

if [[ "${SUMMARY:-0}" == "1" ]]; then
    PYTHONIOENCODING=utf-8 python3 - "$BASE_PATH/cifar" "$TAG" "$TUNE_SEED" "${SEEDS[@]}" <<'EOF'
import json
import os
import statistics
import sys

root, tag, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]


def load(run, seed):
    path = os.path.join(root, run, 'seed_' + seed, 'results.json')
    if os.path.exists(path):
        with open(path) as fr:
            return json.load(fr)


def fmt(values):
    if not values:
        return '-'
    return '{:.2f} ± {:.2f}'.format(statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)


runs = {'CODA-Prompt': {s: load('coda', s) for s in seeds},
        'REMIX + CODA-Prompt': {s: load(('tune_' if i == 0 else '') + tag, s) for i, s in enumerate(seeds)}}
print('\nMean ± std over seeds (seed {} also selected alpha_stat):'.format(seeds[0]))
for name, res in runs.items():
    done = [s for s in seeds if res[s] is not None]
    print('  {:<20} seeds {:<15} avg_acc {:<14} last_acc {}'.format(
        name, ','.join(done) or '-', fmt([res[s]['avg_acc'] for s in done]), fmt([res[s]['last_acc'] for s in done])))
paired = [s for s in seeds if all(res[s] is not None for res in runs.values())]
for metric in ('avg_acc', 'last_acc') if paired else ():
    diff = [runs['REMIX + CODA-Prompt'][s][metric] - runs['CODA-Prompt'][s][metric] for s in paired]
    print('  REMIX - CODA-Prompt, {:<8} seeds {:<15} {} (higher on {}/{} seeds)'.format(
        metric, ','.join(paired), fmt(diff), sum(d > 0 for d in diff), len(diff)))
EOF
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
if (( IDX >= ${#SEEDS[@]} )); then
    echo "Array index $IDX >= number of seeds ${#SEEDS[@]}; nothing to do."
    exit 0
fi
SEED="${SEEDS[$IDX]}"
RUN_PATH="$BASE_PATH/cifar/$TAG/seed_$SEED"
echo "[$IDX/${#SEEDS[@]}] REMIX + CODA-Prompt, alpha_stat=$BEST, seed $SEED -> $RUN_PATH"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "  python3 -u main.py --config=./configs/codaprompt/cf100_inv.yaml --seed $SEED --prefix $TAG --local_path=$RUN_PATH --alpha_frob $BEST"
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
    --alpha_frob "$BEST"

[[ -f "$RUN_PATH/results.json" ]] || { echo "Run ended without results.json"; exit 1; }
