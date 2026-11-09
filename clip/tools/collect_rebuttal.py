# -*-coding:utf8-*-
# Summary of the MoE-Adapter rebuttal runs (scripts/moe_adapter/rebuttal_array.sh): mean ± std over seeds and
# paired REMIX - PMI differences (same seed = same class order) with t-test, 95% CI and wins.
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:  # p-values and confidence intervals are skipped without scipy
    stats = None

ORDER = ['pmi', 'remix', 'real_replay', 'joint']
NAMES = {'pmi': 'PMI + MoE-Adapter', 'remix': 'REMIX + MoE-Adapter', 'real_replay': 'upper bound: real replay',
         'joint': 'upper bound: joint training'}
METRICS = (('avg_acc', 'Avg'), ('last_acc', 'Last'))


def method_of(tag):
    return 'remix' if tag.startswith('remix') else tag


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired(a, b, key):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    d = np.array([a[s][key] - b[s][key] for s in seeds])
    out = {'n': len(seeds), 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           'wins': int((d > 0).sum()), 'p': float('nan'), 'ci_lo': float('nan'), 'ci_hi': float('nan'),
           'order_mismatch': sum(a[s].get('class_order') != b[s].get('class_order') for s in seeds)}
    if stats is not None and len(d) > 1 and d.std() > 0:
        out['p'] = float(stats.ttest_rel([a[s][key] for s in seeds], [b[s][key] for s in seeds]).pvalue)
        half = float(stats.t.ppf(0.975, len(d) - 1)) * out['std'] / np.sqrt(len(d))
        out['ci_lo'], out['ci_hi'] = out['mean'] - half, out['mean'] + half
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/rebuttal')
    parser.add_argument('--out', default='rebuttal_summary')
    args = parser.parse_args()

    runs = defaultdict(dict)          # (setting, method) -> seed -> results.json
    tags = {}
    unfinished = defaultdict(int)
    for seed_dir in sorted(glob.glob(os.path.join(args.root, '*', '*', 'seed_*'))):
        setting = os.path.basename(os.path.dirname(os.path.dirname(seed_dir)))
        tag = os.path.basename(os.path.dirname(seed_dir))
        method = method_of(tag)
        result_file = os.path.join(seed_dir, 'results.json')
        if not os.path.exists(result_file):
            unfinished[(setting, method)] += 1
            continue
        with open(result_file) as fr:
            res = json.load(fr)
        if (setting, method) in tags and tags[(setting, method)] != tag:
            raise SystemExit('Two configurations of {} in {}: {} and {}'.format(method, setting, tags[(setting, method)], tag))
        tags[(setting, method)] = tag
        runs[(setting, method)][int(res['seed'])] = res
    if not runs:
        raise SystemExit('No finished runs under {}'.format(args.root))

    rows, lines = [], []
    lines += ['| Setting | Tasks | Method | Seeds | Avg acc. | Last acc. | Δ Avg vs PMI | Δ Last vs PMI |',
              '|---|---|---|---|---|---|---|---|']
    all_wins = {key: [0, 0] for key, _ in METRICS}
    warnings = []
    for setting in sorted({s for s, _ in runs}):
        methods = sorted({m for s, m in runs if s == setting}, key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))
        pmi = runs.get((setting, 'pmi'))
        for method in methods:
            res = runs[(setting, method)]
            seeds = sorted(res)
            row = {'setting': setting, 'method': method, 'tag': tags[(setting, method)], 'n_seeds': len(seeds),
                   'seeds': ' '.join(map(str, seeds)), 'unfinished': unfinished[(setting, method)],
                   'nb_tasks': res[seeds[0]].get('nb_tasks'),
                   'alpha_frob': res[seeds[0]].get('alpha_frob')}
            for key, _ in METRICS:
                row[key + '_mean'], row[key + '_std'] = mean_std([res[s][key] for s in seeds])
                cmp_ = paired(res, pmi, key) if pmi is not None and method != 'pmi' else None
                for k in ('mean', 'std', 'p', 'ci_lo', 'ci_hi', 'wins', 'n'):
                    row['d_{}_{}'.format(key, k)] = cmp_[k] if cmp_ else float('nan')
                if cmp_ and cmp_['order_mismatch'] and key == METRICS[0][0]:
                    warnings.append('{} {}: {} paired seed(s) with a different class order than PMI'.format(
                        setting, method, cmp_['order_mismatch']))
                if cmp_ and method == 'remix':
                    all_wins[key][0] += cmp_['wins']
                    all_wins[key][1] += cmp_['n']
            rows.append(row)

            def delta(key):
                if np.isnan(row['d_{}_mean'.format(key)]):
                    return '–'
                p, lo, hi = (row['d_{}_{}'.format(key, k)] for k in ('p', 'ci_lo', 'ci_hi'))
                text = '{:+.2f} ± {:.2f}'.format(row['d_{}_mean'.format(key)], row['d_{}_std'.format(key)])
                if not np.isnan(p):
                    text += ' [{:+.2f}, {:+.2f}], p{}'.format(lo, hi, '<0.001' if p < 0.001 else '={:.3f}'.format(p))
                return text + ', {}/{}'.format(row['d_{}_wins'.format(key)], row['d_{}_n'.format(key)])

            lines.append('| {} | {} | {} | {}{} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {} | {} |'.format(
                setting, row['nb_tasks'], NAMES.get(method, method), row['n_seeds'],
                ' (+{} unfinished)'.format(row['unfinished']) if row['unfinished'] else '',
                row['avg_acc_mean'], row['avg_acc_std'], row['last_acc_mean'], row['last_acc_std'],
                delta('avg_acc'), delta('last_acc')))

    lines += ['', 'Mean ± std over seeds. Δ = paired REMIX − PMI (or bound − PMI) difference on the same seeds, '
                  'which share the class order: mean ± std [95% CI], two-sided paired t-test p, wins/n.']
    if all_wins['avg_acc'][1]:
        lines.append('REMIX above PMI: Avg in {}/{} paired runs, Last in {}/{}.'.format(
            *all_wins['avg_acc'], *all_wins['last_acc']))
    lines += ['Warning: ' + w for w in warnings]
    text = '\n'.join(lines)
    with open(args.out + '.md', 'w') as fw:
        fw.write(text + '\n')
    with open(args.out + '.csv', 'w', newline='') as fw:
        writer = csv.DictWriter(fw, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(text)
    print('\nWrote {0}.md and {0}.csv'.format(args.out))


if __name__ == '__main__':
    main()
