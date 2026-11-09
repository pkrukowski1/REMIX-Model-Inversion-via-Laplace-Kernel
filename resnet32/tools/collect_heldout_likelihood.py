import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

FAMILIES = {
    'channel': ['diag', 'lcm', 'lcm_fixed', 'lcm_perm', 'lowrank', 'blockdiag', 'sparse', 'shrinkage', 'full'],
    'flat': ['diag', 'lcm', 'lcm_fixed', 'lcm_perm', 'lowrank', 'blockdiag', 'shrinkage'],
}
REFERENCE = {'channel': 'full', 'flat': 'shrinkage'}   # dense model: "% of dense gain"
NAMES = {'diag': 'diagonal', 'lcm': 'LCM (REMIX)', 'lcm_fixed': 'LCM, coordinates fixed',
         'lcm_perm': 'LCM, learned coordinates permuted', 'lowrank': 'low-rank + diag', 'blockdiag': 'block-diagonal',
         'sparse': 'sparse precision', 'shrinkage': 'dense + shrinkage', 'full': 'dense'}
HPARAM_NAME = {'lowrank': 'rank', 'blockdiag': 'block', 'sparse': 'edges/C', 'shrinkage': 'shrinkage'}


def load(root):
    parts = defaultdict(dict)   # mode -> seed -> part -> results
    for path in sorted(glob.glob(os.path.join(root, 'parts', 'seed_*.json'))):
        with open(path) as fr:
            d = json.load(fr)
        parts[d['mode']].setdefault(int(d['seed']), {})[d['part']] = d['results']
    return parts


def scores(by_part, mode, split):
    """{config: {group: LL/dim}} of one seed; flat groups are the stages plus their mean ('all')."""
    if mode == 'channel':
        return {c: dict(r[split]) for c, r in by_part['channel'].items()}
    stages = sorted(by_part)
    out = {}
    for c in by_part[stages[0]]:
        out[c] = {st: by_part[st][c][split] for st in stages}
        out[c]['all'] = float(np.mean([out[c][st] for st in stages]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/raid/NFS_SHARE/results/patryk.krukowski/LaplaceKernelInContinualLearning/cov_ablation/heldout_likelihood')
    parser.add_argument('--tune_seed', type=int, default=None, help='seed whose validation split selects the values (default: smallest)')
    parser.add_argument('--out', default=None, help='output prefix (default: <root>/heldout_likelihood)')
    args = parser.parse_args()
    out = args.out or os.path.join(args.root, 'heldout_likelihood')

    parts = load(args.root)
    if not parts:
        raise SystemExit('No results under {}/parts'.format(args.root))
    lines, rows, warnings = [], [], []
    for mode in ('channel', 'flat'):
        if mode not in parts:
            continue
        by_seed = parts[mode]
        if mode == 'flat':   # use the stages every seed has
            common = set.intersection(*[set(p) for p in by_seed.values()])
            for seed, p in by_seed.items():
                if set(p) != common:
                    warnings.append('{} seed {}: stages {} missing for other seeds; using {}'.format(
                        mode, seed, sorted(set(p) - common), sorted(common)))
            by_seed = {s: {st: p[st] for st in common} for s, p in by_seed.items()}
            if not common:
                continue
        seeds = sorted(by_seed)
        tune_seed = args.tune_seed if args.tune_seed is not None else seeds[0]
        if tune_seed not in by_seed:
            raise SystemExit('{}: no results for the tuning seed {}'.format(mode, tune_seed))
        val = scores(by_seed[tune_seed], mode, 'val')
        test = {s: scores(by_seed[s], mode, 'test') for s in seeds}
        meta = by_seed[tune_seed]['channel'] if mode == 'channel' else by_seed[tune_seed][sorted(by_seed[tune_seed])[0]]

        chosen = {}
        for family in FAMILIES[mode]:
            cands = [c for c in meta if meta[c]['family'] == family]   # grid order
            if cands:
                chosen[family] = max(cands, key=lambda c: (val[c]['all'], -cands.index(c)))
        diag, ref = chosen['diag'], chosen.get(REFERENCE[mode])
        groups = [g for g in val[diag] if g != 'all']

        def label(g):
            if mode == 'channel':
                return g
            return '{} (D={})'.format(g, by_seed[tune_seed][g][diag]['info']['dim'])

        lines += ['## {}: test log-likelihood per dimension minus the diagonal model ({} seed{}: {})'.format(
            'Per-channel features (BatchNorm inputs, as in REMIX)' if mode == 'channel' else 'Flattened stage outputs',
            len(seeds), '' if len(seeds) == 1 else 's', ' '.join(map(str, seeds))), '',
            '| Model | Config | {} | {} | mean | % of dense gain |'.format(
                'Params/layer (C=64)' if mode == 'channel' else 'Params/dim', ' | '.join(label(g) for g in groups)),
            '|---|---|---|' + '---|' * len(groups) + '---|---|']
        for family in FAMILIES[mode]:
            if family not in chosen:
                continue
            c = chosen[family]
            deltas = {g: [test[s][c][g] - test[s][diag][g] for s in seeds] for g in groups + ['all']}
            gain = []
            for s in seeds:
                ref_delta = test[s][ref]['all'] - test[s][diag]['all'] if ref else float('nan')
                gain.append(100.0 * (test[s][c]['all'] - test[s][diag]['all']) / ref_delta if ref_delta > 0 else float('nan'))
            params = meta[c]['params']
            if mode == 'flat' and family == 'shrinkage':
                params_text = '(D−1)/2'
            else:
                params_text = '–' if params is None or (isinstance(params, float) and np.isnan(params)) else '{:g}'.format(params)
            cells = ['{:+.4f} ± {:.4f}'.format(np.mean(deltas[g]), np.std(deltas[g], ddof=1) if len(seeds) > 1 else 0.0)
                     for g in groups + ['all']]
            value = meta[c]['value']
            lines.append('| {} | {} | {} | {} | {} |'.format(
                NAMES[family], '–' if value is None else '{} = {}'.format(HPARAM_NAME[family], value), params_text,
                ' | '.join(cells), '–' if np.all(np.isnan(gain)) else '{:.0f}%'.format(np.nanmean(gain))))
            rows.append(dict({'mode': mode, 'family': family, 'config': c, 'n_seeds': len(seeds)},
                             **{'delta_' + g: float(np.mean(deltas[g])) for g in groups + ['all']},
                             delta_all_std=float(np.std(deltas['all'], ddof=1)) if len(seeds) > 1 else 0.0,
                             dense_gain_pct=float(np.nanmean(gain)) if not np.all(np.isnan(gain)) else float('nan')))
        lines += ['', '**Tuning ({}; validation split, seed {})**'.format(mode, tune_seed), '',
                  '| Config | ' + ' | '.join(groups) + ' | mean | selected |', '|---|' + '---|' * len(groups) + '---|---|']
        for c in meta:
            if meta[c]['value'] is None:
                continue
            lines.append('| {} | {} | {:+.4f} | {} |'.format(
                c, ' | '.join('{:+.4f}'.format(val[c][g] - val[diag][g]) for g in groups),
                val[c]['all'] - val[diag]['all'], '✓' if chosen[meta[c]['family']] == c else ''))
        lines.append('')
    lines.append('Values are differences in held-out log-likelihood per dimension (nats) to the diagonal Gaussian '
                 '(mean ± std over backbone seeds), on train-standardised features. Per-channel models are fitted '
                 'exactly as in the continual-learning runs (to the training correlation); flattened-feature models by '
                 'maximum likelihood (LCM: exact scan likelihood, Adam; low-rank: factor analysis, EM; block-diagonal '
                 'and dense: closed form, contiguous blocks). "LCM, coordinates fixed" freezes equally spaced '
                 'coordinates in channel order (pixel order for flattened maps); "learned coordinates permuted" '
                 'shuffles the fitted coordinates without refitting. % of dense gain = gain over the diagonal model '
                 'relative to the dense model.')
    lines += ['Warning: ' + w for w in warnings]
    text = '\n'.join(lines)
    with open(out + '.md', 'w') as fw:
        fw.write(text + '\n')
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out + '.csv', 'w', newline='') as fw:
            writer = csv.DictWriter(fw, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(text)
    print('\nWrote {0}.md and {0}.csv'.format(out))


if __name__ == '__main__':
    main()
