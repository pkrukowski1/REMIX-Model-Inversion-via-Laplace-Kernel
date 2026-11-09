# -*-coding:utf8-*-
import argparse
import csv
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:  # p-values are skipped without scipy
    stats = None

ORDER = ['pmi', 'diag', 'lowrank', 'blockdiag', 'sparse', 'shrinkage', 'full', 'lcm-fixed', 'lcm']
HPARAM_NAME = {'lowrank': 'rank', 'blockdiag': 'block', 'sparse': 'edges/C', 'shrinkage': 'shrinkage'}


def family(tag):
    base = tag.split('_a')[0]
    for name in ORDER:
        if base == name or (base.startswith(name) and name not in ('lcm',)):
            return name
    return base


def hparam(tag):
    base, fam = tag.split('_a')[0], family(tag)
    return base[len(fam):].lstrip('-') if base.startswith(fam) else ''


def grid_key(tag):
    """Order of the grid in scripts/cov_ablation_10_task.sh ('oas' first, then increasing values)."""
    h = hparam(tag)
    try:
        return 1, float(h), h
    except ValueError:
        return 0, 0.0, h


def select_best(values, tags):
    """Tag with the highest value; ties go to the earlier tag. Also used by tools/select_cov_config.py."""
    return max(tags, key=lambda t: (values[t], -tags.index(t)))


def load_run(run_dir):
    with open(os.path.join(run_dir, 'accs.pkl'), 'rb') as fr:
        accs = pickle.load(fr)
    per_step = [float(np.mean(a)) for a in accs]
    out = {'final': per_step[-1], 'avg_inc': float(np.mean(per_step)), 'n_tasks': len(accs)}
    log = os.path.join(run_dir, 'cov_fit_log.jsonl')
    if os.path.exists(log):
        with open(log) as fr:
            records = [json.loads(line) for line in fr if line.strip()]
        if records:
            out['fit_err'] = float(np.mean([r['rel_offdiag_err'] for r in records]))
            last_task = max(r['task'] for r in records)
            out['fit_err_last_task'] = float(np.mean([r['rel_offdiag_err'] for r in records if r['task'] == last_task]))
            c64 = [r['n_params'] for r in records if r['dim'] == 64]
            out['params_c64'] = float(np.mean(c64)) if c64 else float('nan')
            shr = [r['shrinkage'] for r in records if 'shrinkage' in r]
            if shr:
                out['shrinkage'] = float(np.mean(shr))
    return out


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def paired(a_by_seed, b_by_seed, key):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2:
        return None
    d = np.array([a_by_seed[s][key] - b_by_seed[s][key] for s in seeds])
    p = float(stats.ttest_rel([a_by_seed[s][key] for s in seeds], [b_by_seed[s][key] for s in seeds]).pvalue) \
        if stats is not None and np.std(d) > 0 else float('nan')
    return {'n': len(seeds), 'mean': float(d.mean()), 'std': float(d.std(ddof=1)), 'p': p, 'wins': int((d > 0).sum())}


def _num(value, fmt='{:.2f}'):
    return '–' if value is None or (isinstance(value, float) and np.isnan(value)) else fmt.format(value)


def _config(fam, tag):
    h = hparam(tag)
    return '{} = {}'.format(HPARAM_NAME.get(fam, 'value'), h) if h else '–'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', default='cov_ablation_summary')
    parser.add_argument('--metric', default='final', choices=['final', 'avg_inc'],
                        help='metric used for the paired comparisons')
    parser.add_argument('--select_metric', default='final', choices=['final', 'avg_inc'],
                        help='metric that picked each configuration on the tuning seed (SELECT_METRIC in the SLURM script)')
    parser.add_argument('--seeds', default='1 2 3',
                        help="seeds to report; the first one is the tuning seed ('all': every finished seed)")
    args = parser.parse_args()
    seeds_wanted = None if args.seeds == 'all' else [int(s) for s in args.seeds.split()]
    tune_seed = seeds_wanted[0] if seeds_wanted else 1

    runs = defaultdict(dict)        # (setting, tag) -> seed -> result
    missing = defaultdict(int)
    for seed_dir in sorted(glob.glob(os.path.join(args.root, '*', '*', 'seed_*'))):
        setting = os.path.basename(os.path.dirname(os.path.dirname(seed_dir)))
        tag = os.path.basename(os.path.dirname(seed_dir))
        seed = int(re.sub(r'\D', '', os.path.basename(seed_dir)))
        if seeds_wanted is not None and seed not in seeds_wanted:
            continue
        if os.path.exists(os.path.join(seed_dir, 'accs.pkl')):
            runs[(setting, tag)][seed] = load_run(seed_dir)
        else:
            missing[(setting, tag)] += 1
    if not runs:
        raise SystemExit('No finished runs under {}'.format(args.root))

    rows, tuning_rows = [], []
    for setting in sorted({s for s, _ in runs}):
        by_family = defaultdict(list)
        for s, tag in runs:
            if s == setting:
                by_family[family(tag)].append(tag)
        chosen = {}
        for fam, tags in by_family.items():
            tags.sort(key=grid_key)
            tuned = [t for t in tags if tune_seed in runs[(setting, t)]]
            if len(tags) > 1 and tuned:   # same rule as the SLURM final stage
                values = {t: runs[(setting, t)][tune_seed][args.select_metric] for t in tuned}
                chosen[fam] = select_best(values, tuned)
            else:
                chosen[fam] = max(tags, key=lambda t: len(runs[(setting, t)]))
            if len(tags) > 1:
                for tag in tags:
                    res = runs[(setting, tag)].get(tune_seed, {})
                    tuning_rows.append({
                        'setting': setting, 'variant': fam, 'config': _config(fam, tag), 'tag': tag,
                        'selected': tag == chosen[fam], 'final': res.get('final', float('nan')),
                        'avg_inc': res.get('avg_inc', float('nan')),
                        'params_c64': res.get('params_c64', float('nan')), 'fit_err': res.get('fit_err', float('nan')),
                    })
        order = sorted(chosen, key=lambda f: ORDER.index(f) if f in ORDER else len(ORDER))
        pmi, lcm = chosen.get('pmi'), chosen.get('lcm')
        for fam in order:
            tag = chosen[fam]
            res = runs[(setting, tag)]
            seeds = sorted(res)
            row = {'setting': setting, 'variant': fam, 'config': _config(fam, tag), 'tag': tag,
                   'n_seeds': len(seeds), 'seeds': ' '.join(map(str, seeds)), 'missing': missing[(setting, tag)],
                   'n_configs': len(by_family[fam])}
            for key in ('final', 'avg_inc'):
                row[key + '_mean'], row[key + '_std'] = mean_std([res[s][key] for s in seeds])
            for key in ('fit_err', 'fit_err_last_task', 'params_c64', 'shrinkage'):
                vals = [res[s][key] for s in seeds if key in res[s]]
                row[key] = float(np.mean(vals)) if vals else float('nan')
            for ref_name, ref in (('pmi', pmi), ('lcm', lcm)):
                cmp_ = paired(res, runs[(setting, ref)], args.metric) if ref and ref != tag else None
                for k in ('mean', 'std', 'p', 'wins', 'n'):
                    row['d_{}_{}'.format(ref_name, k)] = cmp_[k] if cmp_ else float('nan')
            rows.append(row)

    tuning_rows.sort(key=lambda r: (r['setting'], ORDER.index(r['variant']) if r['variant'] in ORDER else len(ORDER),
                                    grid_key(r['tag'])))
    with open(args.out + '.csv', 'w', newline='') as fw:
        writer = csv.DictWriter(fw, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if tuning_rows:
        with open(args.out + '_tuning.csv', 'w', newline='') as fw:
            writer = csv.DictWriter(fw, fieldnames=list(tuning_rows[0].keys()))
            writer.writeheader()
            writer.writerows(tuning_rows)

    def fmt_d(r, ref):
        if np.isnan(r['d_{}_mean'.format(ref)]):
            return '–'
        p = r['d_{}_p'.format(ref)]
        return '{:+.2f} ± {:.2f} (p={}, {}/{})'.format(
            r['d_{}_mean'.format(ref)], r['d_{}_std'.format(ref)],
            'n/a' if np.isnan(p) else '{:.3f}'.format(p), r['d_{}_wins'.format(ref)], r['d_{}_n'.format(ref)])

    m = '' if args.metric == 'final' else ' (avg)'
    lines = ['| Setting | Variant | Config | Seeds | Final acc. | Avg. inc. acc. | Δ{m} vs PMI | Δ{m} vs LCM | '
             'Params/layer (C=64) | Fit error |'.format(m=m),
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {} | {} | {}{} | {}{} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {} | {} | {} | {} |'.format(
            r['setting'], r['variant'], r['config'],
            ' (best of {})'.format(r['n_configs']) if r['n_configs'] > 1 else '',
            r['n_seeds'], ' (+{} unfinished)'.format(r['missing']) if r['missing'] else '',
            r['final_mean'], r['final_std'], r['avg_inc_mean'], r['avg_inc_std'],
            fmt_d(r, 'pmi'), fmt_d(r, 'lcm'), _num(r['params_c64'], '{:.0f}'), _num(r['fit_err'], '{:.3f}')))
    if tuning_rows:
        lines += ['', '**Tuning (seed {}; ✓ = configuration run on all seeds)**'.format(tune_seed), '',
                  '| Setting | Variant | Config | Params/layer (C=64) | Final acc. | Avg. inc. acc. | Fit error | Selected |',
                  '|---|---|---|---|---|---|---|---|']
        for r in tuning_rows:
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                r['setting'], r['variant'], r['config'], _num(r['params_c64'], '{:.0f}'), _num(r['final']),
                _num(r['avg_inc']), _num(r['fit_err'], '{:.3f}'), '✓' if r['selected'] else ''))
    table = '\n'.join(lines)
    note = ('\nFinal acc. = mean accuracy over all tasks after the last task; mean ± std over seeds. '
            'Variants with a hyperparameter were tuned on seed {s} ({k} accuracy, at most three values) and the '
            'selected value was run on the remaining seeds, so seed {s} is reused; this favours the tuned '
            'baselines. PMI, diag, LCM and LCM with fixed coordinates have no tuned hyperparameter (alpha_frob '
            'from Table 3 for all variants). Δ = paired difference on matching seeds (mean ± std, paired t-test p, '
            'wins/n). Fit error = ||offdiag(R_model − R_merged)||_F / ||offdiag(R_merged)||_F, averaged over '
            'layers and tasks.'.format(s=tune_seed, k=args.select_metric))
    with open(args.out + '.md', 'w') as fw:
        fw.write(table + '\n' + note + '\n')
    print(table)
    print(note)
    print('\nWrote {0}.md and {0}.csv{1}'.format(args.out, ' and {}_tuning.csv'.format(args.out) if tuning_rows else ''))


if __name__ == '__main__':
    main()
