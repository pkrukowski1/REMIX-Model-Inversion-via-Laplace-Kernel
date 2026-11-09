import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import utils  # noqa: E402
from inversion import cov_models  # noqa: E402
from inversion.lcm import LCM  # noqa: E402

CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
LOG2PI = math.log(2.0 * math.pi)
JITTER = 1e-5
GRIDS = {'lowrank': [2, 4, 8], 'blockdiag': [4, 8, 16], 'sparse': [1, 2, 4], 'shrinkage': ['oas', 0.1, 0.3]}
FAMILIES = {
    'channel': ['diag', 'lcm', 'lcm_fixed', 'lcm_perm', 'lowrank', 'blockdiag', 'sparse', 'shrinkage', 'full'],
    'flat': ['diag', 'lcm', 'lcm_fixed', 'lcm_perm', 'lowrank', 'blockdiag', 'shrinkage'],
}
NAMES = {'diag': 'diagonal', 'lcm': 'LCM (REMIX)', 'lcm_fixed': 'LCM, coordinates fixed',
         'lcm_perm': 'LCM, learned coordinates permuted', 'lowrank': 'low-rank + diag', 'blockdiag': 'block-diagonal',
         'sparse': 'sparse precision', 'shrinkage': 'dense + shrinkage', 'full': 'dense'}
HPARAM_NAME = {'lowrank': 'rank', 'blockdiag': 'block', 'sparse': 'edges/C', 'shrinkage': 'shrinkage'}


def configs_of(families):
    return [(f, v) for f in families for v in GRIDS.get(f, [None])]


def config_name(family, value):
    return family if value is None else '{}={}'.format(family, value)


# ------------------------------------------------------------------------------------------------
# data and features
# ------------------------------------------------------------------------------------------------
def load_cifar100(data_path, train):
    import torchvision
    return torchvision.datasets.CIFAR100(root=data_path, train=train, download=False).data  # uint8 NHWC


def normalise(images):
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)) / torch.tensor(CIFAR_STD).view(1, 3, 1, 1)


def load_backbone(path, device):
    try:
        model = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13
        model = torch.load(path, map_location='cpu')
    if not isinstance(model, torch.nn.Module):
        raise SystemExit('{} is not a pickled backbone (expected backbone_<task>.pkl written by dump_model)'.format(path))
    return model.to(device).eval()


class Moments:
    """Running first and second moments of the rows of (rows, C) blocks, in float64."""

    def __init__(self):
        self.n, self.s, self.ss = 0, None, None

    def add(self, x):
        x = x.double()
        if self.s is None:
            self.s, self.ss = x.sum(0), x.t() @ x
        else:
            self.s += x.sum(0)
            self.ss += x.t() @ x
        self.n += x.shape[0]


@torch.no_grad()
def extract(backbone, images, device, batch_size, stages, with_moments=True):
    """Moments of every BatchNorm input (pixels as rows) and flattened outputs of the given stages."""
    bns = [m for m in backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)] if with_moments else []
    moments = [Moments() for _ in bns]
    flat = {name: [] for name in stages}

    def bn_hook(i):
        def hook(module, inp, out):
            x = inp[0]
            moments[i].add(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))
        return hook

    def stage_hook(name):
        def hook(module, inp, out):
            x = out[-1] if isinstance(out, (tuple, list)) else out  # Stage returns (block features, output)
            flat[name].append(x.flatten(1).float().cpu())
        return hook

    hooks = [m.register_forward_hook(bn_hook(i)) for i, m in enumerate(bns)]
    hooks += [getattr(backbone, name).register_forward_hook(stage_hook(name)) for name in stages]
    try:
        for start in range(0, len(images), batch_size):
            backbone(normalise(images[start:start + batch_size]).to(device))
    finally:
        for h in hooks:
            h.remove()
    return moments, {name: torch.cat(v) for name, v in flat.items()}


# ------------------------------------------------------------------------------------------------
# per-channel features: everything follows from second moments
# ------------------------------------------------------------------------------------------------
def standardised(m_train, m_eval):
    """Train correlation R and the eval second moment about the train mean, both in train-standardised units."""
    mu = m_train.s / m_train.n
    cov = m_train.ss / m_train.n - torch.outer(mu, mu)
    sd = cov.diagonal().clamp_min(1e-12).sqrt()
    me = m_eval.s / m_eval.n
    second = m_eval.ss / m_eval.n - torch.outer(mu, me) - torch.outer(me, mu) + torch.outer(mu, mu)
    scale = torch.outer(sd, sd)
    return cov / scale, second / scale


def ll_from_moments(corr, second):
    """Mean log-likelihood per row and dimension of N(0, corr) for data with second moment `second`."""
    dim = corr.shape[0]
    chol = torch.linalg.cholesky(corr + JITTER * torch.eye(dim, dtype=corr.dtype))
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    trace = torch.cholesky_solve(second, chol).diagonal().sum()
    return float(-0.5 * (dim * LOG2PI + logdet + trace) / dim)


def fit_channel(family, value, R, n_images, seed, cache):
    """Correlation of one layer fitted exactly as in the continual-learning runs (cov_ablation)."""
    dim = R.shape[0]
    R32 = R.float()
    if family == 'diag':
        return torch.eye(dim, dtype=torch.float64), 0
    if family in ('lcm', 'lcm_perm'):
        if 'lcm' not in cache:
            model = LCM(dim)
            utils.fit_lcm_correlation(model, R32)
            cache['lcm'] = model
        model = cache['lcm']
        if family == 'lcm_perm':
            model = copy.deepcopy(model)
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                model.a.copy_(model.a[torch.randperm(dim, generator=g)])
        return model.correlation().detach().double(), cov_models.n_params(model)
    if family == 'lcm_fixed':
        model = cov_models.LCMFixedCoords(dim)
        model.fit(R32)
    else:
        kwargs = {'lowrank': {'rank': value}, 'blockdiag': {'block': value}, 'sparse': {'edges_per_dim': value},
                  'shrinkage': {'shrinkage': value}}.get(family, {})
        model = cov_models.build_cov_model(family, dim, **kwargs)
        model.fit(R32, ctx={'n_samples': n_images})
    model.eval()
    return model.correlation().detach().double(), cov_models.n_params(model)


def evaluate_channel(configs, moments, n_images, seed, log):
    """{config: {'val', 'test': mean LL/dim per channel group, 'params_c64'}} over all BatchNorm layers."""
    m_tr, m_va, m_te = moments
    out = {c: {'val': {}, 'test': {}, 'params_c64': []} for c in configs}
    per_layer = {c: {'val': [], 'test': [], 'dim': []} for c in configs}
    for layer in range(len(m_tr)):
        R, S_va = standardised(m_tr[layer], m_va[layer])
        _, S_te = standardised(m_tr[layer], m_te[layer])
        cache = {}
        for c in configs:
            corr, n_par = fit_channel(c[0], c[1], R, n_images, seed + layer, cache)
            per_layer[c]['val'].append(ll_from_moments(corr, S_va))
            per_layer[c]['test'].append(ll_from_moments(corr, S_te))
            per_layer[c]['dim'].append(R.shape[0])
            if R.shape[0] == 64:
                out[c]['params_c64'].append(n_par)
        log('    channel layer {:2d} (C={}) done'.format(layer, R.shape[0]))
    for c in configs:
        dims = np.array(per_layer[c]['dim'])
        for split in ('val', 'test'):
            vals = np.array(per_layer[c][split])
            out[c][split] = {'C={}'.format(d): float(vals[dims == d].mean()) for d in sorted(set(dims.tolist()))}
            out[c][split]['all'] = float(vals.mean())
        out[c]['params_c64'] = float(np.mean(out[c]['params_c64'])) if out[c]['params_c64'] else float('nan')
    return out


# ------------------------------------------------------------------------------------------------
# flattened features
# ------------------------------------------------------------------------------------------------
def load_scan_lcm(path):
    spec = importlib.util.spec_from_file_location('scan_lcm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LCM


@torch.no_grad()
def mean_ll(fn, z, chunk):
    total = 0.0
    for start in range(0, z.shape[0], chunk):
        total += float(fn(z[start:start + chunk]).sum())
    return total / (z.shape[0] * z.shape[1])


def fit_dense(z, value):
    n = z.shape[0]
    R = (z.t() @ z).double().div_(n)
    R.diagonal().fill_(1.0)   # z is standardised with the training statistics
    rho = cov_models.oas_shrinkage(R, n) if value == 'oas' else float(value)
    R.mul_(1.0 - rho)
    R.diagonal().add_(rho + JITTER)
    chol = torch.linalg.cholesky(R)
    del R
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()

    def ll(x):
        t = torch.linalg.solve_triangular(chol, x.double().t(), upper=False)
        return -0.5 * (x.shape[1] * LOG2PI + logdet + (t ** 2).sum(0))
    return ll, {'shrinkage': rho}


def fit_blockdiag(z, block):
    n, dim = z.shape
    if dim % block:
        raise ValueError('D={} is not a multiple of the block size {}'.format(dim, block))
    zb = z.view(n, dim // block, block)
    cov = torch.einsum('nkb,nkc->kbc', zb, zb).double() / n
    cov = cov + JITTER * torch.eye(block, dtype=cov.dtype, device=cov.device)
    chol = torch.linalg.cholesky(cov)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum()

    def ll(x):
        xb = x.double().view(x.shape[0], dim // block, block).permute(1, 2, 0)
        t = torch.linalg.solve_triangular(chol, xb, upper=False)
        return -0.5 * (dim * LOG2PI + logdet + (t ** 2).sum((0, 1)))
    return ll, {}


def fa_ll(x, W, psi):
    dim, k = W.shape
    Wp = W / psi[:, None]
    chol = torch.linalg.cholesky(torch.eye(k, dtype=W.dtype, device=W.device) + W.t() @ Wp)
    logdet = torch.log(psi).sum() + 2.0 * torch.log(torch.diagonal(chol)).sum()
    t = torch.linalg.solve_triangular(chol, (x @ Wp).t(), upper=False)
    return -0.5 * (dim * LOG2PI + logdet + (x ** 2 / psi).sum(1) - (t ** 2).sum(0))


def fit_lowrank(z, rank, iters):
    """Factor analysis Sigma = W W^T + diag(psi) by EM (maximum likelihood), initialised with PPCA."""
    n, dim = z.shape
    _, s, V = torch.pca_lowrank(z, q=rank, center=False, niter=4)
    lam = s ** 2 / n
    noise = ((z ** 2).sum() / n - lam.sum()) / max(dim - rank, 1)
    W = V * (lam - noise).clamp_min(1e-6).sqrt()
    second = (z ** 2).sum(0) / n
    psi = (second - (W ** 2).sum(1)).clamp_min(1e-3)
    eye = torch.eye(rank, dtype=z.dtype, device=z.device)
    for _ in range(iters):
        Wp = W / psi[:, None]
        Minv = torch.linalg.inv(eye + W.t() @ Wp)
        Ez = z @ Wp @ Minv                       # E[h | x], (n, k)
        Ezz = n * Minv + Ez.t() @ Ez             # sum_n E[h h^T | x]
        XEz = z.t() @ Ez                         # (D, k)
        W = XEz @ torch.linalg.inv(Ezz)
        psi = (second - (W * XEz).sum(1) / n).clamp_min(1e-4)
    return (lambda x: fa_ll(x, W, psi)), {}


def fit_scan_lcm(ScanLCM, z_tr, z_va, fixed, args, seed, log):
    """Maximum likelihood with the exact Kalman-scan likelihood. Coordinates start in raster order with
    gaps -log|corr| of neighbouring dimensions (equal gaps and frozen for the fixed-coordinate variant)."""
    n, dim = z_tr.shape
    device = z_tr.device
    torch.manual_seed(seed)
    model = ScanLCM(dim).to(device)
    with torch.no_grad():
        gaps = -torch.log((z_tr[:, 1:] * z_tr[:, :-1]).mean(0).abs().clamp(1e-3, 0.999))
        if fixed:
            gaps = gaps.median().expand(dim - 1).clone()
        a = torch.cat([gaps.new_zeros(1), torch.cumsum(gaps, 0)])
        model.a.copy_(a - a.mean())
        model.w.fill_(math.sqrt(0.5))
        model.u_raw.fill_(math.log(math.expm1(0.5)))
    model.mu = torch.zeros(1, dim, device=device)
    groups = [{'params': [model.u_raw, model.w], 'lr': args.lcm_lr}]
    if fixed:
        model.a.requires_grad_(False)
    else:
        groups.append({'params': [model.a], 'lr': args.lcm_lr_a})
    opt = torch.optim.Adam(groups)
    best, best_state, curve = -float('inf'), None, []
    for epoch in range(args.lcm_epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, args.lcm_batch):
            loss = model.nll(z_tr[order[start:start + args.lcm_batch]]) / dim
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = mean_ll(lambda x: scan_rows_ll(model, x), z_va, args.chunk)
        curve.append(val)
        log('      {} epoch {}: val LL/dim {:.4f}'.format('lcm_fixed' if fixed else 'lcm', epoch + 1, val))
        if val > best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {'val_curve': curve}


def evaluate_flat(configs, feats, args, ScanLCM, seed, log):
    """{config: {'val', 'test': mean LL/dim per stage, 'info'}} for flattened stage outputs."""
    out = {c: {'val': {}, 'test': {}, 'info': {}} for c in configs}
    device = torch.device(args.device)
    for stage, (x_tr, x_va, x_te) in feats.items():
        mu, sd = x_tr.mean(0), x_tr.std(0, unbiased=False)
        keep = sd > 1e-6   # constant dimensions carry no information about correlations
        z_tr, z_va, z_te = [((x[:, keep] - mu[keep]) / sd[keep]).to(device) for x in (x_tr, x_va, x_te)]
        dim = z_tr.shape[1]
        log('    flat {}: D={} ({} constant dimensions dropped)'.format(stage, dim, int((~keep).sum())))
        cache = {}
        for c in configs:
            family, value = c
            start = time.time()
            if family == 'diag':
                fn, info = (lambda x: -0.5 * (x.shape[1] * LOG2PI + (x ** 2).sum(1))), {}
            elif family == 'shrinkage':
                fn, info = fit_dense(z_tr, value)
            elif family == 'blockdiag':
                fn, info = fit_blockdiag(z_tr, value)
            elif family == 'lowrank':
                fn, info = fit_lowrank(z_tr, value, args.fa_iters)
            elif family in ('lcm', 'lcm_perm', 'lcm_fixed'):
                key = 'lcm_fixed' if family == 'lcm_fixed' else 'lcm'
                if key not in cache:
                    cache[key] = fit_scan_lcm(ScanLCM, z_tr, z_va, family == 'lcm_fixed', args, seed, log)
                model, info = cache[key]
                if family == 'lcm_perm':
                    model = copy.deepcopy(model)
                    g = torch.Generator().manual_seed(seed)
                    with torch.no_grad():
                        model.a.copy_(model.a[torch.randperm(dim, generator=g).to(model.a.device)])
                    info = {}
                fn = (lambda m: (lambda x: scan_rows_ll(m, x)))(model)
            else:
                raise ValueError(family)
            out[c]['val'][stage] = mean_ll(fn, z_va, args.chunk)
            out[c]['test'][stage] = mean_ll(fn, z_te, args.chunk)
            out[c]['info'][stage] = dict(info, fit_sec=round(time.time() - start, 2), dim=dim)
            log('      {:<16s} val {:.4f}  test {:.4f}  ({:.1f}s)'.format(
                config_name(*c), out[c]['val'][stage], out[c]['test'][stage], time.time() - start))
            fn = None
        del z_tr, z_va, z_te, cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for c in configs:
        for split in ('val', 'test'):
            out[c][split]['all'] = float(np.mean([v for k, v in out[c][split].items()]))
    return out


def scan_rows_ll(model, x):
    # log_likelihood() returns the batch mean of the per-row sums; mean_ll only needs the total
    return model.log_likelihood(x).expand(x.shape[0])


def params_flat(family, value):
    if family in ('lcm', 'lcm_perm'):
        return 3.0
    if family == 'lcm_fixed':
        return 2.0
    if family == 'lowrank':
        return value + 1.0
    if family == 'blockdiag':
        return (value - 1) / 2.0
    return 0.0 if family == 'diag' else float('nan')


# ------------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------------
SPLITS = ('train', 'val', 'test')


def run_part(part, extracted, n_train, args, ScanLCM, log):
    """Scores of every configuration for one part, in a JSON-friendly form."""
    if part == 'channel':
        mode, configs = 'channel', configs_of(FAMILIES['channel'])
        res = evaluate_channel(configs, [extracted[s][0] for s in SPLITS], n_train, args.split_seed, log)
    else:
        mode, configs = 'flat', configs_of(FAMILIES['flat'])
        feats = {part: tuple(extracted[s][1][part] for s in SPLITS)}
        res = evaluate_flat(configs, feats, args, ScanLCM, args.split_seed, log)
    results = {}
    for c in configs:
        entry = {'family': c[0], 'value': c[1]}
        if mode == 'channel':
            entry.update(val=res[c]['val'], test=res[c]['test'], params=res[c]['params_c64'])
        else:
            params = params_flat(*c)
            entry.update(val=res[c]['val'][part], test=res[c]['test'][part], info=res[c]['info'][part],
                         params=None if np.isnan(params) else params)
        results[config_name(*c)] = entry
    return {'mode': mode, 'part': part, 'seed': args.seed, 'ckpt': args.ckpt, 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='backbone_<task>.pkl written by dump_model')
    parser.add_argument('--seed', type=int, required=True, help='seed of the run the backbone comes from (label)')
    parser.add_argument('--parts', nargs='+', default=['channel', 'stage_1', 'stage_2', 'stage_3'])
    parser.add_argument('--data_path', default='/raid/NFS_SHARE/datasets')
    parser.add_argument('--out_dir', default='heldout_likelihood')
    parser.add_argument('--overwrite', action='store_true', help='recompute parts that already have results')
    parser.add_argument('--n_train', type=int, default=20000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--chunk', type=int, default=500, help='rows per evaluation chunk (flattened features)')
    parser.add_argument('--fa_iters', type=int, default=100)
    parser.add_argument('--lcm_epochs', type=int, default=5)
    parser.add_argument('--lcm_batch', type=int, default=128)
    parser.add_argument('--lcm_lr', type=float, default=2e-2)
    parser.add_argument('--lcm_lr_a', type=float, default=2e-2)
    parser.add_argument('--scan_lcm', default=os.path.join(os.path.dirname(ROOT), 'efficient_inversion', 'resnet34', 'lcm.py'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    part_dir = os.path.join(args.out_dir, 'parts')
    os.makedirs(part_dir, exist_ok=True)
    out_file = lambda part: os.path.join(part_dir, 'seed_{}_{}.json'.format(args.seed, part))
    todo = [p for p in args.parts if args.overwrite or not os.path.exists(out_file(p))]
    if not todo:
        print('All requested parts already done for seed {}; nothing to do.'.format(args.seed))
        return
    if not os.path.exists(args.ckpt):
        raise SystemExit('Missing checkpoint: {}'.format(args.ckpt))

    log_file = open(os.path.join(args.out_dir, 'log_seed_{}_{}.txt'.format(args.seed, '_'.join(todo))), 'a')

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()

    device = torch.device(args.device)
    stages = [p for p in todo if p != 'channel']
    ScanLCM = load_scan_lcm(args.scan_lcm) if stages else None
    train_images, test_images = load_cifar100(args.data_path, True), load_cifar100(args.data_path, False)
    perm = np.random.RandomState(args.split_seed).permutation(len(train_images))
    splits = {'train': train_images[perm[:args.n_train]],
              'val': train_images[perm[args.n_train:args.n_train + args.n_val]],
              'test': test_images[:args.n_test]}
    log('==> held-out likelihood, backbone seed {} ({}): parts {}, {} train / {} val / {} test images'.format(
        args.seed, args.ckpt, todo, len(splits['train']), len(splits['val']), len(splits['test'])))

    backbone = load_backbone(args.ckpt, device)
    extracted = {s: extract(backbone, splits[s], device, args.batch_size, stages, 'channel' in todo) for s in SPLITS}
    del backbone
    for part in todo:
        start = time.time()
        result = run_part(part, extracted, len(splits['train']), args, ScanLCM, log)
        result['args'] = vars(args)
        result['minutes'] = round((time.time() - start) / 60.0, 1)
        tmp = out_file(part) + '.tmp'
        with open(tmp, 'w') as fw:
            json.dump(result, fw, indent=1)
        os.replace(tmp, out_file(part))
        log('    wrote {} ({:.1f} min)'.format(out_file(part), result['minutes']))
    log_file.close()


if __name__ == '__main__':
    main()
