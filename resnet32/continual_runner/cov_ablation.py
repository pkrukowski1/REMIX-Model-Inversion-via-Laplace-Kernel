import json
import os
import time

import torch

import utils
from continual_runner import task_contrastive_cl_runner as runner_module
from inversion import cov_models
from inversion.lcm import LCM

_CTX = {}
_ORIG = {}


def install(cfg):
    if _ORIG:
        return
    _ORIG['fit'] = utils.fit_lcm_correlation
    _ORIG['update_buffer_params'] = runner_module.TaskClassContrastiveRunner.update_buffer_params
    _CTX.update(cfg=cfg, images_seen=0, task=-1, layer=0, local_path=None)

    def make_model(dim, *args, **kwargs):
        return cov_models.build_cov_model(
            cfg.cov_type, dim, rank=cfg.cov_rank, block=cfg.cov_block,
            edges_per_dim=cfg.cov_edges_per_dim, shrinkage=cfg.cov_shrinkage)

    def update_buffer_params(self, train_loader, start_class, end_class):
        _CTX['images_seen'] += len(train_loader.dataset)
        _CTX['task'] += 1
        _CTX['layer'] = 0
        _CTX['local_path'] = self.local_path
        if _CTX['task'] == 0:
            _write_config(self)
        return _ORIG['update_buffer_params'](self, train_loader, start_class, end_class)

    runner_module.LCM = make_model
    utils.fit_lcm_correlation = fit_model
    utils.log_memory_comparison = log_parameter_count
    runner_module.TaskClassContrastiveRunner.update_buffer_params = update_buffer_params
    print('==> covariance ablation: cov_type={} (rank={}, block={}, edges/C={}, shrinkage={})'.format(
        cfg.cov_type, cfg.cov_rank, cfg.cov_block, cfg.cov_edges_per_dim, cfg.cov_shrinkage))


def fit_model(model, R_target, epochs=200, lr=0.01):
    """Drop-in replacement for utils.fit_lcm_correlation(model_lcm, R_target)."""
    start = time.time()
    if type(model) is LCM:
        _ORIG['fit'](model, R_target, epochs=epochs, lr=lr)  # exactly the REMIX fit
        info = {}
    else:
        # n_samples (for OAS): images seen so far; pixels of one image are far from independent.
        info = model.fit(R_target, ctx={'n_samples': _CTX['images_seen']}) or {}
        model.eval()
    with torch.no_grad():
        record = {
            'cov_type': _CTX['cfg'].cov_type,
            'task': _CTX['task'],
            'layer': _CTX['layer'],
            'dim': int(model.dim),
            'n_params': cov_models.n_params(model),
            'fit_sec': round(time.time() - start, 4),
        }
        record.update(cov_models.fit_diagnostics(model.correlation(), R_target))
        record.update(info)
    _CTX['layer'] += 1
    if _CTX['local_path']:
        with open(os.path.join(_CTX['local_path'], 'cov_fit_log.jsonl'), 'a') as fw:
            fw.write(json.dumps(record) + '\n')


def log_parameter_count(local_path, lcms):
    """Replacement for utils.log_memory_comparison (which assumes LCM attributes)."""
    if lcms is None:
        print('==> no correlation models (alpha_frob = 0). Skipping parameter log.')
        return
    modules = list(lcms.values()) if isinstance(lcms, torch.nn.ModuleDict) else list(lcms)
    n_model = sum(cov_models.n_params(m) for m in modules)
    n_dense = sum(int(m.dim) * (int(m.dim) - 1) // 2 for m in modules)
    with open(os.path.join(local_path, 'memory_comparison.txt'), 'a') as f:
        f.write('cov_type={} layers={} free parameters={} (dense correlation: {})\n'.format(
            _CTX['cfg'].cov_type, len(modules), n_model, n_dense))


def _write_config(runner):
    cfg = _CTX['cfg']
    config = {
        'cov_type': cfg.cov_type if runner.inversion_params.get('alpha_frob', 0.0) > 0.0 else 'none (PMI, alpha_frob=0)',
        'alpha_frob': runner.inversion_params.get('alpha_frob'),
        'cov_rank': cfg.cov_rank,
        'cov_block': cfg.cov_block,
        'cov_edges_per_dim': cfg.cov_edges_per_dim,
        'cov_shrinkage': cfg.cov_shrinkage,
    }
    with open(os.path.join(runner.local_path, 'cov_config.json'), 'w') as fw:
        json.dump(config, fw, indent=2)
