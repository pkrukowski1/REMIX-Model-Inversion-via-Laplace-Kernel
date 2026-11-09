# -*-coding:utf8-*-
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inversion.lcm import LCM

COV_TYPES = ('lcm', 'lcm_fixed', 'diag', 'lowrank', 'blockdiag', 'sparse', 'shrinkage', 'full')

# Eigenvalue floor for the dense-matrix families. Keeps the precision used by the NLL bounded
# (the LCM is bounded below by its diagonal noise term).
MIN_EIG = 1e-3


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _sym(R):
    return 0.5 * (R + R.transpose(-1, -2))


def to_correlation(S, eps=1e-12):
    """Covariance -> correlation (out of place, differentiable)."""
    d = torch.sqrt(torch.clamp(torch.diagonal(S), min=eps))
    R = _sym(S / (d[:, None] * d[None, :]))
    return R - torch.diag(torch.diagonal(R)) + torch.eye(R.shape[0], dtype=R.dtype, device=R.device)


def psd_correlation(R, min_eig=MIN_EIG):
    """Symmetrise, clip eigenvalues at `min_eig` and renormalise to a unit diagonal."""
    R = _sym(R.detach().double())
    evals, evecs = torch.linalg.eigh(R)
    if evals.min() < min_eig:
        R = (evecs * evals.clamp(min=min_eig)) @ evecs.T
    return to_correlation(R)


def spectral_order(R):
    """Order channels by the Fiedler vector of the graph with weights |R_ij| (spectral seriation)."""
    W = R.detach().double().abs()
    W = W - torch.diag(torch.diagonal(W))
    L = torch.diag(W.sum(dim=1)) - W
    _, evecs = torch.linalg.eigh(L)
    return torch.argsort(evecs[:, 1])


def oas_shrinkage(R, n_samples):
    """Oracle Approximating Shrinkage intensity (Chen et al., 2010; same formula as sklearn.covariance.oas)
    for a correlation matrix, i.e. shrinkage target = identity."""
    if n_samples is None or n_samples <= 1:
        raise ValueError('OAS shrinkage needs the number of samples')
    R = R.detach().double()
    p = R.shape[0]
    mu = torch.trace(R) / p
    alpha = (R ** 2).mean()
    num = alpha + mu ** 2
    den = (n_samples + 1.0) * (alpha - mu ** 2 / p)
    return 1.0 if den.item() <= 0 else float(min((num / den).item(), 1.0))


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def covariance_selection(S, adj, max_iter=200, tol=1e-7):
    """Maximum-likelihood covariance whose precision is zero outside the graph `adj` (Dempster's
    covariance selection; Hastie, Tibshirani & Friedman, ESL Alg. 17.1). Keeps S on the diagonal and on
    the edges, i.e. an un-shrunk refit of the graph selected by the graphical lasso."""
    p = S.shape[0]
    W = S.copy()
    for _ in range(max_iter):
        W_old = W.copy()
        for j in range(p):
            idx = np.r_[0:j, j + 1:p]
            nb = idx[adj[j, idx]]
            if len(nb):
                beta = np.linalg.solve(W[np.ix_(nb, nb)], S[nb, j])
                w12 = W[np.ix_(idx, nb)] @ beta
            else:
                w12 = np.zeros(p - 1)
            W[idx, j] = w12
            W[j, idx] = w12
        if np.abs(W - W_old).max() < tol:
            break
    return W


def n_params(model):
    if hasattr(model, 'n_params'):
        return int(model.n_params())
    return 3 * int(model.dim)  # inversion.lcm.LCM: u, w, a


@torch.no_grad()
def fit_diagnostics(R_fit, R_target):
    """How well the family approximates the (merged) empirical correlation it was fitted to."""
    R_fit = R_fit.detach().double()
    R_target = R_target.detach().double().to(R_fit.device)
    off = ~torch.eye(R_fit.shape[0], dtype=torch.bool, device=R_fit.device)
    err = (R_fit - R_target)[off].norm()
    ref = R_target[off].norm().clamp_min(1e-12)
    return {
        'rel_offdiag_err': float(err / ref),
        'min_eig': float(torch.linalg.eigvalsh(_sym(R_fit)).min()),
        'mean_abs_offdiag_target': float(R_target[off].abs().mean()),
        'mean_abs_offdiag_fit': float(R_fit[off].abs().mean()),
    }


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
class _BufferCorr(nn.Module):
    """Base class for families that store the fitted correlation matrix directly."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('R', torch.eye(dim))

    def correlation(self):
        return self.R

    def _store(self, R):
        self.R.copy_(R.to(device=self.R.device, dtype=self.R.dtype))


class DiagCorr(_BufferCorr):
    """Identity correlation: the NLL becomes a diagonal Gaussian NLL (same weight, same hook)."""
    kind = 'diag'

    def fit(self, R_target, ctx=None):
        return {}

    def n_params(self):
        return 0


class DenseCorr(_BufferCorr):
    """Dense correlation. shrinkage=0 ('full'): the merged empirical correlation itself;
    shrinkage='oas' ('shrinkage'): OAS shrinkage towards the identity; a float: fixed intensity."""

    def __init__(self, dim, shrinkage=0.0):
        super().__init__(dim)
        self.shrinkage = shrinkage
        self.kind = 'full' if (not isinstance(shrinkage, str) and float(shrinkage) == 0.0) else 'shrinkage'

    def fit(self, R_target, ctx=None):
        R = _sym(R_target.detach().double())
        if isinstance(self.shrinkage, str):
            if self.shrinkage != 'oas':
                raise ValueError('unknown shrinkage: {}'.format(self.shrinkage))
            rho = oas_shrinkage(R, (ctx or {}).get('n_samples'))
        else:
            rho = float(self.shrinkage)
        eye = torch.eye(self.dim, dtype=R.dtype, device=R.device)
        self._store(psd_correlation((1.0 - rho) * R + rho * eye))
        return {'shrinkage': rho}

    def n_params(self):
        return self.dim * (self.dim - 1) // 2


class LowRankCorr(nn.Module):
    """Factor model Sigma = U U^T + diag(d) normalised to a correlation (C(k+1) parameters).

    Fitted with the same objective and budget as the LCM (MSE to the merged correlation, Adam,
    200 steps, lr 0.01), but initialised at every task from the top-k eigenpairs of the target,
    which gives this baseline the best starting point."""
    kind = 'lowrank'

    def __init__(self, dim, rank=2, eps=1e-6):
        super().__init__()
        self.dim, self.rank, self.eps = dim, rank, eps
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.u_raw = nn.Parameter(torch.zeros(dim))

    def correlation(self):
        S = torch.diag(F.softplus(self.u_raw) + self.eps) + self.U @ self.U.T
        return to_correlation(S)

    @torch.no_grad()
    def _init_from(self, R):
        evals, evecs = torch.linalg.eigh(_sym(R.double()))
        top = torch.argsort(evals, descending=True)[:self.rank]
        U = evecs[:, top] * evals[top].clamp(min=0.0).sqrt()
        resid = (1.0 - (U ** 2).sum(dim=1)).clamp(min=1e-3)
        self.U.copy_(U.to(self.U))
        self.u_raw.copy_(inv_softplus(resid).to(self.u_raw))

    def fit(self, R_target, ctx=None, epochs=200, lr=0.01):
        R = R_target.detach().to(self.U)
        self._init_from(R)
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(self.correlation(), R)
            loss.backward()
            opt.step()
        self.eval()
        return {}

    def n_params(self):
        return self.dim * (self.rank + 1)


class BlockDiagCorr(_BufferCorr):
    """Block-diagonal correlation. Channels have no natural order, so they are first ordered by
    spectral seriation of |R| (strongly correlated channels end up in the same block), then cut into
    blocks of `block` channels. Within blocks the merged correlation is kept exactly (the Frobenius
    projection onto this block structure); across blocks it is zero."""
    kind = 'blockdiag'

    def __init__(self, dim, block=8):
        super().__init__(dim)
        self.block = block
        self.register_buffer('order', torch.arange(dim))

    def fit(self, R_target, ctx=None):
        R = _sym(R_target.detach().double())
        order = spectral_order(R).to(R.device)
        mask = torch.zeros(self.dim, self.dim, dtype=torch.bool, device=R.device)
        for s in range(0, self.dim, self.block):
            idx = order[s:s + self.block]
            mask[idx[:, None], idx[None, :]] = True
        self._store(psd_correlation(torch.where(mask, R, torch.zeros_like(R))))
        self.order.copy_(order.to(self.order.device))
        return {'n_blocks': int(math.ceil(self.dim / self.block))}

    def n_params(self):
        sizes = [min(self.block, self.dim - s) for s in range(0, self.dim, self.block)]
        return sum(b * (b - 1) // 2 for b in sizes)


class SparsePrecisionCorr(_BufferCorr):
    """Gaussian graphical model with a sparse precision matrix (Friedman et al., 2008). The graph is
    selected with the graphical lasso on the merged correlation, its L1 penalty chosen by bisection so the
    precision has about `edges_per_dim * C` non-zero off-diagonal entries (e=2: ~3C parameters, like the
    LCM). The covariance is then refitted by maximum likelihood on that graph (no L1 shrinkage bias)."""
    kind = 'sparse'

    def __init__(self, dim, edges_per_dim=2.0, max_bisect=16, rel_tol=0.1):
        super().__init__(dim)
        self.edges_per_dim, self.max_bisect, self.rel_tol = edges_per_dim, max_bisect, rel_tol
        self.n_edges = 0
        self.alpha = None

    def fit(self, R_target, ctx=None):
        from sklearn.covariance import graphical_lasso

        S = _sym(R_target.detach().double()).cpu().numpy()
        np.fill_diagonal(S, 1.0)
        max_edges = self.dim * (self.dim - 1) // 2
        target = int(round(min(self.edges_per_dim * self.dim, max_edges)))
        iu = np.triu_indices(self.dim, k=1)
        lo, hi = 1e-4, float(np.abs(S[iu]).max())  # alpha >= max |R_ij| gives an empty graph
        alpha = self.alpha if self.alpha is not None and lo < self.alpha < hi else math.sqrt(lo * hi)
        best = None
        for _ in range(self.max_bisect):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _, prec = graphical_lasso(S, alpha=alpha, mode='cd', max_iter=200, tol=1e-4)
                edges = int((np.abs(prec[iu]) > 1e-8).sum())
            except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                prec, edges = None, None
            if prec is not None and (best is None or abs(edges - target) < abs(best[2] - target)):
                best = (prec, alpha, edges)
            if edges is not None and abs(edges - target) <= self.rel_tol * target:
                break
            if edges is None or edges > target:
                lo = alpha  # unstable or too dense: more penalty
            else:
                hi = alpha
            alpha = math.sqrt(lo * hi)
        if best is None:
            raise RuntimeError('graphical lasso failed for every penalty value')
        prec, self.alpha, edges = best
        adj = np.abs(prec) > 1e-8
        np.fill_diagonal(adj, False)
        cov = covariance_selection(S, adj)
        self.n_edges = edges
        self._store(psd_correlation(torch.from_numpy(cov)))
        return {'glasso_alpha': self.alpha, 'edges': edges, 'target_edges': target}

    def n_params(self):
        return int(self.n_edges) + self.dim


class LCMFixedCoords(LCM):
    """LCM whose coordinates stay at linspace(-1, 1): tests whether *learning* the 1D order matters."""
    kind = 'lcm_fixed'

    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps)
        self.a.requires_grad_(False)

    def fit(self, R_target, ctx=None, epochs=200, lr=0.01):
        # same loop as utils.fit_lcm_correlation, without the coordinates
        self.train()
        opt = torch.optim.Adam([self.w, self.u_raw], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.mse_loss(self.correlation(), R_target)
            loss.backward()
            opt.step()
        self.eval()
        return {}

    def n_params(self):
        return 2 * self.dim


def build_cov_model(cov_type, dim, rank=2, block=8, edges_per_dim=2.0, shrinkage='oas'):
    if cov_type == 'lcm':
        return LCM(dim)
    if cov_type == 'lcm_fixed':
        return LCMFixedCoords(dim)
    if cov_type == 'diag':
        return DiagCorr(dim)
    if cov_type == 'lowrank':
        return LowRankCorr(dim, rank=rank)
    if cov_type == 'blockdiag':
        return BlockDiagCorr(dim, block=block)
    if cov_type == 'sparse':
        return SparsePrecisionCorr(dim, edges_per_dim=edges_per_dim)
    if cov_type == 'shrinkage':
        return DenseCorr(dim, shrinkage=shrinkage)
    if cov_type == 'full':
        return DenseCorr(dim, shrinkage=0.0)
    raise ValueError('unknown cov_type {} (choose from {})'.format(cov_type, COV_TYPES))
