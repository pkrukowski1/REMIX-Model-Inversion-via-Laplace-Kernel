import torch
import torch.nn.functional as F
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def frobenius_dist_reduced(
    u: torch.Tensor,          # (n,)
    a: torch.Tensor,          # (n,)
    w: torch.Tensor,          # (n,)
    V: torch.Tensor,          # (B,n)
):
    B = V.shape[0]

    term_b = torch.sum(u**2)
    term_A2_w = xAx_sum(2 * a, w**2)
    term_bw = 2.0 * torch.sum(u * w**2)

    v_sq_sum = torch.sum(V**2, dim=0)
    term_bv = -(2.0 / B) * torch.sum(u * v_sq_sum)

    VW = V * w                                 # (B,n)
    term_Awv = -(2.0 / B) * xAx_sum(a, VW)

    return (
        term_b
        + term_A2_w
        + term_bw
        + term_bv
        + term_Awv
    )

def frobenius_dist(d, a, w, V):
    B = V.shape[0]
    return frobenius_dist_reduced(d, a, w, V) + torch.sum(((V.T @ V) / B)**2)

def xAx_sum(a, x):
    return torch.abs(torch.sum(x * x_A_complex_stable(x, a)))

def x_A_complex_stable(x, a):
    """
    Computes A @ x efficiently and stably for Laplace kernel A_ij = exp(-|a_i - a_j|).
    Uses positive/negative splitting and a noise floor to avoid logcumsumexp NaN gradients.
    """
    a_s, perm = torch.sort(a)
    x_s = x[..., perm]

    alpha = a_s.view(*([1] * (x.dim() - 1)), -1)

    x_pos = F.relu(x_s)
    x_neg = F.relu(-x_s)

    eps = 1e-8
    log_x_pos = torch.log(x_pos + eps)
    log_x_neg = torch.log(x_neg + eps)

    # ----------------------------------------------------
    # UPPER PART: sum_{j=1}^i exp(a_j - a_i) x_j
    # ----------------------------------------------------
    log_terms_pos = log_x_pos + alpha
    log_terms_neg = log_x_neg + alpha

    log_prefix_pos = torch.logcumsumexp(log_terms_pos, dim=-1)
    log_prefix_neg = torch.logcumsumexp(log_terms_neg, dim=-1)

    upper_pos = torch.exp(log_prefix_pos - alpha)
    upper_neg = torch.exp(log_prefix_neg - alpha)
    upper = upper_pos - upper_neg

    # ----------------------------------------------------
    # LOWER PART: sum_{j=i+1}^n exp(a_i - a_j) x_j
    # ----------------------------------------------------
    log_terms2_pos = log_x_pos - alpha
    log_terms2_neg = log_x_neg - alpha

    # reverse -> cumsum -> reverse back
    log_s_pos = torch.flip(torch.logcumsumexp(torch.flip(log_terms2_pos, dims=[-1]), dim=-1), dims=[-1])
    log_s_neg = torch.flip(torch.logcumsumexp(torch.flip(log_terms2_neg, dims=[-1]), dim=-1), dims=[-1])

    # shift right by 1
    log_s_shift_pos = torch.empty_like(log_s_pos)
    log_s_shift_pos[..., :-1] = log_s_pos[..., 1:]
    log_s_shift_pos[..., -1] = -float('inf')

    log_s_shift_neg = torch.empty_like(log_s_neg)
    log_s_shift_neg[..., :-1] = log_s_neg[..., 1:]
    log_s_shift_neg[..., -1] = -float('inf')

    lower_pos = torch.exp(log_s_shift_pos + alpha)
    lower_neg = torch.exp(log_s_shift_neg + alpha)
    lower = lower_pos - lower_neg

    # ----------------------------------------------------
    # COMBINE
    # ----------------------------------------------------
    y_s = upper + lower

    inv_perm = torch.argsort(perm)
    y = y_s[..., inv_perm]

    return y

def diversity_loss(features):
    batch_size = features.size(0)
    if batch_size <= 1: 
        return torch.tensor(0.0).to(DEVICE)
    f = F.normalize(features.view(batch_size, -1), p=2, dim=1)
    cosine_sim = torch.mm(f, f.t())
    mask = torch.eye(batch_size).to(DEVICE)
    return (cosine_sim * (1 - mask)).sum() / (batch_size * (batch_size - 1))

def baseline_diagonal_ll(x, mu_emp, var_emp):
    std = torch.sqrt(torch.clamp(var_emp, min=1e-5)).unsqueeze(0)
    z = (x - mu_emp) / std
    ll = -0.5 * math.log(2 * math.pi) - torch.log(std) - 0.5 * (z ** 2)
    return ll.sum(dim=1).mean()