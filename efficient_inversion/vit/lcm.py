import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class LCM(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.u_raw = nn.Parameter(torch.zeros(dim))
        self.w = nn.Parameter(torch.ones(dim) * 0.1)
        self.a = nn.Parameter(torch.linspace(-1, 1, dim))
        self.mu = None

    def _parallel_matrix_prefix_prod(self, A):
        A_double = A.to(torch.float64) 
        n = A_double.shape[-3]
        num_steps = int(math.ceil(math.log2(n)))
        res = A_double.clone()
        
        for i in range(num_steps):
            step = 2**i
            if step >= n:
                break
            left = res[..., :-step, :, :]
            right = res[..., step:, :, :]
            combined = torch.matmul(right, left)
            
            norm = combined.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-20)
            combined = combined / norm
            
            res = torch.cat([res[..., :step, :, :], combined], dim=-3)
            
        return res.to(torch.float32)

    def log_likelihood(self, x):
        a_s, idx = torch.sort(self.a)
        x_s = x[:, idx]
        mu_s = self.mu[:, idx]
        
        u_s = F.softplus(self.u_raw[idx]) + 1e-5
        w_s = self.w[idx] 
        
        dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-6)
        phi2 = torch.exp(-2.0 * dt)
        q = (1.0 - phi2).clamp(min=1e-8)
        
        M = torch.zeros(self.dim, 2, 2, device=x.device, dtype=torch.float64)
        
        M[0, 0, 1] = u_s[0]
        M[0, 1, 1] = w_s[0]**2 + u_s[0]
        
        w2 = w_s[1:]**2
        M[1:, 0, 0] = u_s[1:] * phi2
        M[1:, 0, 1] = u_s[1:] * q
        M[1:, 1, 0] = w2 * phi2
        M[1:, 1, 1] = w2 * q + u_s[1:]

        M_cum = self._parallel_matrix_prefix_prod(M)
        
        denom = (M_cum[:, 1, 0] + M_cum[:, 1, 1]).clamp(min=1e-12)
        P_filt = ((M_cum[:, 0, 0] + M_cum[:, 0, 1]) / denom).clamp(min=0.0).float()
        
        P_prev_filt = torch.ones_like(P_filt)
        P_prev_filt[1:] = P_filt[:-1]
        
        phi2_vec = torch.ones(self.dim, device=x.device)
        phi2_vec[1:] = phi2
        q_vec = torch.zeros(self.dim, device=x.device)
        q_vec[1:] = q
        
        P_pred = (phi2_vec * P_prev_filt + q_vec).clamp(min=1e-8)
        
        S_all = (w_s**2 * P_pred + u_s).clamp(min=1e-6) 
        K_all = (P_pred * w_s) / S_all

        B = x.size(0)
        diff = x_s - mu_s
        
        alpha = torch.ones(self.dim, device=x.device)
        alpha[1:] = torch.exp(-dt) * (1.0 - K_all[1:] * w_s[1:])
        beta = K_all * diff 
        
        M_m = torch.zeros(B, self.dim, 2, 2, device=x.device, dtype=torch.float64)
        M_m[:, :, 0, 0] = alpha.unsqueeze(0)
        M_m[:, :, 0, 1] = beta
        M_m[:, :, 1, 1] = 1.0
        
        M_m_cum = self._parallel_matrix_prefix_prod(M_m)
        
        denom_m = M_m_cum[:, :, 1, 1].clamp(min=1e-12)
        m = (M_m_cum[:, :, 0, 1] / denom_m).float()
        
        m_prev = torch.zeros_like(m)
        m_prev[:, 1:] = m[:, :-1]
        
        phi_vec = torch.ones(self.dim, device=x.device)
        phi_vec[1:] = torch.exp(-dt)
        
        v = diff - w_s * phi_vec * m_prev
        ll_elements = -0.5 * (math.log(2 * math.pi) + torch.log(S_all) + (v**2) / S_all)
        
        return ll_elements.sum(dim=1).mean()
    
    def nll(self, x):
        return -self.log_likelihood(x)