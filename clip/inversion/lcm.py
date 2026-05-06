import torch
import torch.nn as nn
import torch.nn.functional as F

class LCM(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.u_raw = nn.Parameter(torch.zeros(dim))
        self.w = nn.Parameter(torch.ones(dim) * 0.1)
        self.a = nn.Parameter(torch.linspace(-1, 1, dim))

    def laplace_kernel(self):
        return torch.exp(-torch.abs(self.a[:, None] - self.a[None, :]))

    def covariance(self):
        u = F.softplus(self.u_raw) + self.eps
        K = self.laplace_kernel()
        w_outer = torch.outer(self.w, self.w)
        
        Sigma = torch.diag(u) + (K * w_outer)
        return Sigma
    
    def correlation(self):
        Sigma = self.covariance()
        diag = torch.diag(Sigma)
        std = torch.sqrt(diag)
        
        std_outer = torch.outer(std, std)
        R = Sigma / torch.clamp(std_outer, min=self.eps)
        
        return R - torch.diag(torch.diag(R)) + torch.eye(self.dim, device=R.device)

    def forward(self, x):
        return self.correlation()