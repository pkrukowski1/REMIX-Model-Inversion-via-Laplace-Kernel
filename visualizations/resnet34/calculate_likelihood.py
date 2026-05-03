import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, glob
import math
import matplotlib.pyplot as plt
import random

from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image

torch.manual_seed(1)

# ============================================================
# 1. CONFIGURATION
# ============================================================
os.environ["TMPDIR"] = "/tmp" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wszystkie 4 klasy 
TARGET_WNIDS = [
    "n02113978", # Dog
    "n04285008", # Cars
    "n01882714", # Koala
    "n03457902"  # Greenhouse
]

BASE_DATA_DIR = "/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/"
STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
LCM_EPOCHS = 100

print(f"Running LCM Fitting and Log-Likelihood Analysis on 4 Classes on: {DEVICE}")

# ============================================================
# 2. MODEL & HOOKS
# ============================================================
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(DEVICE)
model.eval()

for module in model.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False

for p in model.parameters(): 
    p.requires_grad = False

class FeatureHook:
    def __init__(self, module):
        self.features = None
        module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inp, out): 
        self.features = out.clone()

hooks_lcm = {f"layer{i}": FeatureHook(getattr(model, f"layer{i}")) for i in range(1, 5)}
LAYERS = list(hooks_lcm.keys())

with torch.no_grad(): 
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    model(dummy)

layer_dims = {l: hooks_lcm[l].features.numel() // 1 for l in LAYERS}

# ============================================================
# 3. LCM MODULE & HELPER FUNCTIONS
# ============================================================

class LCM(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
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
        M[0] = torch.eye(2, device=x.device, dtype=torch.float64)
        
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
        
        phi2_vec = torch.ones(self.dim, device=x.device); phi2_vec[1:] = phi2
        q_vec = torch.zeros(self.dim, device=x.device); q_vec[1:] = q
        
        P_pred = (phi2_vec * P_prev_filt + q_vec).clamp(min=1e-8)
        S_all = (w_s**2 * P_pred + u_s).clamp(min=1e-6) 
        K_all = (P_pred * w_s) / S_all

        B = x.size(0)
        diff = x_s - mu_s
        
        alpha = torch.ones(self.dim, device=x.device)
        alpha[1:] = torch.exp(-dt) * (1.0 - K_all[1:] * w_s[1:])
        beta = K_all * diff # Zależne od batcha (B, D)
        
        M_m = torch.zeros(B, self.dim, 2, 2, device=x.device, dtype=torch.float64)
        M_m[:, :, 0, 0] = alpha.unsqueeze(0)
        M_m[:, :, 0, 1] = beta
        M_m[:, :, 1, 1] = 1.0
        
        M_m_cum = self._parallel_matrix_prefix_prod(M_m)
        
        denom_m = M_m_cum[:, :, 1, 1].clamp(min=1e-12)
        m = (M_m_cum[:, :, 0, 1] / denom_m).float()
        
        m_prev = torch.zeros_like(m)
        m_prev[:, 1:] = m[:, :-1]
        
        phi_vec = torch.ones(self.dim, device=x.device); phi_vec[1:] = torch.exp(-dt)
        v = diff - w_s * phi_vec * m_prev
        
        ll_elements = -0.5 * (math.log(2*math.pi) + torch.log(S_all) + (v**2)/S_all)
        return ll_elements.sum(dim=1).mean()

    
def baseline_diagonal_ll(x, mu_emp, var_emp):
    std = torch.sqrt(torch.clamp(var_emp, min=1e-5)).unsqueeze(0)
    z = (x - mu_emp) / std
    ll = -0.5 * math.log(2 * math.pi) - torch.log(std) - 0.5 * (z ** 2)
    return ll.sum(dim=1).mean()

def frobenius_dist_reduced(u, a, w, V):
    B = V.shape[0]
    term_b = torch.sum(u**2)
    term_A2_w = xAx_sum(2 * a, w**2)
    term_bw = 2.0 * torch.sum(u * w**2)
    v_sq_sum = torch.sum(V**2, dim=0)
    term_bv = -(2.0 / B) * torch.sum(u * v_sq_sum)
    VW = V * w                                
    term_Awv = -(2.0 / B) * xAx_sum(a, VW)
    return term_b + term_A2_w + term_bw + term_bv + term_Awv

def xAx_sum(a, x):
    return torch.abs(torch.sum(x * x_A_complex_stable(x, a)))

def x_A_complex_stable(x, a):
    a_s, perm = torch.sort(a)
    x_s = x[..., perm]
    alpha = a_s.view(*([1] * (x.dim() - 1)), -1)
    x_pos = F.relu(x_s)
    x_neg = F.relu(-x_s)
    eps = 1e-8
    log_x_pos = torch.log(x_pos + eps)
    log_x_neg = torch.log(x_neg + eps)
    
    log_terms_pos = log_x_pos + alpha
    log_terms_neg = log_x_neg + alpha
    log_prefix_pos = torch.logcumsumexp(log_terms_pos, dim=-1)
    log_prefix_neg = torch.logcumsumexp(log_terms_neg, dim=-1)
    upper_pos = torch.exp(log_prefix_pos - alpha)
    upper_neg = torch.exp(log_prefix_neg - alpha)
    upper = upper_pos - upper_neg

    log_terms2_pos = log_x_pos - alpha
    log_terms2_neg = log_x_neg - alpha
    log_s_pos = torch.flip(torch.logcumsumexp(torch.flip(log_terms2_pos, dims=[-1]), dim=-1), dims=[-1])
    log_s_neg = torch.flip(torch.logcumsumexp(torch.flip(log_terms2_neg, dims=[-1]), dim=-1), dims=[-1])
    
    log_s_shift_pos = torch.empty_like(log_s_pos)
    log_s_shift_pos[..., :-1] = log_s_pos[..., 1:]
    log_s_shift_pos[..., -1] = -float('inf')
    log_s_shift_neg = torch.empty_like(log_s_neg)
    log_s_shift_neg[..., :-1] = log_s_neg[..., 1:]
    log_s_shift_neg[..., -1] = -float('inf')
    
    lower_pos = torch.exp(log_s_shift_pos + alpha)
    lower_neg = torch.exp(log_s_shift_neg + alpha)
    lower = lower_pos - lower_neg
    y_s = upper + lower
    inv_perm = torch.argsort(perm)
    return y_s[..., inv_perm]

lcms = {l: LCM(layer_dims[l]).to(DEVICE) for l in LAYERS}

# ============================================================
# 4. DATA EXTRACTION (ALL 4 CLASSES)
# ============================================================
print("\nGathering images for Dog, Car, Koala, and Greenhouse...")
image_paths = []
for wnid in TARGET_WNIDS:
    folder_path = os.path.join(BASE_DATA_DIR, wnid)
    paths = glob.glob(os.path.join(folder_path, "*.JPEG"))
    image_paths.extend(paths)

random.shuffle(image_paths)
print(f"Total images found across 4 classes: {len(image_paths)}")

transform_target = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(*STATS)
])

class MultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB'))

dataset = MultiClassDataset(image_paths, transform=transform_target)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

print("Extracting feature representations...")
target_means_lcm = {} 
target_vars_lcm = {}

with torch.no_grad():
    all_f = {l: [] for l in LAYERS}
    for i, x in enumerate(loader):
        model(x.to(DEVICE))
        for l in LAYERS: 
            all_f[l].append(hooks_lcm[l].features.view(x.size(0), -1).cpu())
    
    for l in LAYERS:
        f_cat = torch.cat(all_f[l], dim=0)
        target_means_lcm[l] = f_cat.mean(0).to(DEVICE)
        target_vars_lcm[l] = f_cat.var(0, unbiased=False).to(DEVICE) 
        
        lcms[l].mu = target_means_lcm[l].unsqueeze(0)
        
        # lcms[l].u_raw.data.fill_(-10.0)
        # u_initial_contribution = F.softplus(torch.tensor(-10.0, device=DEVICE)) + 1e-5
        # empirical_var = target_vars_lcm[l]
        # w_squared = torch.clamp(empirical_var - u_initial_contribution, min=1e-6)
        # lcms[l].w.data.copy_(torch.sqrt(w_squared))

# ============================================================
# 5. TRAIN LCM (FROBENIUS) & MEASURE LL
# ============================================================
LCM_PATH = f"./lcm_resnet34_4_classes.pth"

hist_lcm_ll = {l: [] for l in LAYERS}
hist_base_ll = {l: [] for l in LAYERS}

print(f"\n--- LCM training (epochs: {LCM_EPOCHS}) ---")
opts = {l: optim.Adam([
    {'params': lcms[l].u_raw, 'lr': 1e-2},
    {'params': lcms[l].w, 'lr': 1e-2},
    {'params': lcms[l].a, 'lr': 0.12},
], betas=(0.9, 0.999)) for l in LAYERS}

extracted_features_gpu = {l: torch.cat(all_f[l], dim=0).to(DEVICE) for l in LAYERS}
BATCH_SIZE_FIT = 128

num_samples = extracted_features_gpu['layer1'].size(0)
batches_per_epoch = math.ceil(num_samples / BATCH_SIZE_FIT)
TOTAL_STEPS = LCM_EPOCHS * batches_per_epoch

def lr_lambda(step):
    t = step / TOTAL_STEPS
    return max(0.9*(1.0 - t)**2 + 0.28, 0.0)

schedulers = {l: optim.lr_scheduler.LambdaLR(opts[l], lr_lambda) for l in LAYERS}

for epoch in range(LCM_EPOCHS):
    epoch_lcm_ll = {l: 0.0 for l in LAYERS}
    epoch_base_ll = {l: 0.0 for l in LAYERS}
    total_frob_loss = {l: 0.0 for l in LAYERS}
    batches = 0
    
    num_samples = extracted_features_gpu['layer1'].size(0)
    indices = torch.randperm(num_samples)
    
    for start_idx in range(0, num_samples, BATCH_SIZE_FIT):
        batch_idx = indices[start_idx:start_idx + BATCH_SIZE_FIT]
        
        for l in LAYERS:
            f = extracted_features_gpu[l][batch_idx]
            V_feat = f - lcms[l].mu 
            D = layer_dims[l]
            
            opts[l].zero_grad()
            u = F.softplus(lcms[l].u_raw) + 1e-5
            frob = frobenius_dist_reduced(u, lcms[l].a, lcms[l].w, V_feat)
            loss = frob / D
            loss.backward()
            opts[l].step()
            schedulers[l].step()

            total_frob_loss[l] += loss.item()
            
            with torch.no_grad():
                l_ll = lcms[l].log_likelihood(f)
                b_ll = baseline_diagonal_ll(f, target_means_lcm[l], target_vars_lcm[l])
                epoch_lcm_ll[l] += l_ll.item()
                epoch_base_ll[l] += b_ll.item()
                
        batches += 1

    msg = f"Ep {epoch+1:2d}/{LCM_EPOCHS}"
    for l in LAYERS:
        avg_lcm = epoch_lcm_ll[l] / batches
        avg_base = epoch_base_ll[l] / batches
        hist_lcm_ll[l].append(avg_lcm)
        hist_base_ll[l].append(avg_base)
        
        msg += f" | {l.upper()}: LCM {avg_lcm/1000:,.0f}k (Base {avg_base/1000:,.0f}k)"
    print(msg)
    
# ============================================================
# 6. LL PLOTS
# ============================================================
print("\n--- Generating Log-Likelihood Plots ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, l in enumerate(LAYERS):
    ax = axes[i]
    epochs = np.arange(1, LCM_EPOCHS + 1)
    
    ax.plot(epochs, hist_base_ll[l], linestyle='--', linewidth=2.5, color='#7f8c8d', label='Diagonal Covariance')
    ax.plot(epochs, hist_lcm_ll[l], linestyle='-', linewidth=2.5, color='#e67e22', label='Full Covariance')
    
    # CRITICAL ADDITION: Injecting the total dimensions (D) formatted with commas
    total_dims = layer_dims[l]
    ax.set_title(f'ResNet34 - {l.capitalize()} ($D = {total_dims:,}$)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Log-Likelihood', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Format Y-axis ticks with commas for massive numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    if i == 0:
        ax.legend(fontsize=12, loc='lower right')

# plt.tight_layout()

plot_path = "./log_likelihood_comparison_4_classes.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Plots saved to {plot_path}")

# ============================================================
# 7. EXPORT LL VALUES TO TXT
# ============================================================
print("\n--- Exporting Log-Likelihood values to TXT ---")
txt_path = "./log_likelihood_values_4_classes.txt"

with open(txt_path, "w") as f:
    header = ["Epoch"]
    for l in LAYERS:
        header.extend([f"{l}_Baseline", f"{l}_LCM"])
    f.write("\t".join(header) + "\n")
    
    for epoch in range(LCM_EPOCHS):
        row = [str(epoch + 1)]
        for l in LAYERS:
            row.append(f"{hist_base_ll[l][epoch]:.2f}")
            row.append(f"{hist_lcm_ll[l][epoch]:.2f}")
        f.write("\t".join(row) + "\n")

print(f"Log-Likelihood values securely saved to {txt_path}")