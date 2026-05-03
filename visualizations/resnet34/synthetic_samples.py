import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, glob

from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
from torchvision.transforms.functional import to_pil_image

torch.manual_seed(1)

# ============================================================
# 1. CONFIGURATION
# ============================================================
os.environ["TMPDIR"] = "/tmp" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TARGET_WNID = "n02113978" # Dog
# TARGET_CLASS_INDEX = 268  # Dog
# TARGET_WNID = "n04285008" # Cars
# TARGET_CLASS_INDEX = 817 # Cars
# TARGET_WNID = "n01882714" # Koala
# TARGET_CLASS_INDEX = 105
TARGET_WNID = "n03457902" # greenhouse
TARGET_CLASS_INDEX = 580 # greenhouse

# NOTE: Please change the directory!
DATA_DIR = f"/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/{TARGET_WNID}"

STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
LCM_EPOCHS = 50
DREAM_BATCH_SIZE = 50

print(f"Running Dreamer on: {DEVICE}")

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



class BNHook:
    def __init__(self, module):
        self.module = module
        self.mean = None
        self.var = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inp, out):
        tmp = inp[0].clone() 
        self.mean = tmp.mean(dim=(0, 2, 3))
        self.var = tmp.var(dim=(0, 2, 3), unbiased=False)

bn_hooks = []
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        bn_hooks.append(BNHook(module))

# ============================================================
# 3. LCM MODULE & HELPER FUNCTIONS & INIT
# ============================================================

class LCM(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.u_raw = nn.Parameter(torch.zeros(dim))
        self.w = nn.Parameter(torch.ones(dim) * 0.1)
        self.a = nn.Parameter(torch.linspace(-1, 1, dim))

        # Will be calculated directly from features
        self.mu = None

    def laplace_kernel(self):
        return torch.exp(-torch.abs(self.a[:, None] - self.a[None, :]))
    
    def nll(self, x):
        a_sorted, idx = torch.sort(self.a)
        x_s = x[:, idx]
       
        mu_s = self.mu[:, idx]
        
        w_s = torch.clamp(torch.abs(self.w[idx]), min=0.05)
        z = (x_s - mu_s) / w_s
        
        dt = torch.abs(a_sorted[1:] - a_sorted[:-1]).unsqueeze(0) + 1e-6
        phi = torch.exp(-dt)
        
        trans_var = 1.0 - phi**2
        trans_var = torch.clamp(trans_var, min=0.05)
        
        resid = z[:, 1:] - phi * z[:, :-1]
        
        std_dev = torch.sqrt(trans_var)
        loss_trans = torch.log(std_dev) + (torch.abs(resid) / std_dev)
        loss_0 = torch.abs(z[:, :1])
        
        total_nll = (loss_trans.sum(dim=1) + loss_0.sum(dim=1)).mean()
        return total_nll / x.size(1)

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

lcms = {l: LCM(layer_dims[l]).to(DEVICE) for l in LAYERS}

print("LCM's statistics initialization...")
transform_target = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(*STATS)
])
dataset = [transform_target(Image.open(p).convert('RGB')) for p in glob.glob(os.path.join(DATA_DIR, "*.JPEG"))]
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

target_means_lcm = {} 
target_vars_lcm = {}

with torch.no_grad():
    all_f = {l: [] for l in LAYERS}
    for x in loader:
        model(x.to(DEVICE))
        for l in LAYERS: 
            all_f[l].append(hooks_lcm[l].features.view(x.size(0), -1).cpu())
    
    for l in LAYERS:
        f_cat = torch.cat(all_f[l], dim=0)
        target_means_lcm[l] = f_cat.mean(0).to(DEVICE)
        target_vars_lcm[l] = f_cat.var(0).to(DEVICE)
        lcms[l].mu = target_means_lcm[l].unsqueeze(0)

# ============================================================
# 4. TRAIN OR LOAD LCM
# ============================================================
LCM_PATH = f"./lcm_resnet34_{TARGET_WNID}.pth"
if os.path.exists(LCM_PATH):
    print(f"\n--- LCM parameters loaded from {LCM_PATH} ---")
    ckpt = torch.load(LCM_PATH, map_location=DEVICE)
    for l in LAYERS: 
        lcms[l].load_state_dict(ckpt[l])
        lcms[l].eval()
else:
    print(f"\n--- LCM training (epochs: {LCM_EPOCHS}) ---")
    opts = {l: optim.Adam([
        {'params': lcms[l].u_raw, 'lr': 1e-2},
        {'params': lcms[l].w, 'lr': 1e-2},
        {'params': lcms[l].a, 'lr': 0.12, 'weight_decay': 1e-5},
    ], betas=(0.9, 0.999)) for l in LAYERS}

    for epoch in range(LCM_EPOCHS):
        total_loss = {l: 0.0 for l in LAYERS}
        batches = 0
        for x in loader:
            x = x.to(DEVICE)
            with torch.no_grad(): 
                model(x)
            
            for l in LAYERS:
                f = hooks_lcm[l].features.view(x.size(0), -1)
                
                V_feat = f - lcms[l].mu 
                
                opts[l].zero_grad()
                u = F.softplus(lcms[l].u_raw) + 1e-5
                
                frob = frobenius_dist_reduced(u, lcms[l].a, lcms[l].w, V_feat)
                
                D = layer_dims[l]
                loss = frob / (D ** 2) 
                
                loss.backward()
                opts[l].step()
                total_loss[l] += loss.item()
            batches += 1

        msg = f"LCM Epoch {epoch+1:4d}/{LCM_EPOCHS}"
        for l in LAYERS:
            msg += f" | {l}: {total_loss[l] / batches:.6f}"
        print(msg)
        
    torch.save({l: lcms[l].state_dict() for l in LAYERS}, LCM_PATH)

# ============================================================
# 5. DREAMING (Main Pyramid)
# ============================================================
print(f"\n--- Starting Dream ({DREAM_BATCH_SIZE} images) ---")
current_img = None

W_CE = 1.0
W_BN = 1.0
# NOTE: Change to 0 to not use REMIX
W_LCM = 5.0
W_TV = 0.1
W_DIV = 0.0

configs = [
    {'size': 112, 'iters': 1000, 'lr': 0.05},
    {'size': 224, 'iters': 2000, 'lr': 0.02},
]

target_label = torch.full((DREAM_BATCH_SIZE,), TARGET_CLASS_INDEX, dtype=torch.long, device=DEVICE)

MEAN = torch.tensor(STATS[0], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(STATS[1], device=DEVICE).view(1, 3, 1, 1)

for cfg in configs:
    size = cfg['size']
    print(f">>> Scale {size}x{size}...")
    
    if current_img is None:
        param = torch.rand(DREAM_BATCH_SIZE, 3, size, size, device=DEVICE) 
    else:
        param = F.interpolate(current_img, size=(size, size), mode='bilinear', align_corners=False)
    
    param = param.detach().requires_grad_(True)
    optimizer = optim.Adam([param], lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['iters'])
    
    aug = nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    )

    for step in range(cfg['iters']):
        optimizer.zero_grad()
        
        if step < (cfg['iters'] // 2) and step % 15 == 0:
            param.data = F.avg_pool2d(param.data, kernel_size=3, stride=1, padding=1)

        sx, sy = np.random.randint(-3, 4, 2)
        img_roll = torch.roll(param, shifts=(sx, sy), dims=(2, 3))
        
        img_aug = aug(img_roll)
        img_input = F.interpolate(img_aug, size=(224, 224), mode='bilinear', align_corners=False)
        
        img_norm = (img_input - MEAN) / STD
        logits = model(img_norm)
        
        loss_ce = F.cross_entropy(logits, target_label)
        
        loss_bn = sum((h.mean - h.module.running_mean).pow(2).mean() + 
                      (h.var - h.module.running_var).pow(2).mean() for h in bn_hooks)
            
        loss_lcm = 0.0
        for l in LAYERS:
            f = hooks_lcm[l].features.view(DREAM_BATCH_SIZE, -1)
            
            loss_lcm += (f.mean(0) - target_means_lcm[l]).pow(2).mean()
            loss_lcm += (f.var(0) - target_vars_lcm[l]).pow(2).mean()
            loss_lcm += (lcms[l].nll(f) / layer_dims[l])
            
        loss_div = diversity_loss(hooks_lcm['layer4'].features)
            
        tv = torch.abs(img_aug[:,:,:,1:] - img_aug[:,:,:,:-1]).mean() + \
             torch.abs(img_aug[:,:,1:,:] - img_aug[:,:,:-1,:]).mean()

        total_loss = W_CE * loss_ce + W_BN * loss_bn + W_LCM * loss_lcm + W_TV * tv + W_DIV * loss_div

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad(): 
            param.clamp_(0.0, 1.0)

        if step % 200 == 0:
            print(f" Step {step:4d} | CE: {loss_ce.item():.2f} | BN: {loss_bn.item():.4f} | NLL: {loss_lcm.item():.2f}")

    current_img = param

# ============================================================
# 5 FINAL ENHANCEMENTS
# ============================================================
print("\n--- Final Polish (Boosting CE, no jitter/blur) ---")

W_CE_FINAL = W_CE * 20.0 
W_TV_FINAL = W_TV * 0.5 

optimizer_polish = optim.Adam([current_img], lr=0.005) 

for step in range(400):
    optimizer_polish.zero_grad()
    
    img_input = F.interpolate(current_img, size=(224, 224), mode='bilinear', align_corners=False)
    
    img_norm = (img_input - MEAN) / STD
    logits = model(img_norm)
    
    loss_ce = F.cross_entropy(logits, target_label)
    
    loss_bn = sum((h.mean - h.module.running_mean).pow(2).mean() + 
                  (h.var - h.module.running_var).pow(2).mean() for h in bn_hooks)
        
    loss_lcm = 0.0
    for l in LAYERS:
        f = hooks_lcm[l].features.view(DREAM_BATCH_SIZE, -1)
        loss_lcm += (f.mean(0) - target_means_lcm[l]).pow(2).mean()
        loss_lcm += (f.var(0) - target_vars_lcm[l]).pow(2).mean()
        loss_lcm += (lcms[l].nll(f) / layer_dims[l])
        
    tv = torch.abs(current_img[:,:,:,1:] - current_img[:,:,:,:-1]).mean() + \
         torch.abs(current_img[:,:,1:,:] - current_img[:,:,:-1,:]).mean()

    total_loss = W_CE_FINAL * loss_ce + W_BN * loss_bn + W_LCM * loss_lcm + W_TV_FINAL * tv

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_([current_img], max_norm=1.0)
    optimizer_polish.step()
    
    with torch.no_grad(): 
        current_img.clamp_(0.0, 1.0)

    if step % 100 == 0:
        conf = F.softmax(logits, dim=1)[:, TARGET_CLASS_INDEX].mean().item() * 100
        print(f" Polish Step {step:4d} | CE: {loss_ce.item():.2f} | Conf: {conf:.1f}%")

# ============================================================
# 6. EXPORT
# ============================================================

with torch.no_grad():
    out = current_img.cpu()

output_folder = "deep_inversion_greenhouse"

os.makedirs(output_folder, exist_ok=True)

for i in range(out.size(0)):
    pil_img = to_pil_image(out[i])
    filename = os.path.join(output_folder, f"dream_greenhouse_{i+1}.pdf")    
    pil_img.save(filename, "PDF", resolution=100.0)
    
print(f"Done! Saved {out.size(0)} files into the '{output_folder}' directory.")