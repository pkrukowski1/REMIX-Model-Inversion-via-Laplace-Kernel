import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, glob
import math
import matplotlib.pyplot as plt
import random
import time

from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
from torch.profiler import profile, ProfilerActivity

from lcm import LCM
from utils import *

torch.manual_seed(1)

os.environ["TMPDIR"] = "/tmp" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All 4 target classes 
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

lcms = {l: LCM(layer_dims[l]).to(DEVICE) for l in LAYERS}

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

print("\n--- Calculating Memory Statistics ---")
mem_txt_path = "./memory_flops_report.txt"

with open(mem_txt_path, "w") as f:
    f.write("=== MEMORY & THEORETICAL COST REPORT ===\n\n")
    total_lcm_bytes = 0
    total_dense_bytes = 0

    for l in LAYERS:
        D = layer_dims[l]
        # LCM parameters: u_raw, w, a, mu (each is size D, float32 = 4 bytes)
        lcm_bytes = 4 * D * 4  
        total_lcm_bytes += lcm_bytes
        
        # Dense Covariance: DxD matrix + D mean vector (float32 = 4 bytes)
        dense_bytes = ((D * D) + D) * 4 
        total_dense_bytes += dense_bytes
        
        ratio = dense_bytes / lcm_bytes if lcm_bytes > 0 else 0
        f.write(f"{l.upper()} (D={D:,}):\n")
        f.write(f"  LCM Memory:   {lcm_bytes / 1024:.2f} KB\n")
        
        if dense_bytes > 1024**3:
            f.write(f"  Dense Memory: {dense_bytes / (1024**3):.2f} GB\n")
        else:
            f.write(f"  Dense Memory: {dense_bytes / (1024**2):.2f} MB\n")
            
        f.write(f"  Savings:      {ratio:,.1f}x smaller\n\n")

    overall_ratio = total_dense_bytes / total_lcm_bytes if total_lcm_bytes > 0 else 0
    f.write(f"TOTAL LCM MEMORY:   {total_lcm_bytes / 1024:.2f} KB\n")
    
    if total_dense_bytes > 1024**3:
        f.write(f"TOTAL DENSE MEMORY: {total_dense_bytes / (1024**3):.2f} GB\n")
    else:
        f.write(f"TOTAL DENSE MEMORY: {total_dense_bytes / (1024**2):.2f} MB\n")
        
    f.write(f"OVERALL SAVINGS:    {overall_ratio:,.1f}x smaller\n")
    f.write("-" * 50 + "\n")

print(f"Memory statistics saved to {mem_txt_path}")

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

flops_measured = False
epoch_times = []
completed_epochs = 0

try:
    for epoch in range(LCM_EPOCHS):
        epoch_start_time = time.time()
        
        epoch_lcm_ll = {l: 0.0 for l in LAYERS}
        epoch_base_ll = {l: 0.0 for l in LAYERS}
        total_frob_loss = {l: 0.0 for l in LAYERS}
        batches = 0
        
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, BATCH_SIZE_FIT):
            batch_idx = indices[start_idx:start_idx + BATCH_SIZE_FIT]
            
            if not flops_measured:
                print("Profiling FLOPs for the first batch...")
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
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
                        
                        with torch.no_grad():
                            l_ll = lcms[l].log_likelihood(f)
                            b_ll = baseline_diagonal_ll(f, target_means_lcm[l], target_vars_lcm[l])
                            epoch_lcm_ll[l] += l_ll.item()
                            epoch_base_ll[l] += b_ll.item()
                
                batch_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops > 0)
                epoch_flops = batch_flops * batches_per_epoch
                
                with open(mem_txt_path, "a") as f_mem:
                    f_mem.write("\n=== FLOPs COMPUTATION ===\n")
                    f_mem.write(f"Measured FLOPs per batch: {batch_flops:,}\n")
                    f_mem.write(f"Estimated FLOPs per epoch ({batches_per_epoch} batches): {epoch_flops:,}\n")
                    f_mem.write(f"Total FLOPs for {LCM_EPOCHS} epochs: {epoch_flops * LCM_EPOCHS:,}\n")
                
                print(f"FLOP profiling complete! Estimated {epoch_flops:,} FLOPs per epoch. Saved to {mem_txt_path}.")
                flops_measured = True
                batches += 1
                continue

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
            
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        completed_epochs += 1

        msg = f"Ep {epoch+1:2d}/{LCM_EPOCHS} [{epoch_duration:.2f}s]"
        for l in LAYERS:
            avg_lcm = epoch_lcm_ll[l] / batches
            avg_base = epoch_base_ll[l] / batches
            hist_lcm_ll[l].append(avg_lcm)
            hist_base_ll[l].append(avg_base)
            
            msg += f" | {l.upper()}: LCM {avg_lcm/1000:,.0f}k (Base {avg_base/1000:,.0f}k)"
        print(msg)
        
        if completed_epochs % 10 == 0:
            torch.save({l: lcms[l].state_dict() for l in LAYERS}, LCM_PATH)
            print(f"   -> [Auto-Save] Model weights checkpointed to {LCM_PATH}")

except KeyboardInterrupt:
    print("\n" + "="*60)
    print(f"[!] Training interrupted by user at Epoch {completed_epochs + 1}!")
    print("[!] Breaking the loop safely. Generating plots and saving current data...")
    print("="*60 + "\n")
    torch.save({l: lcms[l].state_dict() for l in LAYERS}, LCM_PATH)

if len(epoch_times) > 0:
    avg_epoch_time = np.mean(epoch_times[1:]) if len(epoch_times) > 1 else epoch_times[0]
    with open(mem_txt_path, "a") as f_mem:
        f_mem.write("\n=== TIMING REPORT ===\n")
        f_mem.write(f"Total training time: {sum(epoch_times):.2f} seconds\n")
        f_mem.write(f"Average time per epoch (excluding epoch 1 profiler overhead): {avg_epoch_time:.2f} seconds\n")
        f_mem.write("-" * 50 + "\n")
    
print("\n--- Generating Log-Likelihood Plots ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, l in enumerate(LAYERS):
    ax = axes[i]
    epochs_x = np.arange(1, completed_epochs + 1)
    
    ax.plot(epochs_x, hist_base_ll[l][:completed_epochs], linestyle='--', linewidth=2.5, color='#7f8c8d', label='Diagonal Covariance')
    ax.plot(epochs_x, hist_lcm_ll[l][:completed_epochs], linestyle='-', linewidth=2.5, color='#e67e22', label='Full Covariance')
    
    total_dims = layer_dims[l]
    ax.set_title(f'ResNet34 - {l.capitalize()} ($D = {total_dims:,}$)', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Log-Likelihood', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    if i == 0:
        ax.legend(fontsize=12, loc='lower right')

plot_path = "./log_likelihood_comparison_4_classes.pdf"
plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Plots saved to {plot_path}")

print("\n--- Exporting Log-Likelihood values to TXT ---")
txt_path = "./log_likelihood_values_4_classes.txt"

with open(txt_path, "w") as f:
    header = ["Epoch"]
    for l in LAYERS:
        header.extend([f"{l}_Baseline", f"{l}_LCM"])
    f.write("\t".join(header) + "\n")
    
    for epoch in range(completed_epochs):
        row = [str(epoch + 1)]
        for l in LAYERS:
            row.append(f"{hist_base_ll[l][epoch]:.2f}")
            row.append(f"{hist_lcm_ll[l][epoch]:.2f}")
        f.write("\t".join(row) + "\n")

print(f"Log-Likelihood values securely saved to {txt_path}")