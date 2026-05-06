import torch
import torch.optim as optim
import torch.nn.functional as F
import os, glob
import math
import random

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from utils import *
from lcm import LCM

torch.manual_seed(1)

os.environ["TMPDIR"] = "/tmp" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WNIDS = [
    "n02113978", # Dog
    "n04285008", # Cars
    "n01882714", # Koala
    "n03457902"  # Greenhouse
]

# NOTE: Update to your dataset path
BASE_DATA_DIR = "/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/"
STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

LCM_EPOCHS = 200 

print(f"Running LCM Fitting and Log-Likelihood Analysis for ViT on 4 Classes on: {DEVICE}")

# ============================================================
# 2. MODEL & HOOKS
# ============================================================
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE)
model.eval()

for module in model.modules():
    if hasattr(module, 'inplace'):
        module.inplace = False

for p in model.parameters(): 
    p.requires_grad = False

class FeatureHook:
    def __init__(self, module):
        self.features = None
        module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inp, out): 
        # ViT output is (B, 197, 768). Drop CLS token to keep spatial features.
        # Use mean pooling to yield (B, 768) to match LCM expectations
        self.features = out[:, 1:, :].mean(dim=1).clone()

# Hook into all 12 Transformer encoder blocks
target_blocks = [_ for _ in range(12)]
hooks_lcm = {f"block{i}": FeatureHook(model.encoder.layers[i]) for i in target_blocks}
LAYERS = list(hooks_lcm.keys())

with torch.no_grad(): 
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    model(dummy)

layer_dims = {l: hooks_lcm[l].features.size(-1) for l in LAYERS}
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
            all_f[l].append(hooks_lcm[l].features.cpu())
    
    for l in LAYERS:
        f_cat = torch.cat(all_f[l], dim=0)
        target_means_lcm[l] = f_cat.mean(0).to(DEVICE)
        target_vars_lcm[l] = f_cat.var(0, unbiased=False).to(DEVICE) 
        
        lcms[l].mu = target_means_lcm[l].unsqueeze(0)

hist_lcm_ll = {l: [] for l in LAYERS}
hist_base_ll = {l: [] for l in LAYERS}

print(f"\n--- LCM training (epochs: {LCM_EPOCHS}) ---")

def get_layer_lr_multiplier(layer_name):
    block_idx = int(layer_name.replace('block', '')) 
    return 1.0 + (24.0 * ((11 - block_idx) / 11.0))

opts = {}
for l in LAYERS:
    mult = get_layer_lr_multiplier(l)
    opts[l] = optim.Adam([
        {'params': lcms[l].u_raw, 'lr': 1e-2 * mult},
        {'params': lcms[l].w, 'lr': 1e-2 * mult},
        {'params': lcms[l].a, 'lr': 0.12 * mult},
    ], betas=(0.9, 0.999))

extracted_features_gpu = {l: torch.cat(all_f[l], dim=0).to(DEVICE) for l in LAYERS}
BATCH_SIZE_FIT = 128

num_samples = extracted_features_gpu['block0'].size(0)
batches_per_epoch = math.ceil(num_samples / BATCH_SIZE_FIT)
TOTAL_STEPS = LCM_EPOCHS * batches_per_epoch

def lr_lambda(step):
    t = step / TOTAL_STEPS
    
    if t < 0.4:
        return 1.0
    else:
        decay_progress = (t - 0.4) / 0.6 
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

schedulers = {l: optim.lr_scheduler.LambdaLR(opts[l], lr_lambda) for l in LAYERS}

for epoch in range(LCM_EPOCHS):
    epoch_lcm_ll = {l: 0.0 for l in LAYERS}
    epoch_base_ll = {l: 0.0 for l in LAYERS}
    total_frob_loss = {l: 0.0 for l in LAYERS}
    batches = 0
    
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

    msg = f"Ep {epoch+1:3d}/{LCM_EPOCHS} | "
    for l in LAYERS:
        avg_lcm = epoch_lcm_ll[l] / batches
        avg_base = epoch_base_ll[l] / batches
        hist_lcm_ll[l].append(avg_lcm)
        hist_base_ll[l].append(avg_base)
    
    msg += f"B0: LCM {hist_lcm_ll['block0'][-1]:,.1f} | B11: LCM {hist_lcm_ll['block11'][-1]:,.1f}"
    msg += f" | B0: Base {hist_base_ll['block0'][-1]:,.1f} | B11: Base {hist_base_ll['block11'][-1]:,.1f}"
    print(msg)

print("\n--- Exporting Log-Likelihood values to TXT ---")
txt_path = "./log_likelihood_values_vit_4_classes.txt"

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