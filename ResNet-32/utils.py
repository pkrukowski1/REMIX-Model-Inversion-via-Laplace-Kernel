# -*-coding:utf8-*-

import torch
import random
import numpy as np
import torch.nn.functional as F
import os

from backbone import resnet_cifar
from backbone import resnets
from backbone import classifier
from backbone import resnet_deep_inversion


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tv = torch.__version__
    if tv[:3] == '1.7' or tv[:3] == '1.8':
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(d=True)
    elif tv[:4] == '1.10' or tv[:4] == '1.13':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        pass


def build_model(dataset, bias=False, nf=16):
    if dataset == 'seq_cifar100' or dataset == 'seq_tinyimagenet':
        model = resnet_cifar.resnet32(nf=nf)
        head = classifier.DynamicSimpleHead(
            num_features=model.num_features, bias=bias
        )
    elif dataset == 'seq_imagenet100':
        model = resnets.resnet18()
        head = classifier.DynamicSimpleHead(
            num_features=model.num_features, bias=bias
        )
    else:
        raise ValueError('Invalid dataset')
    return model, head


def build_resnet(model_type, num_class):
    if model_type == 'resnet34':
        model = resnet_deep_inversion.ResNet34(num_classes=num_class)
    elif model_type == 'resnet18':
        model = resnet_deep_inversion.ResNet18(num_classes=num_class)
    else:
        raise ValueError('Invalid model type')
    head = torch.nn.Identity()
    return model, head


def check_gpu_memory():
    tv = torch.__version__
    if tv[:4] == '1.13':
        mem_inf = torch.cuda.mem_get_info()
    elif tv[:4] == '1.10':
        mem_inf = torch.cuda.memory_allocated()
    else:
        mem_inf = None
    return mem_inf

def fit_gmrf_correlation(model_gmrf, R_target, epochs=2000, lr=0.01):
    model_gmrf.train()
    optimizer = torch.optim.Adam([
        model_gmrf.a,
        model_gmrf.w,
        model_gmrf.u_raw
    ], lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.mse_loss(model_gmrf.correlation(), R_target)
        loss.backward()
        optimizer.step()
        
    model_gmrf.eval()

def log_memory_comparison(local_path, gmrfs):
    if gmrfs is None:
        print("==> GMRFs not initialized. Skipping memory log.")
        return

    log_path = os.path.join(local_path, "memory_comparison.txt")
    
    total_gmrf_bytes = 0
    total_dense_bytes = 0
    
    modules = gmrfs.values() if isinstance(gmrfs, torch.nn.ModuleDict) else gmrfs

    for gmrf in modules:
        layer_gmrf_bytes = sum(p.numel() * p.element_size() for p in gmrf.parameters())
        total_gmrf_bytes += layer_gmrf_bytes
        
        C = gmrf.a.numel()
        element_size = gmrf.a.element_size() 
        total_dense_bytes += (C * C) * element_size

    gmrf_mb = total_gmrf_bytes / (1024 ** 2)
    dense_mb = total_dense_bytes / (1024 ** 2)
    ratio = total_dense_bytes / total_gmrf_bytes if total_gmrf_bytes > 0 else 0

    with open(log_path, "a") as f:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        f.write(f"\n[{timestamp}] Memory Comparison Report\n")
        f.write(f"GMRF total memory (actual): {gmrf_mb:.4f} MB\n")
        f.write(f"Dense Covariance total memory (theoretical): {dense_mb:.2f} MB\n")
        f.write(f"Memory reduction factor: {ratio:.1f}x\n")
        f.write("-" * 45 + "\n")

    print(f"==> Memory comparison logged. Savings: {ratio:.1f}x smaller than Dense Matrix.")