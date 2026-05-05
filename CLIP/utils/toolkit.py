import os
import numpy as np
import torch
import torch.optim as optim
import math
from datetime import datetime
from collections import OrderedDict
import copy


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    if len(y_pred) == 0 or len(y_true) == 0:
        return {"total": 0.0, "old": 0.0, "new": 0.0, "by_task": {}}
        
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Initialize the specific key the BaseLearner is looking for
    all_acc["by_task"] = {}

    # Grouped accuracy, for initial classes (Task 0)
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    acc_val = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    ) if len(idxes) > 0 else 0.0
    
    all_acc[label] = acc_val
    all_acc["by_task"]["Task_0"] = acc_val

    # For incremental classes (Task 1, 2, 3...)
    task_id = 1
    for class_id in range(init_cls, np.max(y_true) + 1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        acc_val = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        ) if len(idxes) > 0 else 0.0
        
        all_acc[label] = acc_val
        all_acc["by_task"][f"Task_{task_id}"] = acc_val
        task_id += 1

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (
        0 
        if len(idxes) == 0 
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def state_dict_to_vector(state_dict, remove_keys=[]) -> torch.Tensor:
    shared_state_dict = copy.deepcopy(state_dict)
    shared_state_dict_keys = list(shared_state_dict.keys())
    for key in remove_keys:
        for _key in shared_state_dict_keys:
            if key in _key:
                del shared_state_dict[_key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Load vector into state_dict, except the keys in `remove_keys`.
    """
    removed_keys = []
    reference_dict = copy.deepcopy(state_dict)
    reference_dict_keys = list(reference_dict.keys())
    for key in remove_keys:
        for _key in reference_dict_keys:
            if key in _key:
                removed_keys.append(_key)
                del reference_dict[_key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    return sorted_reference_dict


def get_dataset_class_names(base_dir, dataset_name):
    train_dir = os.path.join(base_dir, dataset_name, 'train')
    target_dir = train_dir if os.path.exists(train_dir) else os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Could not find directory to extract classes: {target_dir}")
        
    class_names = [d.name for d in os.scandir(target_dir) if d.is_dir()]
    class_names.sort() 
    
    return class_names



def build_optimizer(network, args):
    """
    Constructs the PyTorch optimizer dynamically based on your args namespace.
    Defaults to SGD (standard for most Continual Learning/Vision tasks) if not specified.
    """
    # Safely extract parameters, providing sensible defaults if they aren't in args
    optim_type = getattr(args, 'optimizer', 'sgd').lower()
    lr = getattr(args, 'init_lr', 0.1)  # Assuming args.init_lr exists based on your snippet
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    momentum = getattr(args, 'momentum', 0.9)

    if optim_type == 'sgd':
        optimizer = optim.SGD(
            network.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
    elif optim_type == 'adam':
        optimizer = optim.Adam(
            network.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(
            network.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
        
    return optimizer


def cosine_lr(optimizer, init_lr, warmup_iters, total_iters):
    """
    Returns a function that manually updates the optimizer's learning rate 
    using a cosine schedule, bypassing PyTorch's class system entirely.
    """
    
    def step(current_iter):
        # Handle optional linear warmup
        if current_iter < warmup_iters:
            new_lr = init_lr * (current_iter / max(1, warmup_iters))
        else:
            # Your exact cosine math applied after any warmup
            adjusted_iter = current_iter - warmup_iters
            adjusted_total = total_iters - warmup_iters
            
            denominator = 200 * (adjusted_total - 1)
            
            # Prevent division by zero if total_iters is 1
            if denominator <= 0:
                decay = 1.0
            else:
                decay = math.cos((99 * math.pi * adjusted_iter) / denominator)
                
            new_lr = init_lr * decay
            
        # Inject the newly calculated LR directly into the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
        # Returning the LR is handy if you want to log it to wandb/tensorboard
        return new_lr 
        
    return step

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