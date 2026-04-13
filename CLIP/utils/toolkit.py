import os
import numpy as np

def split_images_labels(imgs):
    images = []
    labels = []
    for img_path, label in imgs:
        images.append(img_path)
        labels.append(label)
        
    return np.array(images), np.array(labels)


def get_dataset_class_names(base_dir, dataset_name):
    train_dir = os.path.join(base_dir, dataset_name, 'train')
    target_dir = train_dir if os.path.exists(train_dir) else os.path.join(base_dir, dataset_name)
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Could not find directory to extract classes: {target_dir}")
        
    class_names = [d.name for d in os.scandir(target_dir) if d.is_dir()]
    class_names.sort() 
    
    return class_names