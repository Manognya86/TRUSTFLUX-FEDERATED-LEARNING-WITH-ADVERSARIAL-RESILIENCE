# data.py
import os
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import numpy as np
from config import (
    CHEXPERT_ROOT, ISIC_ROOT, IMG_SIZE, BATCH_SIZE,
    NUM_CLIENTS, USE_SYNTHETIC, DATASET, NUM_CLASSES
)
from synthetic import get_synthetic_client_loader, get_synthetic_test_loader

# ------------------------------------------------------------------
# ISIC 2019 dataset loader (requires download and structure)
# Expected structure:
#   isic/
#     train/
#       class0/
#       class1/
#       ...
#     valid/
#       ...
# ------------------------------------------------------------------
def get_isic_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_isic_partition(client_id):
    """Return train loader for one client (non‑IID)."""
    transform = get_isic_transform()
    full_dataset = datasets.ImageFolder(
        root=os.path.join(ISIC_ROOT, 'train'),
        transform=transform
    )
    targets = np.array(full_dataset.targets)
    # Non‑IID: each client gets 80% of one class, 20% of others
    main_class = client_id % len(full_dataset.classes)
    mask = (targets == main_class) & (np.random.RandomState(client_id).rand(len(targets)) < 0.8)
    mask |= (targets != main_class) & (np.random.RandomState(client_id).rand(len(targets)) < 0.2)
    client_indices = np.where(mask)[0]
    client_dataset = Subset(full_dataset, client_indices)
    return DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

def get_isic_test_loader():
    transform = get_isic_transform()
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(ISIC_ROOT, 'valid'),
        transform=transform
    )
    return DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------
# CheXpert loader (unchanged)
# ------------------------------------------------------------------
def get_chexpert_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_chexpert_partition(client_id):
    transform = get_chexpert_transform()
    full_dataset = datasets.ImageFolder(
        root=os.path.join(CHEXPERT_ROOT, 'train'),
        transform=transform
    )
    targets = np.array(full_dataset.targets)
    main_class = client_id % len(full_dataset.classes)
    mask = (targets == main_class) & (np.random.RandomState(client_id).rand(len(targets)) < 0.8)
    mask |= (targets != main_class) & (np.random.RandomState(client_id).rand(len(targets)) < 0.2)
    client_indices = np.where(mask)[0]
    client_dataset = Subset(full_dataset, client_indices)
    return DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

def get_chexpert_test_loader():
    transform = get_chexpert_transform()
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(CHEXPERT_ROOT, 'valid'),
        transform=transform
    )
    return DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------
# Main loader factory
# ------------------------------------------------------------------
def load_client_data(client_id):
    """Return train loader for a specific client based on DATASET."""
    if USE_SYNTHETIC or DATASET == "synthetic":
        print(f"Client {client_id}: using synthetic data")
        return get_synthetic_client_loader(client_id)
    elif DATASET == "chexpert":
        print(f"Client {client_id}: using CheXpert data")
        return load_chexpert_partition(client_id)
    elif DATASET == "isic":
        print(f"Client {client_id}: using ISIC 2019 data")
        return load_isic_partition(client_id)
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

def get_test_loader():
    """Return test loader based on DATASET."""
    if USE_SYNTHETIC or DATASET == "synthetic":
        return get_synthetic_test_loader()
    elif DATASET == "chexpert":
        return get_chexpert_test_loader()
    elif DATASET == "isic":
        return get_isic_test_loader()
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")