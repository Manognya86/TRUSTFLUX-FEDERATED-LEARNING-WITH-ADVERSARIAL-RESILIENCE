# utils.py
import torch
import numpy as np
import hashlib
from collections import OrderedDict

def average_models(updates, weights=None):
    if not updates:
        return None
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    avg_update = [torch.zeros_like(p) for p in updates[0]]
    for u, w in zip(updates, weights):
        for i, p in enumerate(u):
            avg_update[i] += w * p
    return avg_update

def flatten_model(model):
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().numpy().flatten())
    return np.concatenate(params)

def model_hash(model):
    flat = flatten_model(model)
    return hashlib.sha256(flat.tobytes()).hexdigest()

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict()
    for k, v in params_dict:
        if isinstance(v, np.ndarray):
            state_dict[k] = torch.from_numpy(v)
        else:
            state_dict[k] = v.clone().detach()
    model.load_state_dict(state_dict, strict=True)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            # For multi-label (CheXpert) or multi-class (ISIC)
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # multi-label
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.numel()
                correct += (predicted == labels).sum().item()
            else:
                # multi-class
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    return correct / total