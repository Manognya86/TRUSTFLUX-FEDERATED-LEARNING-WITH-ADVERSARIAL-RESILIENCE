# defense.py
import torch
import numpy as np
from config import MULTI_KRUM_K, TRUST_ALPHA

def multi_krum(updates, n_selected, k=None):
    if k is None:
        k = MULTI_KRUM_K
    num_clients = len(updates)
    if num_clients <= n_selected:
        return list(range(num_clients))

    distances = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i+1, num_clients):
            dist = sum((p-q).norm().item() for p,q in zip(updates[i], updates[j]))
            distances[i, j] = dist
            distances[j, i] = dist

    scores = []
    for i in range(num_clients):
        sorted_dists = np.sort(distances[i])[1:k+1]
        scores.append(np.sum(sorted_dists))

    selected = np.argsort(scores)[:n_selected]
    return selected

def update_trust_scores(trust_scores, client_ids, selected_ids, alpha=None):
    if alpha is None:
        alpha = TRUST_ALPHA
    for cid in client_ids:
        if cid not in trust_scores:
            trust_scores[cid] = 1.0
        if cid in selected_ids:
            trust_scores[cid] = trust_scores[cid] * alpha + (1 - alpha)
        else:
            trust_scores[cid] = trust_scores[cid] * alpha
    return trust_scores

def compute_gradient_similarity(updates):
    """Return list of similarity scores (cosine or Euclidean inverse)."""
    if len(updates) < 2:
        return [1.0] * len(updates)
    # Use median as reference
    median_update = [torch.median(torch.stack([u[i] for u in updates]), dim=0)[0] for i in range(len(updates[0]))]
    similarities = []
    for upd in updates:
        sim = 1.0 - sum((p - q).norm().item() for p,q in zip(upd, median_update)) / (1e-6 + sum(p.norm().item() for p in median_update))
        similarities.append(max(0.0, sim))
    return similarities