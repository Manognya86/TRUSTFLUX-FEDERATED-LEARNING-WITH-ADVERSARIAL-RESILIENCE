# attacks.py
import torch
import copy
from config import ATTACK, ATTACK_PARAMS, MALICIOUS_CLIENT_ID

def poison_update(original_update, client_id):
    """Modify client update according to global ATTACK setting."""
    # Treat None or empty string as no attack
    if ATTACK in (None, "") or client_id != MALICIOUS_CLIENT_ID:
        return original_update

    poisoned = copy.deepcopy(original_update)

    if ATTACK == 'amplify':
        scale = ATTACK_PARAMS['scale']
        for p in poisoned:
            p *= scale

    elif ATTACK == 'reverse':
        for p in poisoned:
            p *= -1.0

    elif ATTACK == 'noise':
        noise_std = ATTACK_PARAMS['noise_std']
        for p in poisoned:
            p += torch.randn_like(p) * noise_std

    elif ATTACK == 'label_flip':
        # Label flipping is implemented in the client's training loop
        # We'll just pass through here; the client will handle it.
        pass

    elif ATTACK == 'model_substitution':
        # Replace the update with a malicious pre‑computed model
        # For simplicity, we'll just set all parameters to zero (effectively a poison)
        for p in poisoned:
            p.zero_()

    else:
        raise ValueError(f"Unknown attack type: {ATTACK}")

    return poisoned