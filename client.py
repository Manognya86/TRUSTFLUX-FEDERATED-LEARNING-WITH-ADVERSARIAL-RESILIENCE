# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from models import CheXpertCNN
from data import load_client_data, get_test_loader  # use local test loader for validation
from utils import set_parameters, evaluate_model
from attacks import poison_update
from config import EPOCHS_PER_CLIENT, ATTACK, MALICIOUS_CLIENT_ID, DATASET, NUM_CLASSES
from config import EPOCHS_PER_CLIENT, ATTACK, ATTACK_PARAMS, MALICIOUS_CLIENT_ID, DATASET, NUM_CLASSES

class CheXpertClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        # Model now adapts to NUM_CLASSES via config
        self.model = CheXpertCNN(num_classes=NUM_CLASSES)
        self.train_loader = load_client_data(client_id)
        self.val_loader = get_test_loader()  # same test set used for validation
        self.criterion = nn.CrossEntropyLoss() if DATASET == "isic" else nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Apply label flipping if attack is active
        def maybe_flip_labels(labels):
            if ATTACK == 'label_flip' and self.client_id == MALICIOUS_CLIENT_ID:
                # Flip to target class
                target = ATTACK_PARAMS['target_class']
                if DATASET == "isic":
                    # For classification, set all labels to target
                    return torch.full_like(labels, target)
                else:
                    # For multi‑label, set target class to 1, others 0? Simpler: random flip
                    # We'll just invert all bits for demo
                    return 1 - labels
            return labels

        for _ in range(EPOCHS_PER_CLIENT):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                # Apply label flipping if needed
                flipped_labels = maybe_flip_labels(labels)
                outputs = self.model(images)
                loss = self.criterion(outputs, flipped_labels.float() if DATASET != "isic" else flipped_labels)
                loss.backward()
                self.optimizer.step()

        update = self.get_parameters({})
        # Apply other attacks (amplify, reverse, noise, model_substitution)
        update = poison_update(update, self.client_id)

        # Compute validation accuracy (for trust scoring)
        val_acc = evaluate_model(self.model, self.val_loader)

        return update, len(self.train_loader.dataset), {"cid": self.client_id, "val_acc": val_acc}

def start_client(cid):
    fl.client.start_numpy_client(
        server_address="server:8080",
        client=CheXpertClient(cid)
    )

if __name__ == "__main__":
    import sys
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(cid)