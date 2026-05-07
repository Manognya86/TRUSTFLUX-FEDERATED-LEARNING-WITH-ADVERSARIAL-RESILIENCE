import hashlib
import json
import numpy as np

class MerkleTree:
    def __init__(self):
        self.leaves = []
        self.root = None

    def add_leaf(self, data):
        leaf_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        self.leaves.append(leaf_hash)
        self._build_tree()

    def _build_tree(self):
        if not self.leaves:
            self.root = None
            return
        nodes = self.leaves[:]
        while len(nodes) > 1:
            temp = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i+1] if i+1 < len(nodes) else left
                combined = left + right
                temp.append(hashlib.sha256(combined.encode()).hexdigest())
            nodes = temp
        self.root = nodes[0]

audit_log = MerkleTree()

def log_round(round_num, global_model_hash, client_ids, selected_ids):
    entry = {
        'round': round_num,
        'global_model': global_model_hash,
        'participants': client_ids,
        'selected': selected_ids.tolist() if isinstance(selected_ids, np.ndarray) else selected_ids
    }
    audit_log.add_leaf(entry)