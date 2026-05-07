#!/usr/bin/env python3
# server.py – TRUSTFLUX CUSTOM SERVER – FINAL WITH ENHANCED TRUST
print("🚀 Starting TrustFluxServer script – loading modules...")
import sys
import time
sys.stdout.flush()

import flwr as fl
import numpy as np
import torch
import json
import os
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List
from flwr.common import Parameters, Scalar, FitRes
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History

from config import (
    NUM_CLIENTS, NUM_ROUNDS, FRACTION_FIT,
    AGGREGATION_METHOD, MULTI_KRUM_K, USE_TRUST_SCORING,
    AUDIT_ENABLED, METRICS_FILE, EXPERIMENT_NAME,
    TRUST_VALIDATION_WEIGHT, TRUST_ALPHA, DATASET, NUM_CLASSES   # <-- add these
)
print(f"✅ Config loaded. METRICS_FILE = {METRICS_FILE}")

from models import CheXpertCNN
from defense import multi_krum, update_trust_scores, compute_gradient_similarity
from audit import log_round, audit_log
from utils import average_models, model_hash, set_parameters, evaluate_model
from data import get_test_loader

print("✅ All imports successful.")

class TrustFluxServer(fl.server.Server):
    def __init__(self, strategy=None, client_manager=None):
        print("🚀 TrustFluxServer.__init__() called")
        if client_manager is None:
            client_manager = SimpleClientManager()
        super().__init__(client_manager=client_manager, strategy=strategy)

        self.global_model = CheXpertCNN(num_classes=NUM_CLASSES)
        self.trust_scores = {cid: 1.0 for cid in range(NUM_CLIENTS)}
        self.round = 0
        self.test_loader = get_test_loader()
        self.metrics_history = {
            'rounds': [],
            'test_accuracy': [],
            'trust_scores': [],
            'selected': [],
            'merkle_root': []
        }
        self.attacker_id = 0

        # ✅ Create metrics directory and write test file
        try:
            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            print(f"✅ Created directory: {os.path.dirname(METRICS_FILE)}")
            test_file = os.path.join(os.path.dirname(METRICS_FILE), "test.txt")
            with open(test_file, "w") as f:
                f.write(f"Server started at {np.datetime64('now')}")
                f.flush()
                os.fsync(f.fileno())
            print(f"✅ Test file written: {test_file}")
        except Exception as e:
            print(f"❌ Failed to write test file: {e}")
            import traceback
            traceback.print_exc()

        self._write_live_metrics()
        print(f"✅ Initial metrics file written to {METRICS_FILE}")
        sys.stdout.flush()

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], Dict[str, Scalar]]]:
        print(f"🔄 fit_round called for round {server_round}")
        self.round = server_round

        # ⏳ Wait until all clients are registered
        while True:
            all_clients = self._client_manager.all()
            available = list(all_clients.keys())
            print(f"📊 Available clients: {available}")
            if len(available) >= NUM_CLIENTS:
                print(f"✅ All {NUM_CLIENTS} clients are now registered")
                break
            print(f"⚠️ Waiting for {NUM_CLIENTS - len(available)} more clients...")
            time.sleep(5)

        # Get client instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            print("⚠️ No clients selected for fit (configure_fit returned empty)")
            return None

        print(f"📋 Selected {len(client_instructions)} clients for fit")

        # 🔁 Sequential client training
        results: List[Tuple[ClientProxy, FitRes]] = []
        failures: List[Tuple[ClientProxy, Exception]] = []
        for client_proxy, fit_ins in client_instructions:
            try:
                fit_res = client_proxy.fit(fit_ins, timeout=timeout)
                results.append((client_proxy, fit_res))
                print(f"✅ Client {client_proxy.cid} finished training")
            except Exception as e:
                failures.append((client_proxy, e))
                print(f"⚠️ Client {client_proxy.cid} failed: {e}")

        if not results:
            return None

        # ✅ Our custom aggregation
        aggregated_params, metrics = self.aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            self.parameters = aggregated_params
            self.strategy.parameters = aggregated_params

        return aggregated_params, metrics, {}

    def aggregate_fit(self, rnd, results, failures):
        print(f"🔄 aggregate_fit called for round {rnd}")
        if not results:
            return None, {}

        client_updates, client_ids, val_accs = [], [], []
        for _, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            tensors = [torch.from_numpy(p) for p in params]
            client_updates.append(tensors)
            cid = int(fit_res.metrics.get('cid', 0))
            client_ids.append(cid)
            val_acc = fit_res.metrics.get('val_acc', 0.5)
            val_accs.append(val_acc)

        print(f"📥 Received updates from clients: {client_ids}")
        print(f"📊 Validation accuracies: {val_accs}")

        # Compute gradient similarity (behavior metric)
        if len(client_updates) > 1:
            # Median of updates for similarity
            median_update = average_models(client_updates)  # simple average as proxy
            similarities = []
            for upd in client_updates:
                sim = 1.0 - sum((p - q).norm().item() for p,q in zip(upd, median_update)) / (1e-6 + sum(p.norm().item() for p in median_update))
                similarities.append(max(0.0, sim))
        else:
            similarities = [1.0] * len(client_updates)

        # Update trust scores using validation accuracy and gradient similarity
        if USE_TRUST_SCORING:
            for i, cid in enumerate(client_ids):
                # Combine: trust = alpha * old_trust + (1-alpha) * (val_weight*val_acc + (1-val_weight)*similarity)
                new_metric = TRUST_VALIDATION_WEIGHT * val_accs[i] + (1 - TRUST_VALIDATION_WEIGHT) * similarities[i]
                self.trust_scores[cid] = self.trust_scores.get(cid, 1.0) * TRUST_ALPHA + (1 - TRUST_ALPHA) * new_metric
            print(f"📊 Updated trust scores: {self.trust_scores}")

        # Choose aggregation method
        if AGGREGATION_METHOD == 'fedavg':
            aggregated = average_models(client_updates)
            selected_cids = client_ids
        elif AGGREGATION_METHOD == 'multikrum':
            n_selected = len(results) - 1
            selected_idx = multi_krum(client_updates, n_selected, k=MULTI_KRUM_K)
            selected_updates = [client_updates[i] for i in selected_idx]
            selected_cids = [client_ids[i] for i in selected_idx]
            aggregated = average_models(selected_updates)
        elif AGGREGATION_METHOD == 'trustflux':
            n_selected = len(results) - 1
            selected_idx = multi_krum(client_updates, n_selected, k=MULTI_KRUM_K)
            selected_updates = [client_updates[i] for i in selected_idx]
            selected_cids = [client_ids[i] for i in selected_idx]
            if USE_TRUST_SCORING:
                weights = [self.trust_scores[cid] for cid in selected_cids]
                weights = np.array(weights) / np.sum(weights)
            else:
                weights = None
            aggregated = average_models(selected_updates, weights)
        else:
            raise ValueError(f"Unknown AGGREGATION_METHOD: {AGGREGATION_METHOD}")

        set_parameters(self.global_model, aggregated)
        test_acc = evaluate_model(self.global_model, self.test_loader)
        print(f"📈 Test accuracy: {test_acc:.4f}")

        if AUDIT_ENABLED:
            h = model_hash(self.global_model)
            log_round(rnd, h, client_ids, selected_cids)
            self.metrics_history['merkle_root'].append(audit_log.root)

        self.metrics_history['rounds'].append(rnd)
        self.metrics_history['test_accuracy'].append(test_acc)
        self.metrics_history['trust_scores'].append(self.trust_scores.copy())
        self.metrics_history['selected'].append(selected_cids)

        self._write_live_metrics()

        aggregated_ndarrays = [p.cpu().numpy() for p in aggregated]
        aggregated_params = fl.common.ndarrays_to_parameters(aggregated_ndarrays)

        metrics = {
            'test_accuracy': test_acc,
            'trust_scores': self.trust_scores.copy(),
            'selected': selected_cids,
            'round': rnd
        }
        return aggregated_params, metrics

    def _write_live_metrics(self):
        try:
            data = {
                'experiment': EXPERIMENT_NAME,
                'round': self.round,
                'test_accuracy': self.metrics_history['test_accuracy'][-1] if self.metrics_history['test_accuracy'] else 0,
                'trust_scores': self.metrics_history['trust_scores'][-1] if self.metrics_history['trust_scores'] else {},
                'selected': self.metrics_history['selected'][-1] if self.metrics_history['selected'] else [],
                'merkle_root': audit_log.root if audit_log.root else ""
            }
            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            with open(METRICS_FILE, 'w') as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            print(f"✅ Metrics written for round {self.round}")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ Failed to write metrics: {e}")

    def save_experiment_metrics(self):
        out_file = os.path.join(os.path.dirname(METRICS_FILE), f"{EXPERIMENT_NAME}_full.json")
        try:
            data = {
                'experiment': EXPERIMENT_NAME,
                'config': {k: v for k, v in os.environ.items() if not k.startswith('_')},
                'rounds': self.metrics_history['rounds'],
                'test_accuracy': self.metrics_history['test_accuracy'],
                'trust_scores': self.metrics_history['trust_scores'],
                'selected': self.metrics_history['selected'],
                'merkle_roots': self.metrics_history['merkle_root']
            }
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Experiment metrics saved to {out_file}")
        except Exception as e:
            print(f"❌ Failed to save experiment metrics: {e}")

def start_server():
    print("🚀 start_server() called")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
    )
    server = TrustFluxServer(strategy=strategy)
    print("✅ TrustFluxServer instance created")
    time.sleep(2)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        server=server,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    )
    server.save_experiment_metrics()

if __name__ == "__main__":
    print("🐍 server.py is being executed directly")
    start_server()
    print("🏁 server.py finished")