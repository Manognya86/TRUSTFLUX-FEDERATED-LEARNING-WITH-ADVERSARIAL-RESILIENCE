# 🛡️ TrustFlux — A Federated Learning Framework

> **Trustworthy AI · Term Project**
> Secure, privacy-preserving federated learning for medical imaging with Byzantine-robust aggregation, adaptive trust scoring, and cryptographic audit trails.

---

## 👥 Authors

| Name | 
|---|
| Manognya Lokesh Reddy |
| Vanshika Sangtani |
| Pratik Rajkumar Jadhav |
| Kavan Kumareshan |
| Akshara Devarakonda |

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Attack Modes](#attack-modes)
- [Defense Mechanisms](#defense-mechanisms)
- [Dashboard](#dashboard)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Known Issues & Fixes](#known-issues--fixes)
- [Future Work](#future-work)

---

## Overview

TrustFlux is a federated learning framework designed to answer the question:

> *"How can we design a federated learning system that maintains diagnostic accuracy while being inherently resilient to model poisoning attacks through computationally efficient, interpretable defense mechanisms?"*

In healthcare, hospitals cannot share raw patient data due to HIPAA and GDPR regulations. Federated Learning solves this by training models locally and sharing only gradient updates. However, standard Federated Averaging (FedAvg) treats all client updates equally — a single malicious client can corrupt the global model entirely.

TrustFlux layers three defenses on top of federated learning:
1. **Multi-Krum** — Byzantine-robust aggregation that filters outlier gradient updates
2. **Adaptive Trust Scoring** — per-client reputation that evolves across rounds via Exponential Moving Average
3. **SHA-256 Merkle Audit Trail** — tamper-evident cryptographic log of every training round

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Network: fl_net                  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              FL Server  (port 8080)              │   │
│  │                                                  │   │
│  │  server.py ──► defense.py ──► audit.py           │   │
│  │  (Orchestrator) (Multi-Krum + Trust) (Merkle)    │   │
│  │                    │                             │   │
│  │              config.py (Env-Driven)              │   │
│  └──────────────────────┬───────────────────────────┘   │
│            gRPC          │           gRPC                │
│   ┌──────────────────────▼───────────────────────────┐  │
│   │           FL Clients  (client1 – client5)        │  │
│   │                                                  │  │
│   │  client.py (CheXpertClient / NumPyClient)        │  │
│   │  data.py   (Non-IID Partitioning)                │  │
│   │  attacks.py (poison_update — injected here)      │  │
│   └──────────────────────────────────────────────────┘  │
│                         │                               │
│              trustflux_metrics (Docker Volume)          │
│                         │                               │
│   ┌─────────────────────▼────────────────────────────┐  │
│   │           Dashboard  (port 8050)                 │  │
│   │  dashboard.py — Plotly Dash — 4-tab live UI      │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| 🔒 Privacy | Raw data never leaves each client — only gradients are transmitted |
| ⚔️ 5 Attack Types | amplify, reverse, noise, label_flip, model_substitution |
| 🛡️ Multi-Krum | Byzantine-robust aggregation — excludes outlier gradient updates |
| 📊 Trust Scoring | Per-client EMA trust score evolving across rounds |
| 🔗 Merkle Audit Trail | SHA-256 tamper-evident log of every federated round |
| 📡 Real-Time Dashboard | Plotly Dash UI — accuracy gauge, trust heatmap, audit logs, config |
| 🐳 Fully Containerised | Docker Compose — 7 services, single command startup |
| 🔬 3 Datasets | CheXpert, ISIC 2019, Synthetic (default — no download needed) |
| 🧪 4-Experiment Pipeline | Automated benchmark: Baseline → Attack → Krum → TrustFlux |
| ⚙️ Zero Code Changes | All parameters controlled via environment variables |

---

## Project Structure

```
trustflux/
│
├── server.py              # FL coordination server — orchestrates rounds, runs defense
├── client.py              # FL client — local training loop, gradient submission
├── defense.py             # Multi-Krum, update_trust_scores(), compute_gradient_similarity()
├── attacks.py             # poison_update() — all 5 attack strategies
├── audit.py               # MerkleTree class — SHA-256 tamper-evident logging
├── data.py                # load_client_data() — Non-IID partitioning logic
├── models.py              # CheXpertCNN — lightweight CNN, adapts to 3 datasets
├── config.py              # All constants read from environment variables at import time
├── dashboard.py           # Plotly Dash 4-tab real-time monitoring UI
├── evaluation.py          # run_experiment(), collect_results() — CSV benchmark output
├── synthetic.py           # Synthetic medical imaging data generator
├── utils.py               # Shared utility functions
│
├── Dockerfile.client      # Single Dockerfile used by server, all clients, and dashboard
├── docker-compose.yml     # 7-service definition — server, client1-5, dashboard
├── requirements.txt       # Python dependencies
│
├── metrics/
│   └── live_metrics.json  # Written by server after every round, read by dashboard
│
└── dummy_data/            # Empty placeholder — used when no real dataset is mounted
    └── .gitkeep
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker Desktop | 4.x or later |
| Docker Compose | v2 (included with Docker Desktop) |
| Windows PowerShell | 5.1+ or PowerShell 7+ |
| Available RAM | 8 GB minimum recommended |
| Available Disk | 6 GB (for Docker images) |

> **No Python installation required on your host machine.** Everything runs inside Docker containers.

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/trustflux.git
cd trustflux
```

### 2. Run with synthetic data (no dataset download needed)

```powershell
# PowerShell
$env:USE_SYNTHETIC = "true"
docker compose up --build
```

```bash
# bash / macOS / Linux
USE_SYNTHETIC=true docker compose up --build
```

### 3. Open the dashboard

Once you see this line in the logs:
```
dashboard-1  | 🌐 Dashboard running at http://localhost:8050
```

Open your browser and go to:
```
http://localhost:8050
```

### 4. Full clean rebuild (recommended after any code change)

```powershell
# PowerShell
$env:USE_SYNTHETIC = "true"
docker compose down -v
docker system prune -a -f
docker compose build --no-cache
docker compose up
```

---

## Configuration

All parameters are set via environment variables — no code changes needed. Set them in your shell before running `docker compose up`, or edit `docker-compose.yml` directly.

| Variable | Default | Description |
|---|---|---|
| `USE_SYNTHETIC` | `true` | Use synthetic data — no external dataset required |
| `DATASET` | `synthetic` | Dataset to use: `synthetic`, `chexpert`, or `isic` |
| `NUM_CLIENTS` | `5` | Number of federated learning clients |
| `NUM_ROUNDS` | `20` | Number of federated training rounds |
| `BATCH_SIZE` | `32` | Local training batch size |
| `EPOCHS_PER_CLIENT` | `2` | Local epochs per client per round |
| `AGGREGATION_METHOD` | `trustflux` | `fedavg`, `multikrum`, or `trustflux` |
| `MULTI_KRUM_K` | `2` | Number of nearest neighbours for Krum scoring |
| `TRUST_ALPHA` | `0.9` | EMA decay factor for trust score updates |
| `TRUST_VALIDATION_WEIGHT` | `0.5` | Weight of validation accuracy vs gradient similarity in trust |
| `ATTACK` | *(empty)* | Attack type: `amplify`, `reverse`, `noise`, `label_flip`, `model_substitution` |
| `ATTACK_SCALE` | `100.0` | Amplification factor for the `amplify` attack |
| `ATTACK_NOISE_STD` | `0.1` | Gaussian noise standard deviation for the `noise` attack |
| `MALICIOUS_CLIENT_ID` | `0` | Which client ID is the attacker |
| `ATTACK_TARGET_CLASS` | `0` | Target class for `label_flip` attack |
| `AUDIT_ENABLED` | `true` | Enable SHA-256 Merkle audit trail |
| `METRICS_DIR` | `/app/metrics` | Directory for metrics output (inside container) |
| `CHEXPERT_PATH` | `./dummy_data` | Host path to CheXpert dataset |
| `ISIC_PATH` | `./dummy_data` | Host path to ISIC 2019 dataset |

### Example: Run with amplify attack and no defense

```powershell
$env:USE_SYNTHETIC = "true"
$env:ATTACK = "amplify"
$env:AGGREGATION_METHOD = "fedavg"
docker compose up --build
```

### Example: Run full TrustFlux defense against amplify attack

```powershell
$env:USE_SYNTHETIC = "true"
$env:ATTACK = "amplify"
$env:AGGREGATION_METHOD = "trustflux"
docker compose up --build
```

---

## Attack Modes

All attacks are implemented in `attacks.py` via the `poison_update()` function, which is called by the malicious client after local training.

| Attack | Type | Mechanism |
|---|---|---|
| `amplify` | Post-training | Multiplies all gradient parameters by `ATTACK_SCALE` (default 100×). Overwhelms FedAvg aggregation. |
| `reverse` | Post-training | Multiplies all gradient parameters by −1. Maximises divergence from the true optimum. |
| `noise` | Post-training | Adds Gaussian noise (std = `ATTACK_NOISE_STD`) to all parameters. Gradual degradation — hard to detect. |
| `label_flip` | During training | Flips training labels via `maybe_flip_labels()` inside `client.py`. Data poisoning — gradients look structurally normal. |
| `model_substitution` | Post-training | Zeroes out all gradient parameters before submission. Silent denial-of-service — degrades convergence over time. |

---

## Defense Mechanisms

All defenses are implemented in `defense.py`.

### Multi-Krum (`multi_krum()`)

For each client, computes the sum of squared L2 distances to its K nearest neighbours in gradient space. Clients with the lowest scores (closest to the honest majority cluster) are selected. Provably Byzantine-robust when fewer than (n−1)/2 clients are malicious.

```
Krum score(i) = Σ  ||gradient_i − gradient_j||²
              j ∈ K-nearest-neighbours(i)
```

### Adaptive Trust Scoring (`update_trust_scores()`)

Each client starts with a trust score of 1.0. After every round the server updates scores using an Exponential Moving Average:

```
trust[client] = TRUST_ALPHA × trust[client] + (1 − TRUST_ALPHA) × metric

where metric = TRUST_VALIDATION_WEIGHT  × validation_accuracy
             + (1 − TRUST_VALIDATION_WEIGHT) × gradient_similarity
```

With `TRUST_ALPHA = 0.9`: a consistently honest client converges to 1.0, a consistently malicious client converges to 0.0 over approximately 10 rounds.

### Gradient Similarity (`compute_gradient_similarity()`)

Computes the per-parameter median across all clients and measures each client's normalised inverse L2 distance from that median. Score of 1.0 = identical to consensus. Score near 0.0 = maximally divergent. Used as the secondary trust signal and displayed in the dashboard heatmap.

### SHA-256 Merkle Audit Trail (`audit.py`)

After every round, `log_round()` hashes the round data (round number, global model hash, selected clients, accuracy) as a leaf in a Merkle tree. The root hash is a single cryptographic fingerprint of the entire training history. Any tampering with historical data changes the root hash — providing tamper-evident compliance logging.

---

## Dashboard

Open `http://localhost:8050` after startup.

| Tab | Contents |
|---|---|
| **Overview** | Accuracy gauge, live Merkle root, trust score heatmap (RdYlGn), client selection bar chart |
| **Trust Analytics** | Per-client trust score bars — green = trusted, red = suspicious |
| **Audit Logs** | Full table of every completed round: round number, Merkle root, selected clients, accuracy |
| **Config** | Live JSON dump of all environment variables — full experiment reproducibility at a glance |

The dashboard auto-refreshes every 2 seconds without page reload.

---

## Evaluation

`evaluation.py` implements a four-experiment automated benchmark pipeline.

| Experiment | Attack | Defense | Purpose |
|---|---|---|---|
| 1 — Baseline FedAvg | None | None | Clean accuracy ceiling |
| 2 — FedAvg under Attack | Amplify | None | Quantify FedAvg vulnerability |
| 3 — Multi-Krum Only | Amplify | Multi-Krum | Isolate Krum's Byzantine robustness |
| 4 — Full TrustFlux | Amplify | Multi-Krum + Trust | Full composite defense |

Results are saved to CSV. The key metric is **Attack Success Rate (ASR)** — the fraction of rounds where the attack successfully degraded accuracy below the clean baseline threshold.

To run the evaluation pipeline manually:

```powershell
docker exec trustflux-server-1 python evaluation.py
```

---

## Datasets

| Dataset | Classes | Task | Setup |
|---|---|---|---|
| **Synthetic** | 5 | Multi-label classification | Built-in — set `USE_SYNTHETIC=true`, no download needed |
| **CheXpert** | 5 | Chest X-ray multi-label (Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion) | Download from [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) and set `CHEXPERT_PATH` |
| **ISIC 2019** | 8 | Skin lesion multi-class | Download from [ISIC Archive](https://challenge.isic-archive.com/data/#2019) and set `ISIC_PATH` |

---

## Known Issues & Fixes

### Dashboard shows "Loading..." and never updates

**Cause:** `serve_locally` was not set, so Dash attempted to load Bootstrap CSS from an external CDN. On Windows Docker this request is blocked silently, preventing the Dash callback renderer from initialising — so `dcc.Interval` never fires.

**Fix applied in `dashboard.py`:**
```python
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    serve_locally=True,               # serve all assets locally — no CDN calls
    suppress_callback_exceptions=True
)

app.run_server(
    host='0.0.0.0',
    port=8050,
    debug=False,
    use_reloader=False,   # prevents Werkzeug spawning a child process in Docker
    threaded=True         # handles interval callbacks without blocking
)
```

### Site can't be reached at localhost:8050

**Cause:** The `dashboard` service in `docker-compose.yml` depended on `server: healthy`, but the server only becomes healthy after Flower binds port 8080, which requires clients to connect first — creating a startup deadlock.

**Fix:** Increased healthcheck `start_period` to 20 seconds and `retries` to 30 to give the server enough time to start before clients attempt connection.

### Server crashes with empty ATTACK variable

**Cause:** `docker-compose.yml` passed `ATTACK=${ATTACK}` with no fallback. When unset in the host shell, Docker passed an empty string instead of `None`, breaking the attack conditional logic.

**Fix:** Changed to `ATTACK=${ATTACK:-}` — explicit empty default.

### Build fails with import errors

**Cause:** `requirements.txt` included `docker==6.1.3` (the Docker SDK for Python), which is not used in any script and conflicts with the container environment.

**Fix:** Removed `docker==6.1.3` from `requirements.txt`.

---

## Future Work

- **Multi-adversary support** — extend Krum and trust scoring for 2+ colluding malicious clients
- **Differential Privacy** — integrate DP-SGD via the Opacus library for formal privacy guarantees
- **Secure Aggregation** — encrypt gradients so the server never sees raw client updates
- **Personalized Federated Learning** — per-client fine-tuning layers for better Non-IID performance
- **Dashboard anomaly alerts** — push notification when a client's trust score drops below a threshold
- **Blockchain audit trail** — migrate the Merkle tree to an on-chain smart contract
- **Expanded datasets** — NIH ChestX-ray14 and MIMIC-CXR data loaders
- **Inference attack defenses** — membership inference and gradient inversion protections

---

## Tech Stack

| Component | Technology |
|---|---|
| Federated Learning Framework | [Flower (flwr) 1.5.0](https://flower.dev/) |
| Deep Learning | PyTorch 2.0.1 + TorchVision 0.15.2 |
| Dashboard | Plotly Dash 2.14.0 + Dash Bootstrap Components 1.5.0 |
| Containerisation | Docker Compose v2 |
| Cryptographic Audit | SHA-256 via Python `hashlib` |
| Data Processing | NumPy 1.24.3 + Pandas 2.0.3 |
| Image Processing | Pillow 9.5.0 |

---

## License

This project was developed as a term project for the Trustworthy AI course. All rights reserved by the authors.
