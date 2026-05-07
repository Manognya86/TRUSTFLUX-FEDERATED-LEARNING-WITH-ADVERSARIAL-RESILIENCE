# config.py
import os

# ----------------------------
# Federated learning settings
# ----------------------------
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "5"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "20"))
FRACTION_FIT = float(os.getenv("FRACTION_FIT", "1.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS_PER_CLIENT = int(os.getenv("EPOCHS_PER_CLIENT", "2"))

# ----------------------------
# Model
# ----------------------------
IMG_SIZE = 224
DATASET = os.getenv("DATASET", "synthetic")  # 'chexpert', 'isic', 'synthetic'
if DATASET == "chexpert":
    NUM_CLASSES = 5
elif DATASET == "isic":
    NUM_CLASSES = 8  # ISIC 2019 has 8 disease categories
else:
    NUM_CLASSES = 5  # synthetic defaults to 5

# ----------------------------
# Dataset paths
# ----------------------------
CHEXPERT_ROOT = os.getenv("CHEXPERT_ROOT", "/data/chexpert")
ISIC_ROOT = os.getenv("ISIC_ROOT", "/data/isic")
USE_SYNTHETIC = os.getenv("USE_SYNTHETIC", "true").lower() == "true"

# ----------------------------
# Aggregation & Defense
# ----------------------------
AGGREGATION_METHOD = os.getenv("AGGREGATION_METHOD", "trustflux")  # fedavg, multikrum, trustflux
MULTI_KRUM_K = int(os.getenv("MULTI_KRUM_K", "2"))
USE_TRUST_SCORING = os.getenv("USE_TRUST_SCORING", "true").lower() == "true"
TRUST_ALPHA = float(os.getenv("TRUST_ALPHA", "0.9"))       # decay factor
TRUST_VALIDATION_WEIGHT = float(os.getenv("TRUST_VALIDATION_WEIGHT", "0.5"))  # weight of validation accuracy in trust

# ----------------------------
# Attack
# ----------------------------
ATTACK = os.getenv("ATTACK", None)          # None, 'amplify', 'reverse', 'noise', 'label_flip', 'model_substitution'
ATTACK_PARAMS = {
    'scale': float(os.getenv("ATTACK_SCALE", "100.0")),
    'noise_std': float(os.getenv("ATTACK_NOISE_STD", "0.1")),
    'target_class': int(os.getenv("ATTACK_TARGET_CLASS", "0"))  # for label flipping
}
# Malicious client ID (default 0)
MALICIOUS_CLIENT_ID = int(os.getenv("MALICIOUS_CLIENT_ID", "0"))

# ----------------------------
# Audit
# ----------------------------
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() == "true"

# ----------------------------
# Metrics & Dashboard
# ----------------------------
METRICS_DIR = os.getenv("METRICS_DIR", "/app/metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(METRICS_DIR, "live_metrics.json")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "unnamed")