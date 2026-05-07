import subprocess
import time
import os
import json
import pandas as pd
from config import METRICS_DIR

EXPERIMENTS = [
    {
        "name": "1_Baseline_FedAvg",
        "env": {
            "ATTACK": "",
            "AGGREGATION_METHOD": "fedavg",
            "USE_TRUST_SCORING": "false",
            "NUM_ROUNDS": "20"
        }
    },
    {
        "name": "2_FedAvg_Attack",
        "env": {
            "ATTACK": "amplify",
            "AGGREGATION_METHOD": "fedavg",
            "USE_TRUST_SCORING": "false",
            "NUM_ROUNDS": "20"
        }
    },
    {
        "name": "3_MultiKrum_Only",
        "env": {
            "ATTACK": "amplify",
            "AGGREGATION_METHOD": "multikrum",
            "USE_TRUST_SCORING": "false",
            "NUM_ROUNDS": "20"
        }
    },
    {
        "name": "4_TrustFlux",
        "env": {
            "ATTACK": "amplify",
            "AGGREGATION_METHOD": "trustflux",
            "USE_TRUST_SCORING": "true",
            "NUM_ROUNDS": "20"
        }
    }
]

def run_experiment(exp):
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env.update(exp['env'])
    env['EXPERIMENT_NAME'] = exp['name']
    env['USE_SYNTHETIC'] = 'true'

    try:
        subprocess.run(["docker", "compose", "up", "--build", "--abort-on-container-exit"], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp['name']} failed: {e}")
    finally:
        subprocess.run(["docker", "compose", "down"])

    time.sleep(2)

def collect_results():
    records = []
    for exp in EXPERIMENTS:
        fname = os.path.join(METRICS_DIR, f"{exp['name']}_full.json")
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
            final_acc = data['test_accuracy'][-1] if data['test_accuracy'] else 0
            records.append({
                'Experiment': exp['name'],
                'Final Accuracy': final_acc,
                'Rounds': len(data['rounds'])
            })
        except FileNotFoundError:
            print(f"Warning: {fname} not found")
    baseline = None
    for rec in records:
        if rec['Experiment'] == '1_Baseline_FedAvg':
            baseline = rec['Final Accuracy']
    if baseline:
        for rec in records:
            if rec['Experiment'] != '1_Baseline_FedAvg':
                rec['Attack Success Rate'] = 1 - (rec['Final Accuracy'] / baseline)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(METRICS_DIR, "experiment_summary.csv"), index=False)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)

    for exp in EXPERIMENTS:
        run_experiment(exp)

    collect_results()