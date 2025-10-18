import subprocess
import os
import json
import pandas as pd

def run_accuracy_benchmark(model_id):
    """Runs IFEval using lighteval and the custom task suite."""
    print("--- Starting Accuracy Benchmark ---")
    results = []

    print("Running IFEval benchmark...")
    ifeval_output_dir = "./ifeval_results"
    ifeval_command = [
        "lighteval",
        "accelerate",
        f"model_name={model_id},trust_remote_code=True",
        "extended|ifeval|0|0",
        "--output-dir", ifeval_output_dir
    ]
    print(f"Running command: {' '.join(ifeval_command)}")
    subprocess.run(ifeval_command, check=True)

    try:
        with open(os.path.join(ifeval_output_dir, "results.json"), "r") as f:
            ifeval_results = json.load(f)
            ifeval_score = ifeval_results.get('results', {}).get('extended|ifeval|0|0', {}).get('strict_accuracy,none', 0.0)
        results.append({
            'benchmark_type': 'accuracy',
            'metric_name': 'ifeval_strict_accuracy',
            'metric_value': ifeval_score,
            'metric_unit': 'accuracy',
            'task': 'ifeval'
        })
    except Exception as e:
        print(f"Could not parse IFEval results: {e}")

    print("--- Accuracy Benchmark Complete ---")
    return pd.DataFrame(results)
