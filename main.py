from datetime import datetime
import pandas as pd
from IPython.display import display
from benchmarks import run_latency_benchmark, run_memory_benchmark, get_gpu_type
import yaml
import os

def main(model_id="google/gemma-3-1b-it"):
    MODEL_ID = model_id

    with open("latency_benchmark.yaml", "r") as f:
        latency_config = yaml.safe_load(f)
    
    with open("memory_benchmark.yaml", "r") as f:
        memory_config = yaml.safe_load(f)

    LATENCY_BATCH_SIZES = latency_config["batch_sizes"]
    LATENCY_SEQ_LENGTHS = latency_config["seq_lengths"]
    MEMORY_BATCH_SIZES = memory_config["batch_sizes"]
    MEMORY_MAX_SEQ_LEN = memory_config["max_seq_len"]
    
    results_file = f"{MODEL_ID.replace('/', '_')}_latency_results.json"
    output_file = f"{MODEL_ID.replace('/', '_')}_benchmark_report.csv"

    all_results = []
    gpu_type = get_gpu_type()

    try:
        latency_df = run_latency_benchmark(MODEL_ID, LATENCY_BATCH_SIZES, LATENCY_SEQ_LENGTHS, results_file)
        latency_df['model_id'] = MODEL_ID
        latency_df['gpu_type'] = gpu_type
        latency_df['timestamp'] = datetime.now().isoformat()
        all_results.append(latency_df)
    except Exception as e:
        print(f"Latency benchmark failed: {e}")

    try:
        memory_df = run_memory_benchmark(MODEL_ID, MEMORY_BATCH_SIZES, MEMORY_MAX_SEQ_LEN)
        memory_df['model_id'] = MODEL_ID
        memory_df['gpu_type'] = gpu_type
        memory_df['timestamp'] = datetime.now().isoformat()
        all_results.append(memory_df)
    except Exception as e:
        print(f"Memory benchmark failed: {e}")

    print("To run the accuracy benchmark, please run the `accuracy_benchmark.py` script in a separate environment with the required dependencies.")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols = ['timestamp', 'model_id', 'gpu_type', 'benchmark_type', 'batch_size', 'input_len', 'output_len', 'metric_name', 'metric_value', 'metric_unit']
        for col in cols:
            if col not in final_df.columns:
                final_df[col] = None
        final_df = final_df[cols]

        final_df.to_csv(output_file, index=False)
        print(f"Benchmark complete. Results saved to {output_file}")
        display(final_df)
    else:
        print("No benchmark results were generated.")

if __name__ == "__main__":
    # To run with a different model, pass it as an argument.
    # e.g. python main.py "google/gemma-3-9b-it-qat"
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
