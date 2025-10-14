from datetime import datetime
import pandas as pd
from IPython.display import display
from benchmarks import run_latency_benchmark, run_memory_benchmark, run_accuracy_benchmark, get_gpu_type

def main(model_id="google/gemma-3-270m-it-qat"):
    MODEL_ID = model_id
    LATENCY_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    LATENCY_SEQ_LENGTHS = ["64,64", "512,128", "2048,256", "4096,512"]
    MEMORY_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    MEMORY_MAX_SEQ_LEN = 4096

    all_results = []
    gpu_type = get_gpu_type()

    try:
        latency_df = run_latency_benchmark(MODEL_ID, LATENCY_BATCH_SIZES, LATENCY_SEQ_LENGTHS, "gemma_270m_latency_results.csv")
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

    try:
        accuracy_df = run_accuracy_benchmark(MODEL_ID)
        accuracy_df['model_id'] = MODEL_ID
        accuracy_df['gpu_type'] = gpu_type
        accuracy_df['timestamp'] = datetime.now().isoformat()
        all_results.append(accuracy_df)
    except Exception as e:
        print(f"Accuracy benchmark failed: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols = ['timestamp', 'model_id', 'gpu_type', 'benchmark_type', 'batch_size', 'input_len', 'output_len', 'metric_name', 'metric_value', 'metric_unit']
        for col in cols:
            if col not in final_df.columns:
                final_df[col] = None
        final_df = final_df[cols]

        output_file = "gemma_270m_benchmark_report.csv"
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
