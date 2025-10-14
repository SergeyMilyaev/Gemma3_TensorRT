import subprocess
import pandas as pd
import pynvml
import os
import json
from datasets import load_dataset

def get_gpu_type():
    """Returns the GPU type as a string."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8')
        return gpu_name
    except Exception as e:
        print(f"Could not determine GPU type: {e}")
        return "unknown"

def run_latency_benchmark(model_id, batch_sizes, seq_lengths, results_file):
    """Uses trtllm-bench to measure TTFT and TPOT across configurations."""
    print("--- Starting Latency Benchmark ---")

    command = [
        "trtllm-bench", "throughput",
        "--model", model_id,
        "--batch_size", ",".join(map(str, batch_sizes)),
        "--input_output_len", ",".join(seq_lengths),
        "--results_file", results_file,
        "--log_level", "info"
    ]

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    df = pd.read_csv(results_file)
    df_long = df.melt(
        id_vars=['batch_size', 'input_len', 'output_len', 'gpu_arch'],
        value_vars=['latency', 'tokens_per_sec', 'percentile_90_latency', 'percentile_95_latency', 'percentile_99_latency'],
        var_name='metric_name',
        value_name='metric_value'
    )

    df_long['benchmark_type'] = 'latency'
    df_long['metric_unit'] = df_long['metric_name'].apply(lambda x: 's' if 'latency' in x else 'tok/s')

    print("--- Latency Benchmark Complete ---")
    return df_long

def run_memory_benchmark(model_id, batch_sizes, seq_len):
    """Measures static and dynamic VRAM usage using pynvml."""
    print("--- Starting Memory Benchmark ---")

    try:
        from tensorrt_llm import LLM
        from transformers import AutoTokenizer
    except ImportError:
        print("TensorRT-LLM or Transformers not installed. Skipping memory benchmark.")
        return pd.DataFrame()

    results = []
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    mem_info_before = pynvml.nvmlDeviceGetMemoryInfo(handle)
    baseline_used_gb = mem_info_before.used / (1024**3)

    print(f"Loading model: {model_id}...")
    llm = LLM(model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Model and tokenizer loaded successfully.")

    mem_info_static = pynvml.nvmlDeviceGetMemoryInfo(handle)
    static_used_gb = (mem_info_static.used / (1024**3)) - baseline_used_gb
    results.append({
        'benchmark_type': 'memory',
        'metric_name': 'static_vram',
        'metric_value': static_used_gb,
        'metric_unit': 'GB',
        'batch_size': None,
        'input_len': None,
        'output_len': None
    })

    for bs in batch_sizes:
        print(f"Running memory benchmark for batch size: {bs}")
        prompts = [" ".join(["test"] * (seq_len // 2))] * bs
        outputs = llm.generate(prompts, max_new_tokens=seq_len//2)

        mem_info_dynamic = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak_used_gb = (mem_info_dynamic.used / (1024**3)) - baseline_used_gb
        dynamic_used_gb = peak_used_gb - static_used_gb

        results.append({
            'benchmark_type': 'memory',
            'metric_name': 'peak_dynamic_vram',
            'metric_value': dynamic_used_gb,
            'metric_unit': 'GB',
            'batch_size': bs,
            'input_len': seq_len // 2,
            'output_len': seq_len // 2
        })

    pynvml.nvmlShutdown()
    print("--- Memory Benchmark Complete ---")
    return pd.DataFrame(results)

def run_accuracy_benchmark(model_id):
    """Runs IFEval using lighteval and the custom task suite."""
    print("--- Starting Accuracy Benchmark ---")
    results = []

    print("Running IFEval benchmark...")
    ifeval_output_dir = "./ifeval_results"
    ifeval_command = [
        "lighteval",
        "--model_args", f"pretrained={model_id}",
        "--tasks", "extended|ifeval|0|0",
        "--output_dir", ifeval_output_dir,
        "--override_batch_size", "1"
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

    try:
        from tensorrt_llm import LLM
        from transformers import AutoTokenizer
        llm = LLM(model=model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except ImportError:
        print("TensorRT-LLM or Transformers not installed. Skipping accuracy benchmark.")
        return pd.DataFrame(results)

    print("Running Sentiment Analysis (GLUE/SST2) benchmark...")
    dataset = load_dataset("glue", "sst2", split="validation")
    metric = evaluate.load("accuracy")

    predictions, references = [], []
    sample_dataset = dataset.select(range(100))
    for item in sample_dataset:
        prompt = f"Sentence: {item['sentence']}\nQuestion: What is the sentiment of this sentence? Answer with 'positive' or 'negative'."
        outputs = llm.generate([prompt], max_new_tokens=5)
        output_text = outputs[0].text.lower()

        if "positive" in output_text:
            predicted_label = 1
        elif "negative" in output_text:
            predicted_label = 0
        else:
            predicted_label = -1

        predictions.append(predicted_label)
        references.append(item['label'])

    accuracy = metric.compute(predictions=predictions, references=references)
    results.append({
        'benchmark_type': 'accuracy',
        'metric_name': 'sst2_accuracy',
        'metric_value': accuracy['accuracy'],
        'metric_unit': 'accuracy',
        'task': 'glue/sst2'
    })

    print("--- Accuracy Benchmark Complete ---")
    return pd.DataFrame(results)
