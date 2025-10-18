import subprocess
import pandas as pd
import pynvml
import os
import json
from datasets import load_dataset
import evaluate

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

    dataset_file = "latency_dataset.jsonl"
    
    # Generate dataset using prepare_dataset.py
    for seq_len_pair in seq_lengths:
        input_len, output_len = map(int, seq_len_pair.split(','))
        prepare_dataset_command = [
            "python",
            "TensorRT-LLM/benchmarks/cpp/prepare_dataset.py",
            "--tokenizer", model_id,
            "--trust-remote-code",
            "--stdout",
            "token-norm-dist",
            "--input-mean", str(input_len),
            "--output-mean", str(output_len),
            "--input-stdev", "0",
            "--output-stdev", "0",
            "--num-requests", "1"
        ]
        print(f"Running command: {' '.join(prepare_dataset_command)}")
        with open(dataset_file, "a") as f:
            subprocess.run(prepare_dataset_command, stdout=f, check=True)

    command = [
        "trtllm-bench",
        "--model", model_id,
        "latency",
        "--dataset", dataset_file,
        "--report_json", results_file,
        "--backend", "pytorch",
    ]

    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    with open(results_file, "r") as f:
        results = json.load(f)

    data = []
    # The structure of the JSON output is a dictionary with a "performance" key
    # which contains a list of dictionaries, one for each request.
    performance_results = results.get("performance", [])
    for r in performance_results:
        data.append({
            'batch_size': 1, # Latency benchmark is run with batch size 1
            'input_len': r.get('input_length'),
            'output_len': r.get('output_length'),
            'metric_name': 'latency',
            'metric_value': r.get('latency'),
            'metric_unit': 'ms',
            'benchmark_type': 'latency'
        })
        data.append({
            'batch_size': 1,
            'input_len': r.get('input_length'),
            'output_len': r.get('output_length'),
            'metric_name': 'time_to_first_token',
            'metric_value': r.get('time_to_first_token'),
            'metric_unit': 'ms',
            'benchmark_type': 'latency'
        })
        data.append({
            'batch_size': 1,
            'input_len': r.get('input_length'),
            'output_len': r.get('output_length'),
            'metric_name': 'tokens_per_second',
            'metric_value': r.get('tokens_per_second'),
            'metric_unit': 'tok/s',
            'benchmark_type': 'latency'
        })

    df_long = pd.DataFrame(data)

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
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
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
        outputs = llm.generate(prompts)

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