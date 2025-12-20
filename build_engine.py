import argparse
import subprocess
import os
import sys
from pathlib import Path
import yaml

def run_command(command, description):
    print(f"--- {description} ---")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download HF checkpoint and build TensorRT-LLM engine.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Hugging Face model ID")
    parser.add_argument("--download_dir", type=str, default="checkpoints/hf", help="Directory to download HF checkpoint")
    parser.add_argument("--converted_dir", type=str, default="checkpoints/trtllm", help="Directory to save converted TRT-LLM checkpoint")
    parser.add_argument("--engine_dir", type=str, default="engines", help="Directory to save built engine")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the engine (e.g., bfloat16, float16)")
    parser.add_argument("--world_size", type=str, default="1", help="World size (number of GPUs)")
    
    args = parser.parse_args()

    model_name = args.model_id.split("/")[-1]
    hf_ckpt_path = Path(args.download_dir) / model_name
    trt_ckpt_path = Path(args.converted_dir) / model_name / args.dtype / f"tp{args.world_size}"
    engine_path = Path(args.engine_dir) / model_name / args.dtype / f"tp{args.world_size}"

    # 1. Download from Hugging Face
    # Ensure huggingface-cli is available or install it
    download_cmd = [
        "huggingface-cli", "download",
        args.model_id,
        "--local-dir", str(hf_ckpt_path),
        "--local-dir-use-symlinks", "False"
    ]
    run_command(download_cmd, "Downloading model from Hugging Face")

    # 2. Convert Checkpoint
    # We assume we are in the project root and the conversion script is at TensorRT-LLM/examples/models/core/gemma/convert_checkpoint.py
    # This logic assumes a Gemma-like model structure. 
    convert_script = Path("TensorRT-LLM/examples/models/core/gemma/convert_checkpoint.py")
    if not convert_script.exists():
        print(f"Error: Conversion script not found at {convert_script}")
        sys.exit(1)

    convert_cmd = [
        sys.executable, str(convert_script),
        "--ckpt-type", "hf",
        "--model-dir", str(hf_ckpt_path),
        "--dtype", args.dtype,
        "--world-size", args.world_size,
        "--output-model-dir", str(trt_ckpt_path)
    ]
    run_command(convert_cmd, "Converting HF checkpoint to TensorRT-LLM format")

    # 3. Build Engine
    build_cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(trt_ckpt_path),
        "--gemm_plugin", "auto",
        "--output_dir", str(engine_path)
    ]
    
    # Infer parameters from benchmarks
    import re
    
    # Defaults
    default_max_input_len = 3000
    default_max_seq_len = 3100
    
    bench_max_input_len = 0
    bench_max_seq_len = 0
    
    # Parse latency_benchmark.yaml
    latency_file = Path("latency_benchmark.yaml")
    if latency_file.exists():
        try:
            with open(latency_file, 'r') as f:
                latency_config = yaml.safe_load(f)
            if 'benchmarks' in latency_config:
                for benchmark in latency_config['benchmarks']:
                    if 'input_output_len' in benchmark:
                        for io_len_str in benchmark['input_output_len']:
                            inp, out = map(int, io_len_str.split(','))
                            bench_max_input_len = max(bench_max_input_len, inp)
                            bench_max_seq_len = max(bench_max_seq_len, inp + out)
        except Exception as e:
            print(f"Warning: parsing latency_benchmark.yaml failed: {e}")

    # Parse memory_benchmark.yaml
    memory_file = Path("memory_benchmark.yaml")
    if memory_file.exists():
        try:
            with open(memory_file, 'r') as f:
                memory_config = yaml.safe_load(f)
            if 'max_seq_len' in memory_config:
                val = int(memory_config['max_seq_len'])
                bench_max_seq_len = max(bench_max_seq_len, val)
                bench_max_input_len = max(bench_max_input_len, val)
        except Exception as e:
            print(f"Warning: parsing memory_benchmark.yaml failed: {e}")

    # Determine final values
    final_max_input_len = bench_max_input_len if bench_max_input_len > 0 else default_max_input_len
    final_max_seq_len = bench_max_seq_len if bench_max_seq_len > 0 else default_max_seq_len
    final_max_seq_len = max(final_max_seq_len, final_max_input_len)

    print(f"Inferred max_input_len: {final_max_input_len}, max_seq_len: {final_max_seq_len}")

    # Gemma 3 specific default parameters (as seen in README)
    # Adjust these defaults if necessary or expose them as arguments
    if "gemma" in args.model_id.lower():
         build_cmd.extend([
             "--max_batch_size", "8",
             "--max_input_len", str(final_max_input_len),
             "--max_seq_len", str(final_max_seq_len)
         ])

    run_command(build_cmd, "Building TensorRT-LLM Engine")

    print(f"\nSuccess! Engine built at: {engine_path}")

if __name__ == "__main__":
    main()
