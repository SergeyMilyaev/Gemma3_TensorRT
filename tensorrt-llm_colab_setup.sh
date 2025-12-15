#!/bin/sh
apt-get update && apt-get -y install cuda-12-8 git git-lfs
git lfs install

TRTLLM_VER="1.1.0rc5"
git clone -b v$TRTLLM_VER https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

TRTLLM_USE_PRECOMPILED=$TRTLLM_VER pip -v wheel . --no-deps --wheel-dir ./build
pip install ./build/tensorrt_llm*.whl
