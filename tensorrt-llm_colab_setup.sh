#!/bin/sh
apt-get update && apt-get -y install cuda-12-8

pip3 install -r requirements.txt
git clone -b v1.1.0rc5 https://github.com/NVIDIA/TensorRT-LLM.git