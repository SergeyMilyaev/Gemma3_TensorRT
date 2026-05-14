#!/bin/sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-13-0

export PATH=/usr/local/cuda-13.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

pip uninstall -y torch torchvision torchaudio
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu130

apt-get -y install libopenmpi-dev
apt-get -y install libzmq3-dev

pip3 install --ignore-installed pip setuptools wheel && pip3 install tensorrt_llm==1.3.0rc14