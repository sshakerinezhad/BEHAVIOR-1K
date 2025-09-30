#!/bin/bash

conda create -n behavior python=3.10 -c conda-forge -y
eval "$(conda shell.bash hook)"
conda activate behavior
echo $CONDA_DEFAULT_ENV

pip install "numpy<2" "setuptools<=79"
CUDA_VER_SHORT=$(echo $CUDA_VERSION | sed 's/\.//g')
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}

./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

python -m pip install -e /workspace/openpi/packages/openpi-client/
python -m pip install -e /workspace/openpi/
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
