#!/bin/bash

# This script creates a conda environment and installs the required packages for running PyTorch benchmarks with Intel extensions and Triton
# It requires the user to provide a value for ENV_NAME, which is the name of the conda environment to create
# It also clones the PyTorch, IPEX, and Triton projects from specific branches, and installs them from source

# Usage: You need to set ENV_NAME explicitly BEFORE calling the script
#   ENV_NAME='triton' bash build_env.sh

# exit if any command fails
set -e

BASE=$(cd $(dirname "$0")/.. && pwd)
TRITON_PROJ=$BASE/intel-xpu-backend-for-triton
PYTORCH_PROJ=$BASE/pytorch
IPEX_PROJ=$BASE/ipex

ENV_NAME=${ENV_NAME:-}

if [ -z "$ENV_NAME" ]; then
  echo "Please provide a value for ENV_NAME"
  exit 1
fi

echo "==============================="
echo "ENV_NAME: ${ENV_NAME}"
echo "BASE : ${BASE}"
echo "PYTORCH_PROJ : ${PYTORCH_PROJ}"
echo "IPEX_PROJ : ${IPEX_PROJ}"
echo "TRITON_PROJ : ${TRITON_PROJ}"
echo "==============================="

eval "$(conda shell.bash hook)"
conda create --name ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}

if [ ! -d "$PYTORCH_PROJ" ]; then
  cd $BASE
  git clone https://github.com/Stonepia/pytorch.git -b dev/triton-test-3.0
fi
cd $PYTORCH_PROJ
conda install cmake ninja mkl mkl-include -y
conda install -c conda-forge libstdcxx-ng -y
pip install pyyaml
pip install -r requirements.txt
git submodule sync && git submodule update --init --recursive --jobs 0
python setup.py develop

python -c "import torch;print(f'torch version {torch.__version__}')"

if [ ! -d "$IPEX_PROJ" ]; then
  cd $BASE
  git clone https://github.com/intel/intel-extension-for-pytorch.git -b dev/triton-test-3.0 ipex
fi
cd $IPEX_PROJ
git submodule sync && git submodule update --init --recursive --jobs 0
source $BASE/env.sh
pip install -r requirements.txt
python setup.py develop
python -c "import intel_extension_for_pytorch as ipex;print(f'ipex version {ipex.__version__}')"

if [ ! -d "$TRITON_PROJ" ]; then
  cd $BASE
  git clone https://github.com/intel/intel-xpu-backend-for-triton.git -b llvm-target
fi
cd $TRITON_PROJ
scripts/compile-triton.sh
python -c "import triton;print(f'triton version {triton.__version__}')"
