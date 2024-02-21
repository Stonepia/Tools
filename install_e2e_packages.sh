#!/bin/bash

# This script sets up the environment and installs the dependencies for running PyTorch benchmarks
# It requires the user to provide a value for ENV_NAME, which is the name of the conda environment to use
# It also allows the user to optionally set some flags to enable or disable the installation of certain dependencies
# The dependencies are installed from specific pinned commits, which can be overridden by the user as well

# Usage:  You need to set ENV_NAME explicitly BEFORE calling the script
#               ENV_NAME='triton'
#               [HUGGINGFACE=true|false] 
#               [TIMM_MODELS=true|false] 
#               [TORCHBENCH=true|false] 
#               [HUGGINGFACE_PIN_COMMIT=commit_hash] 
#               [TIMM_MODELS_PIN_COMMIT=commit_hash] 
#               [TORCH_VISION_PIN_COMMIT=commit_hash] 
#               [TORCH_AUDIO_PIN_COMMIT=commit_hash] 
#               [TORCH_TEXT_PIN_COMMIT=commit_hash] 
#               [TORCH_DATA_PIN_COMMIT=commit_hash] 
#               [TORCH_BENCH_PIN_COMMIT=commit_hash]
#               bash install_e2e_packages.sh

# Exit if any command fails
set -e
set -u
set -o pipefail

# Declare and assign variables
HUGGINGFACE=${HUGGINGFACE:-true}
TIMM_MODELS=${TIMM_MODELS:-true}
TORCHBENCH=${TORCHBENCH:-true}
ENV_NAME=${ENV_NAME:-}

if [ -z "$ENV_NAME" ]; then
  echo "Please provide a value for ENV_NAME"
  exit 1
fi

BASE=$(cd $(dirname "$0")/.. && pwd)
echo Base is set as ${BASE}

PYTORCH_PROJ=$BASE/pytorch
DEPS_FOLDER=$BASE/deps

if [ ! -d "${DEPS_FOLDER}" ]; then
  mkdir ${DEPS_FOLDER}
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Define a function to get the pinned commit for a given project
function get_pinned_commit() {
  # Check if PyTorch project folder exists
  if [[ -d "$PYTORCH_PROJ" ]]; then
    # huggingface and timm are in different folder
    if [[ ${1} == "huggingface" ]] || [[ ${1} == "timm" ]]; then
      cat "${PYTORCH_PROJ}"/.ci/docker/ci_commit_pins/"${1}".txt
    else
      cat "${PYTORCH_PROJ}"/.github/ci_commit_pins/"${1}".txt
    fi
  else
    # Print a warning message and return empty string
    printf 'WARNING: PyTorch Source folder does not exist, the commit is not pinned!\\n' >&2
    echo ""
  fi
}

# Set the pinned commit variables with default values
HUGGINGFACE_PIN_COMMIT=${HUGGINGFACE_PIN_COMMIT:-$(get_pinned_commit huggingface)}
TIMM_MODELS_PIN_COMMIT=${TIMM_MODELS_PIN_COMMIT:-$(get_pinned_commit timm)}

TORCH_VISION_PIN_COMMIT=${TORCH_VISION_PIN_COMMIT:-$(get_pinned_commit vision)}
TORCH_AUDIO_PIN_COMMIT=${TORCH_AUDIO_PIN_COMMIT:-$(get_pinned_commit audio)}
TORCH_TEXT_PIN_COMMIT=${TORCH_TEXT_PIN_COMMIT:-$(get_pinned_commit text)}
TORCH_DATA_PIN_COMMIT=${TORCH_DATA_PIN_COMMIT:-$(get_pinned_commit data)}
TORCH_BENCH_PIN_COMMIT=${TORCH_BENCH_PIN_COMMIT:-$(get_pinned_commit torchbench)}

# Print the pinned commit variables
echo "HUGGINGFACE_PIN_COMMIT: $HUGGINGFACE_PIN_COMMIT"
echo "TIMM_MODELS_PIN_COMMIT: $TIMM_MODELS_PIN_COMMIT"
echo "TORCH_VISION_PIN_COMMIT: $TORCH_VISION_PIN_COMMIT"
echo "TORCH_AUDIO_PIN_COMMIT: $TORCH_AUDIO_PIN_COMMIT"
echo "TORCH_TEXT_PIN_COMMIT: $TORCH_TEXT_PIN_COMMIT"
echo "TORCH_DATA_PIN_COMMIT: $TORCH_DATA_PIN_COMMIT"
echo "TORCH_BENCH_PIN_COMMIT: $TORCH_BENCH_PIN_COMMIT"

# Install the dependencies

# HUGGINGFACE
if [ "${HUGGINGFACE}" = "true" ]; then
  pip install pandas
  pip install transformers==${HUGGINGFACE_PIN_COMMIT}
  # TODO : Use the new huggingface
  # pip install "git+https://github.com/huggingface/transformers@${HUGGINGFACE_PIN_COMMIT}"
fi

# TIMM Models

if [ "${TIMM_MODELS}" = "true" ]; then
  pip install pandas
  # TODO : Use new timm repo once we are upgraded. For now we use the old timm repos.
  # Uncomment the new timm repos
  #pip install "git+https://github.com/huggingface/pytorch-image-models@${TIMM_MODELS_PIN_COMMIT}"
  #TODO : For the old TIMM, it automatically install the cuda version of PyTorch. Thus we install without deps.
  pip install "git+https://github.com/rwightman/pytorch-image-models@${TIMM_MODELS_PIN_COMMIT}" --no-deps
fi

# TorchBench

if [ "${TORCHBENCH}" = "true" ]; then

  cd ${DEPS_FOLDER}

  # TorchData
  if [ ! -d "${DEPS_FOLDER}/data" ]; then
    git clone --recursive https://github.com/pytorch/data.git
  fi
  cd data
  git checkout ${TORCH_DATA_PIN_COMMIT}
  pip install .
  cd ..

  # Torchvision
  if [ ! -d "${DEPS_FOLDER}/vision" ]; then
    git clone --recursive https://github.com/pytorch/vision.git
  fi
  cd vision
  git checkout ${TORCH_VISION_PIN_COMMIT}
  conda install -y libpng jpeg
  # TODO: We use an older version ffmpeg to avoid the vision capability issue.
  conda install -y -c conda-forge 'ffmpeg<4.4'
  python setup.py install
  cd ..

  # Torchtext
  if [ ! -d "${DEPS_FOLDER}/text" ]; then
    git clone --recursive https://github.com/pytorch/text.git
  fi
  cd text
  git checkout ${TORCH_TEXT_PIN_COMMIT}
  python setup.py clean install
  cd ..

  # Torch audio
  if [ ! -d "${DEPS_FOLDER}/audio" ]; then
    git clone --recursive https://github.com/pytorch/audio.git
  fi

  cd audio
  # Optionally `git checkout {pinned_commit}`
  git checkout ${TORCH_AUDIO_PIN_COMMIT}
  python setup.py install
  cd ..

  # Check first
  python -c "import torchvision,torchtext,torchaudio;print(torchvision.__version__, torchtext.__version__, torchaudio.__version__)"

  # Torchbench
  conda install -y git-lfs pyyaml pandas scipy psutil
  pip install pyre_extensions
  pip install torchrec
  # TODO : We use a temporary private repo. Thus we don't checkout commit.
  if [ ! -d "${DEPS_FOLDER}/benchmark" ]; then
    git clone --recursive https://github.com/weishi-deng/benchmark
  fi

  # git checkout ${TORCH_BENCH_PIN_COMMIT}
  cd benchmark
  pip install -e .
fi
