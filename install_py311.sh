#!/bin/bash

# Install PyTorch dependencies for DSMNet with Python 3.11
# This script creates a conda environment with Python 3.11 and installs PyTorch with CUDA support

set -e

echo "=== Setting up DSMNet PyTorch environment with Python 3.11 ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://www.anaconda.com/products/distribution"
    exit 1
fi

# Environment name
ENV_NAME="dsmnet_pytorch_py311"

echo "🔍 Checking for existing environment..."

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "❌ Aborting installation."
        exit 1
    fi
fi

echo "🚀 Creating conda environment '${ENV_NAME}' with Python 3.11..."
conda create -n ${ENV_NAME} python=3.11 -y

echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "🎯 Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA 11.8 support for Python 3.11
# Using conda-forge channel for better compatibility
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "📦 Installing scientific computing packages..."
conda install numpy scipy scikit-learn matplotlib pandas opencv pillow -c conda-forge -y

echo "🔧 Installing additional packages with pip..."
pip install scikit-image tqdm

echo "✅ Verifying installation..."
python -c "
import sys
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import skimage

print(f'✓ Python version: {sys.version}')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Torchvision version: {torchvision.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA device count: {torch.cuda.device_count()}')
    print(f'✓ Current CUDA device: {torch.cuda.current_device()}')
    print(f'✓ CUDA device name: {torch.cuda.get_device_name(0)}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ OpenCV version: {cv2.__version__}')
print(f'✓ Scikit-image version: {skimage.__version__}')
print('✓ All dependencies installed successfully!')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 To use the DSMNet PyTorch implementation:"
echo "   1. Activate the environment: conda activate ${ENV_NAME}"
echo "   2. Run the setup test: python test_setup.py"
echo "   3. Configure your dataset paths in config.py"
echo "   4. Start training: ./run_train_mtl.sh"
echo ""
echo "🔧 Environment created: ${ENV_NAME}"
echo "💡 To deactivate the environment later: conda deactivate"
