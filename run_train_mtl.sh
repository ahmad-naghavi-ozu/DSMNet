#!/bin/bash

# Run script for PyTorch MTL training with multi-GPU support
# Usage: ./run_train_mtl_pytorch.sh [OPTIONS]

set -e

echo "=== DSMNet PyTorch MTL Training Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check if CUDA is available
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo "GPU devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
else
    echo "Warning: CUDA not available, running on CPU"
fi

# Set default parameters
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-0.001}
USE_MULTI_GPU=${USE_MULTI_GPU:-true}
DISTRIBUTED=${DISTRIBUTED:-false}

echo ""
echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Multi-GPU: $USE_MULTI_GPU"
echo "  Distributed: $DISTRIBUTED"
echo ""

# Create output directories if they don't exist
mkdir -p checkpoints
mkdir -p output
mkdir -p plots

# Run training
if [ "$DISTRIBUTED" = true ]; then
    echo "Running distributed training..."
    python -m torch.distributed.launch --nproc_per_node=2 train_mtl.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --distributed
else
    echo "Running single/multi-GPU training..."
    python train_mtl.py \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE
fi

echo ""
echo "Training completed! Check the checkpoints/ directory for saved models."
