#!/bin/bash

# Run script for PyTorch DAE training
# Usage: ./run_train_dae_pytorch.sh [OPTIONS]

set -e

echo "=== DSMNet PyTorch DAE Training Script ==="
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
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-0.0001}
MTL_CHECKPOINT=${MTL_CHECKPOINT:-"checkpoints/mtl_model_best.pth"}

echo ""
echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  MTL checkpoint: $MTL_CHECKPOINT"
echo ""

# Check if MTL checkpoint exists
if [ ! -f "$MTL_CHECKPOINT" ]; then
    echo "Error: MTL checkpoint not found at $MTL_CHECKPOINT"
    echo "Please train the MTL model first using run_train_mtl.sh"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p checkpoints
mkdir -p output
mkdir -p plots

# Run DAE training
echo "Running DAE training..."
python train_dae.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --mtl_checkpoint $MTL_CHECKPOINT

echo ""
echo "DAE training completed! Check the checkpoints/ directory for saved models."
