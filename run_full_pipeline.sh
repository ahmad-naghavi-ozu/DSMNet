#!/bin/bash

# Complete PyTorcecho "Step 1: Training MTL model..."
echo "==============================="
EPOCHS=$MTL_EPOCHS BATCH_SIZE=$BATCH_SIZE LEARNING_RATE=$LEARNING_RATE ./run_train_mtl.sh

echo ""
echo "Step 2: Training DAE model..."
echo "============================="
EPOCHS=$DAE_EPOCHS BATCH_SIZE=$BATCH_SIZE LEARNING_RATE=$LEARNING_RATE ./run_train_dae.sh

echo ""
echo "Step 3: Testing complete pipeline..."
echo "===================================="
./run_test.shd testing pipeline
# Usage: ./run_full_pipeline_pytorch.sh

set -e

echo "=== DSMNet PyTorch Full Pipeline ==="
echo "This script will run the complete DSMNet training and testing pipeline"
echo "Current directory: $(pwd)"
echo ""

# Set default parameters
MTL_EPOCHS=${MTL_EPOCHS:-100}
DAE_EPOCHS=${DAE_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-8}
LEARNING_RATE=${LEARNING_RATE:-0.001}

echo "Pipeline parameters:"
echo "  MTL epochs: $MTL_EPOCHS"
echo "  DAE epochs: $DAE_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo ""

# Create output directories
mkdir -p checkpoints
mkdir -p output
mkdir -p plots

echo "Step 1: Training MTL model..."
echo "==============================="
EPOCHS=$MTL_EPOCHS BATCH_SIZE=$BATCH_SIZE LEARNING_RATE=$LEARNING_RATE ./run_train_mtl_pytorch.sh

echo ""
echo "Step 2: Training DAE model..."
echo "============================="
EPOCHS=$DAE_EPOCHS BATCH_SIZE=$BATCH_SIZE LEARNING_RATE=$LEARNING_RATE ./run_train_dae_pytorch.sh

echo ""
echo "Step 3: Testing complete pipeline..."
echo "===================================="
BATCH_SIZE=$BATCH_SIZE ./run_test_pytorch.sh

echo ""
echo "=== Full Pipeline Completed Successfully ==="
echo "Results saved in:"
echo "  - Model checkpoints: checkpoints/"
echo "  - Test outputs: output/"
echo "  - Plots and visualizations: plots/"
