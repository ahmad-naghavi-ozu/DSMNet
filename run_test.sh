#!/bin/bash

# Run script for PyTorch DSM testing and evaluation
# Usage: ./run_test_pytorch.sh [OPTIONS]

set -e

echo "=== DSMNet PyTorch Testing Script ==="
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
MTL_CHECKPOINT=${MTL_CHECKPOINT:-"checkpoints/mtl_model_best.pth"}
DAE_CHECKPOINT=${DAE_CHECKPOINT:-"checkpoints/dae_model_best.pth"}
TEST_STAGE=${TEST_STAGE:-"both"}  # Options: mtl, dae, both
BATCH_SIZE=${BATCH_SIZE:-4}
SAVE_OUTPUTS=${SAVE_OUTPUTS:-true}

echo ""
echo "Testing parameters:"
echo "  MTL checkpoint: $MTL_CHECKPOINT"
echo "  DAE checkpoint: $DAE_CHECKPOINT"
echo "  Test stage: $TEST_STAGE"
echo "  Batch size: $BATCH_SIZE"
echo "  Save outputs: $SAVE_OUTPUTS"
echo ""

# Check checkpoints based on test stage
if [ "$TEST_STAGE" = "mtl" ] || [ "$TEST_STAGE" = "both" ]; then
    if [ ! -f "$MTL_CHECKPOINT" ]; then
        echo "Error: MTL checkpoint not found at $MTL_CHECKPOINT"
        echo "Please train the MTL model first using run_train_mtl_pytorch.sh"
        exit 1
    fi
fi

if [ "$TEST_STAGE" = "dae" ] || [ "$TEST_STAGE" = "both" ]; then
    if [ ! -f "$DAE_CHECKPOINT" ]; then
        echo "Error: DAE checkpoint not found at $DAE_CHECKPOINT"
        echo "Please train the DAE model first using run_train_dae.sh"
        exit 1
    fi
fi

# Create output directories if they don't exist
mkdir -p output
mkdir -p plots

# Run testing
echo "Running DSM testing and evaluation..."
python test_dsm.py \
    --mtl_checkpoint $MTL_CHECKPOINT \
    --dae_checkpoint $DAE_CHECKPOINT \
    --test_stage $TEST_STAGE \
    --batch_size $BATCH_SIZE \
    $([ "$SAVE_OUTPUTS" = true ] && echo "--save_outputs")

echo ""
echo "Testing completed! Check the output/ and plots/ directories for results."
