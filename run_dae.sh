#!/bin/bash

# DAE Training Script
# To run: chmod +x train_dae.sh && ./train_dae.sh

echo "=== DAE Training ==="
echo "Starting DAE training with correction = True..."
CORRECTION=True python train_dae.py

if [ $? -eq 0 ]; then
    echo "DAE training completed successfully"
else
    echo "DAE training failed"
    exit 1
fi
