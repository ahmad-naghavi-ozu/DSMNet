#!/bin/bash

# DAE Training Script
# To run: chmod +x train_dae.sh && ./train_dae.sh

echo "=== DAE Training ==="
echo "Setting correction = True for DAE training..."
sed -i 's/correction = .*/correction = True/' config.py

echo "Starting DAE training..."
python train_dae.py

if [ $? -eq 0 ]; then
    echo "DAE training completed successfully"
else
    echo "DAE training failed"
    exit 1
fi
