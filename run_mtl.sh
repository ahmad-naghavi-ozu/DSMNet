#!/bin/bash

# MTL Training Script
# To run: chmod +x train_mtl.sh && ./train_mtl.sh

echo "=== MTL Training ==="
echo "Setting correction = False for MTL training..."
sed -i 's/correction = .*/correction = False/' config.py

echo "Starting MTL training..."
python train_mtl.py

if [ $? -eq 0 ]; then
    echo "MTL training completed successfully"
else
    echo "MTL training failed"
    exit 1
fi
