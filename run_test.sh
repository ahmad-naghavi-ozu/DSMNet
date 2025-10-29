#!/bin/bash

# To run the script, execute the following command: 
# chmod +x run_test.sh && ./run_test.sh

echo "Starting testing with correction=False (MTL mode)..."
CORRECTION=False python test.py
if [ $? -ne 0 ]; then
    echo "Testing with correction=False failed"
    exit 1
fi

echo "Waiting 15 seconds for GPU cooldown..."
sleep 15

echo -e "\nStarting testing with correction=True (DAE mode)..."
CORRECTION=True python test.py
if [ $? -ne 0 ]; then
    echo "Testing with correction=True failed"
    exit 1
fi

echo "All testing operations completed successfully"