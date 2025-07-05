#!/usr/bin/env python3
"""
Multi-GPU Test Script for DSMNet
This script tests if the multi-GPU configuration is working correctly.
"""

from config import *
import tensorflow as tf

def test_multi_gpu_setup():
    print("=== DSMNet Multi-GPU Configuration Test ===")
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Check visible devices (after CUDA_VISIBLE_DEVICES)
    visible_gpus = tf.config.list_logical_devices('GPU')
    print(f"Visible GPUs: {len(visible_gpus)}")
    for i, gpu in enumerate(visible_gpus):
        print(f"  Visible GPU {i}: {gpu}")
    
    # Check strategy configuration
    print(f"\nMulti-GPU enabled: {multi_gpu_enabled}")
    print(f"GPU devices setting: {gpu_devices}")
    print(f"Strategy type: {type(strategy).__name__}")
    print(f"Number of replicas in sync: {strategy.num_replicas_in_sync}")
    
    # Check batch size adjustments
    print(f"\nBatch size adjustments:")
    print(f"Original batch_size: {batch_size}")
    print(f"Global batch_size: {global_batch_size}")
    print(f"MTL global batch_size: {mtl_global_batch_size}")
    print(f"DAE global batch_size: {dae_global_batch_size}")
    
    # Test simple model creation within strategy scope
    print(f"\nTesting model creation within strategy scope...")
    try:
        with strategy.scope():
            # Create a simple test model
            test_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(1)
            ])
            test_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            print("✓ Model and optimizer creation successful")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print(f"\n=== Configuration Test Complete ===")
    
    if multi_gpu_enabled and strategy.num_replicas_in_sync > 1:
        print("✓ Multi-GPU configuration is active and ready!")
        return True
    elif not multi_gpu_enabled:
        print("✓ Single-GPU configuration is active and ready!")
        return True
    else:
        print("⚠ Multi-GPU enabled but only 1 replica detected. Check GPU availability.")
        return False

if __name__ == "__main__":
    success = test_multi_gpu_setup()
    exit(0 if success else 1)
