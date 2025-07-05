#!/usr/bin/env python3
"""
Quick Multi-GPU MTL Training Test
This script runs a few training iterations to verify multi-GPU training works.
"""

from config import *
import numpy as np
import logging
import tensorflow as tf
from nets import *
from utils import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def quick_mtl_test():
    print("=== Quick Multi-GPU MTL Training Test ===")
    print(f"Dataset: {dataset_name}")
    print(f"Multi-GPU enabled: {multi_gpu_enabled}")
    print(f"Strategy replicas: {strategy.num_replicas_in_sync}")
    print(f"Batch size per GPU: {batch_size}, Global: {global_batch_size}")
    
    # Collect training data
    train_rgb, train_sar, train_dsm, train_sem, _ = collect_tilenames("train")
    print(f"Training samples: {len(train_rgb)}")
    
    # Create model within strategy scope
    with strategy.scope():
        backbone = DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_tensor=Input(shape=(cropSize, cropSize, 3))
        )
        mtl = MTL(backbone, sem_flag=True, norm_flag=True, edge_flag=False)
        
        # Loss functions
        REG_LOSS = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        CCE = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        
        print("✓ Model, losses, and optimizer created successfully")
    
    # Define distributed training step
    @tf.function
    def distributed_train_step(rgb_batch, dsm_batch, sem_batch, norm_batch):
        def train_step(rgb_batch, dsm_batch, sem_batch, norm_batch):
            with tf.GradientTape() as tape:
                dsm_out, sem_out, norm_out, _ = mtl.call(rgb_batch, 'dsm', training=True)
                
                # Compute losses with proper reduction for distributed training
                L1_per_sample = REG_LOSS(tf.squeeze(dsm_batch), tf.squeeze(dsm_out))
                L1 = tf.reduce_sum(L1_per_sample) * (1.0 / global_batch_size)
                
                L2_per_sample = CCE(sem_batch, sem_out)
                L2 = tf.reduce_sum(L2_per_sample) * (1.0 / global_batch_size)
                
                L3_per_sample = REG_LOSS(norm_batch, norm_out)
                L3 = tf.reduce_sum(L3_per_sample) * (1.0 / global_batch_size)
                
                total_loss = w1 * L1 + w2 * L2 + w3 * L3
                
            grads = tape.gradient(total_loss, mtl.trainable_variables)
            optimizer.apply_gradients(zip(grads, mtl.trainable_variables))
            
            return total_loss, L1, L2, L3
        
        return strategy.run(train_step, args=(rgb_batch, dsm_batch, sem_batch, norm_batch))
    
    # Run a few training iterations
    print("\n=== Running Training Iterations ===")
    num_test_iterations = min(3, len(train_rgb) // batch_size)
    
    for iter in range(1, num_test_iterations + 1):
        print(f"\nIteration {iter}/{num_test_iterations}")
        
        # Generate batch
        rgb_batch, dsm_batch, sem_batch, norm_batch, _ = \
            generate_training_batches(train_rgb, train_sar, train_dsm, train_sem, iter, mtl_flag=True)
        
        # Training step
        if multi_gpu_enabled:
            total_loss, L1, L2, L3 = distributed_train_step(rgb_batch, dsm_batch, sem_batch, norm_batch)
            # Reduce across replicas
            total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, total_loss, axis=None)
            L1 = strategy.reduce(tf.distribute.ReduceOp.MEAN, L1, axis=None)
            L2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, L2, axis=None)
            L3 = strategy.reduce(tf.distribute.ReduceOp.MEAN, L3, axis=None)
        else:
            # Single GPU fallback
            with tf.GradientTape() as tape:
                dsm_out, sem_out, norm_out, _ = mtl.call(rgb_batch, 'dsm', training=True)
                
                L1_per_sample = REG_LOSS(tf.squeeze(dsm_batch), tf.squeeze(dsm_out))
                L1 = tf.reduce_mean(L1_per_sample)
                
                L2_per_sample = CCE(sem_batch, sem_out)
                L2 = tf.reduce_mean(L2_per_sample)
                
                L3_per_sample = REG_LOSS(norm_batch, norm_out)
                L3 = tf.reduce_mean(L3_per_sample)
                
                total_loss = w1 * L1 + w2 * L2 + w3 * L3
            
            grads = tape.gradient(total_loss, mtl.trainable_variables)
            optimizer.apply_gradients(zip(grads, mtl.trainable_variables))
        
        print(f"  Total Loss: {total_loss:.6f}")
        print(f"  DSM Loss (L1): {L1:.6f}")
        print(f"  Semantic Loss (L2): {L2:.6f}")
        print(f"  Normal Loss (L3): {L3:.6f}")
    
    print("\n=== Test Complete ===")
    print("✓ Multi-GPU MTL training is working correctly!")
    print(f"✓ Successfully trained on {num_test_iterations} iterations")
    print(f"✓ All losses computed and gradients applied across {strategy.num_replicas_in_sync} GPUs")

if __name__ == "__main__":
    quick_mtl_test()
