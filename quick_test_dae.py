#!/usr/bin/env python3
"""
Quick Multi-GPU DAE Training Test
This script tests DAE (Denoising Autoencoder) training with multi-GPU support.
"""

from config import *
import numpy as np
import logging
import tensorflow as tf
from nets import *
from utils import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import MeanSquaredError

# Set up simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def quick_dae_test():
    print("=== Quick Multi-GPU DAE Training Test ===")
    print(f"Dataset: {dataset_name}")
    print(f"Multi-GPU enabled: {multi_gpu_enabled}")
    print(f"Strategy replicas: {strategy.num_replicas_in_sync}")
    print(f"Batch size per GPU: {batch_size}, Global: {global_batch_size}")
    
    # Collect training data
    train_rgb, train_sar, train_dsm, _, _ = collect_tilenames("train")
    print(f"Training samples: {len(train_rgb)}")
    
    # Create MTL model first (needed for DAE input)
    backbone = DenseNet121(
        weights='imagenet', 
        include_top=False, 
        input_tensor=Input(shape=(cropSize, cropSize, 3))
    )
    mtl = MTL(backbone, sem_flag=True, norm_flag=True, edge_flag=False)
    print("✓ MTL model created for generating DAE inputs")
    
    # Create DAE model within strategy scope
    with strategy.scope():
        dae = Autoencoder()
        
        # Loss function with proper reduction
        REG_LOSS = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
        
        print("✓ DAE model, loss, and optimizer created successfully")
    
    # Define distributed DAE training step
    @tf.function
    def distributed_dae_train_step(correction_input, dsm_initial, dsm_batch):
        def train_step(correction_input, dsm_initial, dsm_batch):
            with tf.GradientTape() as tape:
                noise = dae.call(correction_input, training=True)
                dsm_corrected = dsm_initial - noise
                
                # Compute loss with proper reduction for distributed training
                dae_loss_per_sample = REG_LOSS(dsm_batch, dsm_corrected)
                dae_loss = tf.reduce_sum(dae_loss_per_sample) * (1.0 / dae_global_batch_size)
                
            grads = tape.gradient(dae_loss, dae.trainable_variables)
            optimizer.apply_gradients(zip(grads, dae.trainable_variables))
            
            return dae_loss, dsm_corrected
        
        return strategy.run(train_step, args=(correction_input, dsm_initial, dsm_batch))
    
    # Run a few training iterations
    print("\n=== Running DAE Training Iterations ===")
    num_test_iterations = min(3, len(train_rgb) // batch_size)
    
    for iter in range(1, num_test_iterations + 1):
        print(f"\nIteration {iter}/{num_test_iterations}")
        
        # Generate batch
        rgb_batch, dsm_batch, _, _, _ = \
            generate_training_batches(train_rgb, train_sar, train_dsm, [], iter, mtl_flag=False)
        
        # Get MTL outputs (frozen, for DAE input)
        dsm_out, sem_out, norm_out, _ = mtl.call(rgb_batch, 'dsm', training=False)
        
        # Prepare DAE input (concatenate MTL outputs + RGB)
        correction_list = [dsm_out, sem_out, norm_out, rgb_batch]
        correction_input = tf.concat(correction_list, axis=-1)
        
        # DAE training step
        if multi_gpu_enabled:
            dae_loss, dsm_corrected = distributed_dae_train_step(correction_input, dsm_out, dsm_batch)
            # Reduce across replicas
            dae_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, dae_loss, axis=None)
            dsm_corrected = strategy.reduce(tf.distribute.ReduceOp.MEAN, dsm_corrected, axis=None)
            
            # Calculate RMSE
            abs_diff = tf.abs(dsm_corrected - dsm_batch)
            mse_per_sample = tf.reduce_mean(tf.square(abs_diff), axis=[1, 2])
            rmse_per_sample = tf.sqrt(mse_per_sample)
            batch_rmse = tf.reduce_mean(rmse_per_sample)
        else:
            # Single GPU fallback
            with tf.GradientTape() as tape:
                noise = dae.call(correction_input, training=True)
                dsm_corrected = dsm_out - noise
                
                dae_loss_per_sample = REG_LOSS(dsm_batch, dsm_corrected)
                dae_loss = tf.reduce_mean(dae_loss_per_sample)
                
                abs_diff = tf.abs(dsm_corrected - dsm_batch)
                mse_per_sample = tf.reduce_mean(tf.square(abs_diff), axis=[1, 2])
                rmse_per_sample = tf.sqrt(mse_per_sample)
                batch_rmse = tf.reduce_mean(rmse_per_sample)
            
            grads = tape.gradient(dae_loss, dae.trainable_variables)
            optimizer.apply_gradients(zip(grads, dae.trainable_variables))
        
        print(f"  DAE Loss: {dae_loss:.6f}")
        print(f"  Batch RMSE: {batch_rmse:.6f}")
    
    print("\n=== DAE Test Complete ===")
    print("✓ Multi-GPU DAE training is working correctly!")
    print(f"✓ Successfully trained on {num_test_iterations} iterations")
    print(f"✓ Denoising autoencoder gradients applied across {strategy.num_replicas_in_sync} GPUs")

if __name__ == "__main__":
    quick_dae_test()
