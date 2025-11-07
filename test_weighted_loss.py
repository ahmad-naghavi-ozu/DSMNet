#!/usr/bin/env python3
"""
Test class-weighted categorical cross entropy
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

def test_weighted_loss():
    print("Testing class-weighted categorical cross entropy...")

    # Create test data: 4 pixels, 2 classes [background=0, building=1]
    batch_size, height, width = 1, 2, 2
    num_classes = 2

    # Ground truth: 3 background (class 0), 1 building (class 1)
    sem_batch = np.zeros((batch_size, height, width, num_classes))
    sem_batch[0, 0, 0, 0] = 1  # background
    sem_batch[0, 0, 1, 0] = 1  # background
    sem_batch[0, 1, 0, 0] = 1  # background
    sem_batch[0, 1, 1, 1] = 1  # building

    # Model predictions: slightly wrong predictions
    sem_pred = np.zeros((batch_size, height, width, num_classes))
    # Predict background with 0.6 confidence, building with 0.4
    sem_pred[0, :, :, 0] = 0.6  # predict background
    sem_pred[0, :, :, 1] = 0.4  # predict building

    print("Ground truth classes:")
    print(np.argmax(sem_batch[0], axis=-1))
    print("Predictions (background prob, building prob):")
    print(sem_pred[0])

    # Standard CCE
    CCE = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    standard_loss = CCE(sem_batch.astype(np.float32), sem_pred)
    print(f"\nStandard CCE loss per pixel: {standard_loss.numpy().flatten()}")

    # Weighted CCE (7x weight for buildings)
    class_weights = tf.constant([1.0, 7.0], dtype=tf.float32)
    true_classes = tf.argmax(sem_batch, axis=-1, output_type=tf.int32)
    weight_mask = tf.gather(class_weights, true_classes)
    # Ensure both tensors are same type
    standard_loss = tf.cast(standard_loss, tf.float32)
    weight_mask = tf.cast(weight_mask, tf.float32)
    weighted_loss = standard_loss * weight_mask
    print(f"Weight mask (based on true classes): {weight_mask.numpy().flatten()}")
    print(f"Weighted CCE loss per pixel: {weighted_loss.numpy().flatten()}")

    print(f"\nStandard total loss: {tf.reduce_sum(standard_loss):.4f}")
    print(f"Weighted total loss: {tf.reduce_sum(weighted_loss):.4f}")
    print(f"Weight factor: {tf.reduce_sum(weighted_loss) / tf.reduce_sum(standard_loss):.2f}x")

if __name__ == "__main__":
    test_weighted_loss()