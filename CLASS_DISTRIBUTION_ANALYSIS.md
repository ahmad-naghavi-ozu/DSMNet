# Class Distribution Analysis - DFC2019_crp512_bin Dataset

**Date**: November 6, 2025  
**Branch**: `investigate-dfc19-dataset`

## Summary

Comprehensive analysis of ground truth class distribution in DFC2019_crp512_bin dataset to determine appropriate class weighting for semantic segmentation loss.

## Dataset Statistics

### DFC2019_crp512_bin_mini (Debug/Small Dataset)
- **Training**: 140 files (36.7M pixels)
  - Background (class 0): 51.06% (18.7M pixels)
  - Buildings (class 1): 48.94% (18.0M pixels)
  - **Imbalance ratio**: 1.04:1 (nearly balanced)
  
- **Validation**: 20 files (5.2M pixels)
  - Background (class 0): 49.37% (2.6M pixels)
  - Buildings (class 1): 50.63% (2.7M pixels)
  - **Imbalance ratio**: 0.97:1 (nearly balanced)

### DFC2019_crp512_bin (Full Dataset) ✅
- **Training**: 14,000 files (3.67B pixels)
  - Background (class 0): 78.62% (2.89B pixels)
  - Buildings (class 1): 21.38% (785M pixels)
  - **Imbalance ratio**: 3.68:1 (moderate imbalance)
  
- **Validation**: 2,000 files (524M pixels)
  - Background (class 0): 80.15% (420M pixels)
  - Buildings (class 1): 19.85% (104M pixels)
  - **Imbalance ratio**: 4.04:1 (moderate imbalance)

## Class Weighting Strategy

### Inverse Frequency Weighting
For imbalanced datasets, we use inverse frequency weighting:

```
weight_class_i = total_pixels / (num_classes × count_class_i)
```

For DFC2019_crp512_bin training set:
- Background weight: 3.67B / (2 × 2.89B) = **0.64x** (down-weighted)
- Building weight: 3.67B / (2 × 785M) = **2.34x** (up-weighted)

### Normalized Weights (Used in Implementation)
To keep background at baseline (1.0x):
- Background: **1.0x**
- Buildings: **2.34x**

This means building misclassifications contribute 2.34× more to the loss than background misclassifications, compensating for the 3.68:1 class imbalance.

## Implementation Changes

### config.py
```python
# Changed from mini to full dataset
dataset_name = 'DFC2019_crp512_bin'  # was: 'DFC2019_crp512_bin_mini'

# Reset MTL loss weights to 1:1:1 (class weighting handles semantic loss internally)
w1, w2, w3 = (1, 1, 1)  # weights for: dsm, sem, norm
```

### train_mtl.py
Added class-weighted categorical cross entropy in both distributed and single-GPU training paths:

```python
# Class-weighted categorical cross entropy for imbalanced building detection
# DFC2019_crp512_bin: Background 78.6%, Buildings 21.4% → 3.68:1 imbalance
# Using inverse frequency weighting: buildings get 2.34x weight
class_weights = tf.constant([1.0, 2.34], dtype=tf.float32)  # [background, building]

# Get true class indices and create weight mask based on ground truth
true_classes = tf.argmax(sem_batch, axis=-1, output_type=tf.int32)
weight_mask = tf.gather(class_weights, true_classes)

# Apply weights to per-sample losses (weight based on true class)
L2_per_sample = CCE(tf.cast(sem_batch, tf.float32), sem_out) * weight_mask
L2 = tf.reduce_sum(L2_per_sample) * (1.0 / global_batch_size)  # distributed
# or L2 = tf.reduce_mean(L2_per_sample)  # single-GPU
```

## Key Findings

1. **Mini vs Full Dataset**: The mini dataset is nearly balanced (1:1), while the full dataset has 3.68:1 imbalance favoring background
2. **Previous Error**: Initial assumption of 87.6%/12.4% split was incorrect - actual full dataset is 78.6%/21.4%
3. **Weight Choice**: Using 2.34x for buildings (inverse frequency) instead of arbitrary 7x
4. **Type Safety**: Added `tf.cast(sem_batch, tf.float32)` to prevent dtype mismatches

## Expected Improvements

With proper class weighting:
- Building IoU should improve (currently underrepresented class gets more attention)
- Training should converge more stably (balanced gradient contributions)
- False negatives on buildings should decrease (higher penalty for missing buildings)

## Testing Plan

1. ✅ Analyze ground truth class distribution
2. ✅ Implement inverse frequency weighting
3. ✅ Update dataset configuration
4. ⏳ Run full training with weighted loss
5. ⏳ Compare semantic metrics (IoU, F1) before/after
6. ⏳ Evaluate if further adjustments needed (e.g., focal loss for hard examples)

## References

- Analysis script: `analyze_class_distribution.py`
- Test script: `test_weighted_loss.py`
- Dataset path: `../../datasets/DFC2019_crp512_bin/`
