#!/usr/bin/env python3
"""
Analyze class distribution in DFC2019_crp512_bin dataset
"""

import os
import numpy as np
from glob import glob
import tifffile

def analyze_dataset(dataset_path, split='train'):
    """Analyze class distribution in semantic labels"""
    sem_path = os.path.join(dataset_path, split, 'sem')
    sem_files = sorted(glob(os.path.join(sem_path, '*.tif')))
    
    print(f"\n{'='*60}")
    print(f"Analyzing {split} split: {len(sem_files)} files")
    print(f"{'='*60}")
    
    total_pixels = 0
    class_counts = {}
    
    for i, sem_file in enumerate(sem_files):
        # Read semantic label
        sem = tifffile.imread(sem_file)
        
        # Count pixels per class
        unique, counts = np.unique(sem, return_counts=True)
        
        for cls, count in zip(unique, counts):
            class_counts[cls] = class_counts.get(cls, 0) + count
            total_pixels += count
        
        if i == 0:
            print(f"\nFirst file: {os.path.basename(sem_file)}")
            print(f"  Shape: {sem.shape}")
            print(f"  Dtype: {sem.dtype}")
            print(f"  Unique classes: {unique}")
    
    # Print statistics
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    print(f"\nClass distribution:")
    print(f"{'Class':<10} {'Count':<15} {'Percentage':<12} {'Weight (inverse)'}")
    print("-" * 60)
    
    sorted_classes = sorted(class_counts.keys())
    for cls in sorted_classes:
        count = class_counts[cls]
        percentage = (count / total_pixels) * 100
        # Calculate inverse frequency weight
        weight = total_pixels / (len(sorted_classes) * count)
        print(f"{cls:<10} {count:<15,} {percentage:<11.2f}% {weight:.2f}x")
    
    # Calculate class imbalance ratio
    if len(sorted_classes) == 2:
        ratio = class_counts[sorted_classes[0]] / class_counts[sorted_classes[1]]
        print(f"\nClass imbalance ratio (class {sorted_classes[0]} : class {sorted_classes[1]}): {ratio:.2f}:1")
    
    return class_counts, total_pixels

if __name__ == "__main__":
    # Check both mini and full datasets
    datasets = [
        ('DFC2019_crp512_bin_mini', '../../datasets/DFC2019_crp512_bin_mini'),
        ('DFC2019_crp512_bin', '../../datasets/DFC2019_crp512_bin')
    ]
    
    for dataset_name, dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\n{'#'*60}")
            print(f"# Dataset: {dataset_name}")
            print(f"{'#'*60}")
            
            # Analyze train split
            train_counts, train_total = analyze_dataset(dataset_path, 'train')
            
            # Analyze valid split if exists
            valid_path = os.path.join(dataset_path, 'valid')
            if os.path.exists(valid_path):
                valid_counts, valid_total = analyze_dataset(dataset_path, 'valid')
        else:
            print(f"\nDataset not found: {dataset_path}")
