# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

import os
import torch
import torch.nn as nn
import torch.distributed as dist

# Multi-GPU Configuration
USE_MULTI_GPU = True  # Set to True to enable multi-GPU training
GPU_IDS = [0, 1, 2, 3]  # List of GPU IDs to use (adjust based on available GPUs)
MASTER_GPU = 0  # Master GPU for model initialization

# Set CUDA device visibility (optional, can be controlled externally)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Uncomment and adjust as needed

# PyTorch CUDA settings
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic operations for speed

# Define the dataset to be used for training and testing
# Options include Vaihingen, Vaihingen_crp256, DFC2018, DFC2018_crp256, DFC2019_crp256, DFC2019_crp256_bin, DFC2019_crp512, 
# and DFC2023 derivatives as follows:
# DFC2023A (Ahmad's splitting), DFC2023Asmall, DFC2023Amini, and DFC2023S (Sinan's splitting) datasets
dataset_name = 'DFC2023Amini'  # Change this to the desired dataset name

# Shortcut path to the datasets parent folder
# Because these files may be voluminous, thus you may put them inside another folder to be 
# globally available to other projects as well. You should end the path with a '/'
shortcut_path = '../datasets/'  # Change this to the desired path

# Whether the input image tile is large, thus random patches are selected out of that, (DFC2018 and Vaihingen)
# Or the input image is like a normal patch, thus as a whole could be fed to the model, (DFC2023)
large_tile_data = ['Vaihingen', 'DFC2018']
large_tile_mode = dataset_name in large_tile_data

# Define datasets that use RGB triplets for semantic labels
rgb_label_datasets = ['Vaihingen', 'Vaihingen_crp256']  # Add other RGB-triplet datasets here
uses_rgb_labels = dataset_name in rgb_label_datasets

# Define the flag for synthetic aperture radar (SAR) channel for the input tensor.
# This could be the case for DFC2023 in which the input RGB and the 1-channel SAR images are fused together 
# to provide the model with more precise information.
# However, in such a case, as the DenseNet architecture cannot capture more than 3 channels for 
# its input, thus we should use a convolution layer prior to that to convert the 4-channel input 
# to a 3-channel one.
datasets_with_sar = ['DFC2023']  # List of datasets that support SAR mode
sar_path_indicator = dataset_name.startswith(tuple(datasets_with_sar))
sar_mode = False

# Normalization flag for input RGB, DSM, etc
normalize_flag = False

# Set flags for additive heads of MTL, viz semantic segmentation, surface normals, and edgemaps
sem_flag, norm_flag, edge_flag = True, True, False

# Set flag for MTL heads interconnection mode, either fully intertwined ('full') or just for the DSM head ('dsm')
mtl_head_mode = 'dsm'  # 'full' or 'dsm'

# MTL backbone frozen mode (define early for batch size calculation)
mtl_bb_freeze = False

# Dynamic batch size calculation based on model complexity factors
def calculate_model_complexity_factor():
    """
    Calculate memory complexity multiplier based on active model components.
    Returns a factor that will be used to adjust batch sizes.
    """
    complexity_factor = 1.0
    
    # MTL head mode impact (most significant factor)
    if mtl_head_mode == 'full':
        complexity_factor *= 3.5  # Full interconnected heads use ~3.5x more memory
    else:  # 'dsm' mode
        complexity_factor *= 1.0  # Base complexity
    
    # Active heads impact (affects concatenation complexity in 'full' mode)
    active_heads = sum([sem_flag, norm_flag, edge_flag])
    if mtl_head_mode == 'full':
        # In full mode, each additional head multiplies concatenation channels
        complexity_factor *= (1.0 + active_heads * 0.3)  # ~30% increase per additional head
    else:
        # In dsm mode, additional heads have minimal impact
        complexity_factor *= (1.0 + active_heads * 0.1)  # ~10% increase per additional head
    
    # SAR mode impact (additional input channel processing)
    if sar_mode:
        complexity_factor *= 1.2  # 20% increase for SAR processing
    
    # Large tile mode impact (more complex data augmentation patterns)
    if large_tile_mode:
        complexity_factor *= 1.1  # 10% increase for large tile processing
    
    # Frozen backbone reduces memory usage (no gradient storage for backbone)
    if mtl_bb_freeze:
        complexity_factor *= 0.8  # 20% reduction when backbone is frozen
    
    return complexity_factor

def calculate_dynamic_batch_size(base_crop_size, base_batch_per_gpu, dataset_size=None):
    """
    Calculate dynamic batch size based on crop size and model complexity.
    
    Args:
        base_crop_size: Base crop size for the dataset
        base_batch_size_per_gpu: Base batch size per GPU for this crop size
        dataset_size: Number of training samples (for small dataset optimization)
    
    Returns:
        Tuple of (total_batch_size, complexity_factor, adjusted_batch_per_gpu)
    """
    num_gpus = len(GPU_IDS) if USE_MULTI_GPU else 1
    complexity_factor = calculate_model_complexity_factor()
    
    # Calculate memory usage scaling with optimized factors
    # Memory scales with crop_size^2 * complexity_factor
    crop_memory_factor = (base_crop_size / 256) ** 2  # Normalize to 256x256 baseline
    
    # Apply less conservative scaling for better GPU utilization
    # The complexity factors were too conservative for your available memory
    if complexity_factor < 1.5:  # Low complexity (dsm mode)
        effective_complexity = complexity_factor * 0.7  # Reduce impact by 30%
    elif complexity_factor < 2.5:  # Medium complexity
        effective_complexity = complexity_factor * 0.8  # Reduce impact by 20%
    else:  # High complexity (full mode)
        effective_complexity = complexity_factor * 0.9  # Reduce impact by 10%
    
    total_memory_factor = crop_memory_factor * effective_complexity
    
    # Adjust batch size inversely to memory requirements
    adjusted_batch_per_gpu = max(1, int(base_batch_per_gpu / total_memory_factor))
    
    # Special handling for small datasets
    if dataset_size is not None:
        # For small datasets, ensure we have at least 4-5 batches per epoch for stability
        min_batches_per_epoch = 5
        max_batch_size_per_gpu = max(1, dataset_size // (min_batches_per_epoch * num_gpus))
        if max_batch_size_per_gpu < adjusted_batch_per_gpu:
            adjusted_batch_per_gpu = max_batch_size_per_gpu
            print(f"Reducing batch size for small dataset ({dataset_size} samples) to ensure {min_batches_per_epoch}+ batches per epoch")
    
    # Apply GPU memory-based boost for better utilization
    # With 7-10GB available per GPU, we can afford larger batch sizes
    if adjusted_batch_per_gpu < 4 and (dataset_size is None or dataset_size >= 100):  # Only boost for larger datasets
        memory_boost = min(3, 4 // adjusted_batch_per_gpu)  # Boost by 2-3x
        adjusted_batch_per_gpu = min(base_batch_per_gpu, adjusted_batch_per_gpu * memory_boost)
    
    total_batch_size = adjusted_batch_per_gpu * num_gpus
    
    # Ensure minimum batch size
    total_batch_size = max(num_gpus, total_batch_size)
    
    return total_batch_size, complexity_factor, adjusted_batch_per_gpu

# Known small datasets that need special batch size handling
small_datasets = {
    'DFC2023Amini': 20,  # Only 20 training samples
    # Add other small datasets here as needed
}

# Get dataset size for optimization
dataset_size = small_datasets.get(dataset_name, None)

# Base dataset configurations optimized for available GPU memory (7-10GB per GPU)
# These will be dynamically adjusted based on actual configuration flags
base_dataset_configs = {
    # Optimized base batch sizes for better GPU utilization
    # With 7-10GB available memory per GPU, we can afford larger batch sizes
    'Vaihingen': (320, 24),          # Increased from 16 to 24 for 320x320
    'DFC2018': (320, 24),            # Increased from 16 to 24 for 320x320
    
    'Vaihingen_crp256': (256, 32),   # Increased from 24 to 32 for 256x256
    'DFC2018_crp256': (256, 32),     # Increased from 24 to 32 for 256x256  
    'DFC2019_crp256': (256, 32),     # Increased from 24 to 32 for 256x256
    
    'DFC2019_crp512': (512, 12),     # Increased from 8 to 12 for 512x512
    'DFC2023': (512, 12),            # Increased from 8 to 12 for 512x512
}

# Get base cropSize and batch_per_gpu, then calculate dynamic batch size
if dataset_name in base_dataset_configs:
    cropSize, base_batch_per_gpu = base_dataset_configs[dataset_name]
else:
    # Find by prefix match if exact match not found
    matching_datasets = [d for d in base_dataset_configs.keys() if dataset_name.startswith(d)]
    if matching_datasets:
        # Sort by length descending to get the most specific match
        best_match = sorted(matching_datasets, key=len, reverse=True)[0]
        cropSize, base_batch_per_gpu = base_dataset_configs[best_match]
    else:
        # Conservative fallback for unknown datasets
        cropSize, base_batch_per_gpu = 256, 4  # Very conservative base

# Calculate dynamic batch size based on actual configuration
batch_size, complexity_factor, batch_per_gpu = calculate_dynamic_batch_size(cropSize, base_batch_per_gpu, dataset_size)

# Print dynamic batch size configuration results
print(f"\n=== Dynamic Batch Size Configuration ===")
print(f"Dataset: {dataset_name}")
print(f"Crop size: {cropSize}x{cropSize}")
print(f"MTL head mode: {mtl_head_mode}")
print(f"Active heads: sem={sem_flag}, norm={norm_flag}, edge={edge_flag}")
print(f"SAR mode: {sar_mode}")
print(f"Large tile mode: {large_tile_mode}")
print(f"Backbone frozen: {mtl_bb_freeze}")
print(f"Model complexity factor: {complexity_factor:.2f}")
print(f"Base batch per GPU: {base_batch_per_gpu}")
print(f"Adjusted batch per GPU: {batch_per_gpu}")
print(f"Total batch size: {batch_size} (across {len(GPU_IDS) if USE_MULTI_GPU else 1} GPUs)")
print(f"========================================\n")

# Parameters for the Multitask Learning (MTL) component
mtl_lr_decay = False  # Flag to enable/disable learning rate decay
mtl_lr = 0.0002  # Initial learning rate for the MTL network
mtl_batchSize = batch_size  # Batch size for training the MTL network, now dynamic based on dataset
mtl_numEpochs = 1000  # Number of epochs for training the MTL network

# Total number of training samples available for MTL generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
mtl_training_samples = 10000
# Calculate the total number of iterations required for training based on batch size and samples count
mtl_train_iters = int(mtl_training_samples / mtl_batchSize)
mtl_log_freq = int(mtl_train_iters / 5)  # Frequency at which evaluation metrics are calculated during training
mtl_min_loss = float('inf')  # Minimum DSM loss threshold to save the MTL network weights as checkpoints

# Parameters for the Denoising AutoEncoder (DAE) component defined as the same way for MTL
dae_lr_decay = False
dae_lr = 0.0002
dae_batchSize = batch_size  # Batch size for training the DAE network, now dynamic based on dataset
dae_numEpochs = 1000

# Total number of training samples available for DAE generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
dae_training_samples = 10000
dae_train_iters = int(dae_training_samples / dae_batchSize)
dae_log_freq = int(dae_train_iters / 5)
dae_min_loss = float('inf')  # Minimum loss (DSM noise) threshold to save the DAE network weights as checkpoints

# MTL saved weights preloading mode. If True, then all MTL model will be initialized with saved weights before training
mtl_preload = False

# DAE saved weights preloading mode. If True, then all DAE model will be initialized with saved weights before training
dae_preload = False

# Define the status and the path to save checkpoints for MTL and Unet
# Only add SAR mode indicator for DFC2023 datasets
sar_indicator = ('+sar' if sar_mode else '-sar') if sar_path_indicator else '.'
predCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/mtl'  # MTL checkpoints path
corrCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/refinement'  # DAE checkpoints path

# Log file directory configuration
log_output_dir = f'./output/{dataset_name}/_logs'

# Ensure log directory exists
os.makedirs(log_output_dir, exist_ok=True)

# Define log file paths
mtl_log_file = f'{log_output_dir}/{dataset_name}_mtl_train_pytorch_output.log'
dae_log_file = f'{log_output_dir}/{dataset_name}_dae_train_pytorch_output.log'
test_log_file = f'{log_output_dir}/{dataset_name}_test_pytorch_output.log'

# Initialize the epoch counter for the last saved weights when validation is disabled
last_epoch_saved = None 

# Set flag to either calculate the train/valid error after every epoch or just ignore it
# If ignored, then the train and valid sets will be merged to form one unique train set
# !!! If set to True, then be careful about the 'correction' flag as it will affect the train/valid error computations
train_valid_flag = True

# Early stopping configuration only if train_valid_flag is set to True
early_stop_flag = True  # Enable/disable early stopping 
early_stop_patience = 10  # Number of epochs to wait for improvement before stopping
early_stop_delta = 1e-2  # Minimum change in monitored value to qualify as an improvement

# Evaluation metric configuration
# Define available metrics first, since other configs depend on it
metric_names = ['mse', 'mae', 'rmse', 'delta1', 'delta2', 'delta3']

# Categorize metrics into error metrics (lower is better) and accuracy metrics (higher is better)
height_error_metrics = ['mse', 'mae', 'rmse']
height_accuracy_metrics = ['delta1', 'delta2', 'delta3']

# Segmentation metrics separated by type
segmentation_class_metrics = ['iou', 'precision', 'recall', 'f1_score']  # Per-class metrics
segmentation_scalar_metrics = ['miou', 'oa', 'fwiou']  # Overall metrics
segmentation_accuracy_metrics = segmentation_class_metrics + segmentation_scalar_metrics

eval_metric = 'rmse'  # Options: must be one of metric_names
# Automatically determine if lower is better based on metric type
eval_metric_lower_better = eval_metric in height_error_metrics

# Initialize early stopping variables based on metric configuration
best_metric = float('inf') if eval_metric_lower_better else float('-inf')
patience_counter = 0

# Plot configuration for train/valid errors
plot_train_error = False  # Flag to enable/disable calculating and plotting training errors
# Example: Add 'miou' to see segmentation metrics
plot_metrics = ['rmse', 'delta1', 'iou', 'miou']  # List of metrics to plot from metric_names + segmentation_accuracy_metrics

# Set the regression loss mode, either MSE or Huber
reg_loss = 'mse'  # 'mse' or 'huber'
huber_delta = 0.1  # Huber loss hyperparameter, delta

# Edgemap configuration
# Threshold for potential rooftops out of nDSM (pixel values are supposed to be in [0, 255])
roof_height_threshold = 50
# Canny edge detection algorithm low and high thresholds for detecting potential edges for rooftops
canny_lt, canny_ht = 50, 150

# Set flag for applying denoising autoencoder during testing. 
# Note: If set to True, this will affect train/valid error computations
correction = True

# Define label codes for semantic segmentation task, and
# scaling factors (weights) for different types of loss functions in MTL
# Note: You may change the scaling factors based on your discretion.
if 'Vaihingen' in dataset_name:
    label_codes = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    w1, w2, w3, w4 = (1e-4, 1e-1, 1e-5, 0.001)  # weights for: dsm, sem, norm, edge

elif 'DFC2018' in dataset_name:
    label_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    w1, w2, w3, w4 = (100.0, 1.0, 10.0, 100.0)  # weights for: dsm, sem, norm, edge

elif dataset_name.startswith('DFC2019'):
    if dataset_name.endswith('bin'):
        label_codes = [0, 1]
        w1, w2, w3, w4 = (1e-2, 1e-1, 1e-5, 100.0)  # weights for: dsm, sem, norm, edge
    else:
        label_codes = [2, 5, 6, 9, 17, 65]
        w1, w2, w3, w4 = (1e-2, 1e-1, 1e-5, 100.0)  # weights for: dsm, sem, norm, edge

elif dataset_name.startswith('DFC2023'):
    label_codes = [0, 1]
    w1, w2, w3, w4 = (1e-3, 1.0, 1e-5, 1e-3)  # weights for: dsm, sem, norm, edge

# Create dictionary and indicator for semantic label codes
semantic_label_map = {k: v for k, v in enumerate(label_codes)}
sem_k = len(semantic_label_map)

# Check if the dataset is binary classification based on label codes
binary_classification_flag = len(label_codes) == 2 and set(label_codes) == {0, 1}

# Device Configuration
def setup_device():
    """Setup device configuration for single or multi-GPU training"""
    if torch.cuda.is_available():
        if USE_MULTI_GPU and len(GPU_IDS) > 1:
            device = torch.device(f'cuda:{MASTER_GPU}')
            print(f"Using multi-GPU training on devices: {GPU_IDS}")
        else:
            device = torch.device(f'cuda:{GPU_IDS[0] if GPU_IDS else 0}')
            print(f"Using single GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    return device

def setup_distributed():
    """Setup distributed training if using multiple GPUs"""
    if USE_MULTI_GPU and len(GPU_IDS) > 1 and torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            # Initialize the process group for multi-GPU training
            dist.init_process_group(backend='nccl')
        return True
    return False

# Initialize device
device = setup_device() if 'torch' in globals() else None
