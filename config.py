# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024

import os
# Set up GPU environment variables for CUDA device management
# Ensure CUDA devices are ordered by PCI bus ID for consistent behavior across sessions
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Multi-GPU configuration
# Specify which GPUs to use for training (comma-separated)
# For single GPU: "0", For multi-GPU: "0,1" or "0,1,2,3" etc.
gpu_devices = "0"  # Change this to your available GPU indices
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

# Automatically determine multi-GPU mode based on number of devices specified
# Set to False for single GPU, True for multi-GPU training
multi_gpu_enabled = len(gpu_devices.split(',')) > 1
# Set TensorFlow log level to a specific level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all messages, 1 = filter out INFO, 2 = filter out INFO & WARNINGS, 3 = only ERROR messages

# Define the dataset to be used for training and testing
# Options include Vaihingen, Vaihingen_crp256, DFC2018, DFC2018_crp256, DFC2019_crp256, DFC2019_crp256_bin, DFC2019_crp512, 
# DFC2019_crp512_bin, and DFC2023 derivatives as follows:
# DFC2023A (Ahmad's splitting), DFC2023Asmall, DFC2023mini, and DFC2023S (Sinan's splitting) datasets
dataset_name = 'SSBHmini'  # Change this to the desired dataset name

# Shortcut path to the datasets parent folder
# Because these files may be voluminous, thus you may put them inside another folder to be 
# globally available to other projects as well. You should end the path with a '/'
shortcut_path = '../../datasets/'  # Change this to the desired path

# Dataset organization configuration
# Large tile datasets: require random patch extraction
large_tile_datasets = ['Vaihingen', 'DFC2018']
# Regular size datasets: use standard folder structure (train/valid/test with rgb/dsm/sem/sar subfolders)
regular_size_datasets = ['DFC2019_crp256', 'DFC2019_crp512', 'DFC2023', 'Vaihingen_crp256', 
                         'DFC2018_crp256', 'Dublin', 'Dublin_ndsm', 'Contest', 'Huawei_Contest', 'SSBH']

# Datasets with SAR data available
sar_datasets = ['DFC2023']

# Datasets without semantic segmentation labels
# These datasets will not have semantic segmentation labels, thus the model will not be trained for that
# They will only be trained for DSM and surface normals, e.g., Dublin
no_sem_datasets = ['Dublin']  # Datasets without semantic segmentation labels

# Datasets that require center cropping to match model input dimensions
center_crop_datasets = ['Dublin']

# Whether the input image tile is large, thus random patches are selected out of that, (DFC2018 and Vaihingen)
# Or the input image is like a normal patch, thus as a whole could be fed to the model, (DFC2023)
large_tile_mode = dataset_name in large_tile_datasets

# Define datasets that use RGB triplets for semantic labels
rgb_label_datasets = ['Vaihingen', 'Vaihingen_crp256']  # Add other RGB-triplet datasets here
uses_rgb_labels = dataset_name in rgb_label_datasets

# Define a combined dictionary with (cropSize, batchSize) tuples for each dataset
dataset_configs = {
    'Vaihingen': (320, 4),
    'DFC2018': (320, 4),
    'Vaihingen_crp256': (256, 6),
    'DFC2018_crp256': (256, 6),
    'DFC2019_crp256': (256, 6),
    'DFC2019_crp512': (512, 2),
    'DFC2023': (512, 2),
    'Contest': (512, 2),
    'Huawei_Contest': (512, 2),  # Huawei Contest dataset
    'Dublin': (480, 2),  # Updated to match model output dimensions (480 is divisible by 32)
    'SSBH': (256, 6),  # SSBH dataset: 256x256px samples with moderate batch size
}

# Get cropSize and batchSize based on dataset name, with fallback logic
# First try exact match, then prefix match, then default to (256, 2)
if dataset_name in dataset_configs:
    cropSize, batch_size = dataset_configs[dataset_name]
else:
    # Find by prefix match if exact match not found
    matching_datasets = [d for d in dataset_configs.keys() if dataset_name.startswith(d)]
    if matching_datasets:
        # Sort by length descending to get the most specific match
        best_match = sorted(matching_datasets, key=len, reverse=True)[0]
        cropSize, batch_size = dataset_configs[best_match]
    else:
        # Default values if no match found
        cropSize, batch_size = 256, 2

# Define the flag for synthetic aperture radar (SAR) channel for the input tensor.
# This could be the case for DFC2023 in which the input RGB and the 1-channel SAR images are fused together 
# to provide the model with more precise information.
# However, in such a case, as the DenseNet architecture cannot capture more than 3 channels for 
# its input, thus we should use a convolution layer prior to that to convert the 4-channel input 
# to a 3-channel one.
sar_path_indicator = any(dataset_name.startswith(d) for d in sar_datasets)
sar_mode = False

## Dataset-specific learning rate configuration
# Learning Rate Configuration Notes for SSBH Dataset:
# ===================================================
# SSBH uses uint16 RGB images normalized to [0,1] instead of the typical uint8 [0,255] range.
# This smaller input magnitude affects gradient flow and requires adjusted learning rates, typically lower than usual.
# ===================================================
if dataset_name.startswith('SSBH'):
    # REDUCED learning rates to fix semantic segmentation failure (Class 1 IoU = 0.0000)
    mtl_lr = 0.000001 
    dae_lr = 0.0000001
else:
    # Default learning rates for uint8 [0,255] RGB datasets
    mtl_lr = 0.0002   # Initial learning rate for the MTL network
    dae_lr = 0.00002  # Initial learning rate for the DAE network

# Parameters for the Multitask Learning (MTL) component
mtl_lr_decay = False  # Flag to enable/disable learning rate decay
mtl_batchSize = batch_size  # Batch size for training the MTL network, now dynamic based on dataset
mtl_numEpochs = 1000  # Number of epochs for training the MTL network (reduced for testing)

# Total number of training samples available for MTL generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
mtl_training_samples = 10000
# Note: mtl_train_iters will be calculated after global batch size is determined
mtl_min_loss = float('inf')  # Minimum DSM loss threshold to save the MTL network weights as checkpoints

# Parameters for the Denoising AutoEncoder (DAE) component defined as the same way for MTL
dae_lr_decay = False
# Note: For DAE, we may use a smaller learning rate for better convergence
# This is especially useful for datasets with high noise levels or when fine-tuning a pre-trained model
# Example: For DFC2019_bin, we may use a smaller learning rate
dae_batchSize = batch_size  # Batch size for training the DAE network, now dynamic based on dataset
dae_numEpochs = 1000  # Number of epochs for training the DAE network (reduced for testing)

# Total number of training samples available for DAE generated out of data augmentation technique for large tiles, 
# o.w. for input data as patches, the true number of training samples will be used accordingly
dae_training_samples = 10000
# Note: dae_train_iters will be calculated after global batch size is determined
dae_min_loss = float('inf')  # Minimum loss (DSM noise) threshold to save the DAE network weights as checkpoints

# MTL saved weights preloading mode. If True, then all MTL model will be initialized with saved weights before training
mtl_preload = False
# MTL backbone frozen mode. If True, then the MTL backbone weights will not get updated during training to save time
mtl_bb_freeze = False

# DAE saved weights preloading mode. If True, then all DAE model will be initialized with saved weights before training
dae_preload = False

# Define the status and the path to save checkpoints for MTL and Unet
# Only add SAR mode indicator for DFC2023 datasets
sar_indicator = ('+sar' if sar_mode else '-sar') if sar_path_indicator else '.'
predCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/mtl'  # MTL checkpoints path
corrCheckPointPath = f'./checkpoints/{dataset_name}/{sar_indicator}/refinement'  # DAE checkpoints path

# Define output paths for logs and plots
output_base_path = f'./output/{dataset_name}/{sar_indicator}'
log_output_path = f'{output_base_path}/_logs'
plot_output_path = f'{output_base_path}/_plots'

# Ensure output directories exist
import os
os.makedirs(log_output_path, exist_ok=True)
os.makedirs(plot_output_path, exist_ok=True)

# Initialize the epoch counter for the last saved weights when validation is disabled
last_epoch_saved = None 

# Set flag to either calculate the train/valid error after every epoch or just ignore it
# If ignored, then the train and valid sets will be merged to form one unique train set
# !!! If set to True, then be careful about the 'correction' flag as it will affect the train/valid error computations
train_valid_flag = True

# Early stopping configuration only if train_valid_flag is set to True
early_stop_flag = True  # Enable/disable early stopping 
early_stop_patience = 10  # Number of epochs to wait for improvement before stopping
early_stop_delta = 1e-1  # Minimum change in monitored value to qualify as an improvement

# Evaluation metric configuration
# Define available metrics first, since other configs depend on it
metric_names = ['mse', 'mae', 'rmse', 'delta1', 'delta2', 'delta3', 
                'rmse_building', 'rmse_matched', 
                'high_rise_rmse', 'mid_rise_rmse', 'low_rise_rmse']

# Categorize metrics into error metrics (lower is better) and accuracy metrics (higher is better)
height_error_metrics = ['mse', 'mae', 'rmse', 'rmse_building', 'rmse_matched', 
                        'high_rise_rmse', 'mid_rise_rmse', 'low_rise_rmse']
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
plot_metrics = ['rmse', 'delta1', 'iou', 'f1_score']  # List of metrics to plot from metric_names + segmentation_accuracy_metrics

# Set the regression loss mode, either MSE or Huber
reg_loss = 'mse'  # 'mse' or 'huber'
huber_delta = 0.1  # Huber loss hyperparameter, delta

# Edgemap configuration
# Threshold for potential rooftops out of nDSM (pixel values are supposed to be in [0, 255])
roof_height_threshold = 50
# Canny edge detection algorithm low and high thresholds for detecting potential edges for rooftops
canny_lt, canny_ht = 50, 150

# Building height classification thresholds for height-based metrics
# These thresholds define building categories based on height in meters
low_rise_max = 15    # Buildings with height < 15m are considered low-rise
mid_rise_max = 40    # Buildings with height >= 15m and < 40m are considered mid-rise
                     # Buildings with height >= 40m are considered high-rise

# Set flags for additive heads of MTL, viz semantic segmentation, surface normals, and edgemaps
sem_flag, norm_flag, edge_flag = True, True, False
sem_flag = False if any(dataset_name.startswith(d) for d in no_sem_datasets) else sem_flag  # Disable semantic segmentation for datasets without labels

# Set flag for MTL heads interconnection mode, either fully intertwined ('full') or just for the DSM head ('dsm')
mtl_head_mode = 'dsm'  # 'full' or 'dsm'

# Set flag for applying denoising autoencoder during testing. 
# Note: If set to True, this will affect train/valid error computations
# This is because the DAE will be used to denoise the DSM before calculating the errors.
# Can be overridden by setting CORRECTION environment variable
correction = os.environ.get('CORRECTION', 'True').lower() == 'true'
correction = False  # IGNORE

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

elif dataset_name.startswith(('Contest', 'Huawei_Contest')):
    label_codes = [0, 1]
    w1, w2, w3, w4 = (1e-5, 1e-6, 1e-10, 1e-3)  # weights for: dsm, sem, norm, edge

elif dataset_name.startswith('Dublin'):
    # Dublin dataset has no semantic segmentation labels
    label_codes = [0]  # Dummy label code for datasets without semantic segmentation
    w1, w2, w3, w4 = (1e-3, 0.0, 1e-5, 1e-3)  # weights for: dsm, sem (disabled), norm, edge

elif dataset_name.startswith('SSBH'):
    # SSBH dataset: binary building classification (background=0, building=1) with much class imbalance
    label_codes = [0, 1]
    # w1, w2, w3, w4 = (1e-1, 50.0, 1e-5, 1e-3)  # weights for: dsm, sem, norm, edge
    w1, w2, w3, w4 = (1e0, 1e3, 1e-3, 1e-3)  # weights for: dsm, sem, norm, edge

# Handle datasets without semantic labels
if any(dataset_name.startswith(d) for d in no_sem_datasets):
    # For datasets without semantic labels, set minimal configuration
    # Note: This dummy mapping is only for configuration consistency
    # Actual semantic processing is disabled via sem_flag=False
    semantic_label_map = {0: 0}  # Minimal mapping for config consistency
    sem_k = 1  # One dummy class for config purposes only
    binary_classification_flag = False
else:
    # Create dictionary and indicator for semantic label codes
    semantic_label_map = {k: v for k, v in enumerate(label_codes)}
    sem_k = len(semantic_label_map)
    # Check if the dataset is binary classification based on label codes
    binary_classification_flag = len(label_codes) == 2 and set(label_codes) == {0, 1}

# Multi-GPU Strategy Configuration
# This section handles the TensorFlow distributed strategy for multi-GPU training
import tensorflow as tf

if multi_gpu_enabled:
    # Use MirroredStrategy for synchronous training across multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    
    # For multi-GPU training, we need to be careful about memory usage
    # strategy.run() will automatically distribute the batch across GPUs
    # So if we want each GPU to process 'batch_size' samples, we need to generate
    # batch_size * num_gpus total samples
    
    # However, for memory efficiency, let's reduce per-GPU batch size for multi-GPU
    # Reduce batch size based on crop size and number of GPUs to avoid OOM
    if cropSize >= 512 and strategy.num_replicas_in_sync >= 3:
        # For large images (512x512) with 3+ GPUs, use very small per-GPU batch
        per_gpu_mtl_batch = 1
        per_gpu_dae_batch = 1
        print(f"INFO: Reducing per-GPU batch size to 1 due to large image size ({cropSize}px) with {strategy.num_replicas_in_sync} GPUs to avoid OOM")
    elif strategy.num_replicas_in_sync >= 4:
        # For 4+ GPUs, use smaller per-GPU batch size to avoid OOM
        per_gpu_mtl_batch = max(1, mtl_batchSize // 2)  # Reduce by half
        per_gpu_dae_batch = max(1, dae_batchSize // 2)  # Reduce by half
        print(f"INFO: Reducing per-GPU batch size from {mtl_batchSize} to {per_gpu_mtl_batch} due to {strategy.num_replicas_in_sync} GPUs to avoid OOM")
    else:
        # For 2 GPUs or smaller images, use the original batch size
        per_gpu_mtl_batch = mtl_batchSize
        per_gpu_dae_batch = dae_batchSize
        print(f"INFO: Using original per-GPU batch size ({per_gpu_mtl_batch}) - optimal configuration for {strategy.num_replicas_in_sync} GPUs with {cropSize}px images")
    
    # Global batch size is what we'll generate (will be split across GPUs)
    global_batch_size = per_gpu_mtl_batch * strategy.num_replicas_in_sync
    mtl_global_batch_size = per_gpu_mtl_batch * strategy.num_replicas_in_sync  
    dae_global_batch_size = per_gpu_dae_batch * strategy.num_replicas_in_sync
    
    print(f"Per-GPU MTL batch size: {per_gpu_mtl_batch}")
    print(f"Per-GPU DAE batch size: {per_gpu_dae_batch}")
    print(f"Global MTL batch size: {mtl_global_batch_size}")
    print(f"Global DAE batch size: {dae_global_batch_size}")
    
    # Calculate training iterations based on GLOBAL batch size for multi-GPU
    mtl_train_iters = int(mtl_training_samples / mtl_global_batch_size)
    dae_train_iters = int(dae_training_samples / dae_global_batch_size)
else:
    # Use default strategy for single GPU
    strategy = tf.distribute.get_strategy()
    global_batch_size = batch_size
    mtl_global_batch_size = mtl_batchSize
    dae_global_batch_size = dae_batchSize
    
    # Calculate training iterations based on per-GPU batch size for single GPU
    mtl_train_iters = int(mtl_training_samples / mtl_batchSize)
    dae_train_iters = int(dae_training_samples / dae_batchSize)

# Calculate log frequency based on training iterations
mtl_log_freq = int(mtl_train_iters / 5)  # Frequency at which evaluation metrics are calculated during training
dae_log_freq = int(dae_train_iters / 5)
