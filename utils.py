# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024

import random
import numpy as np
import glob
import cv2
import PIL
import os
import tensorflow as tf
from PIL import Image
from skimage import io
from typing import Optional, List, Tuple
import logging
from config import *

Image.MAX_IMAGE_PIXELS = 1000000000


def collect_tilenames(mode):
    """
    Collects filenames for RGB, SAR, DSM, and SEM images based on the specified dataset and mode (train, valid, or test).

    Parameters:
    - mode (str): The mode for which to collect filenames ('train', 'valid', or 'test').

    Returns:
    - Tuple of lists: A tuple containing lists of filepaths for RGB, SAR, DSM, and SEM images.
    """
    all_rgb, all_sar, all_dsm, all_sem = [], [], [], []

    # Handle large tile datasets with predefined frames
    if dataset_name in large_tile_datasets:
        if dataset_name == 'Vaihingen':
            base_path = shortcut_path + 'Vaihingen/'
            train_frames = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 30, 34]
            valid_frames = [28, 32, 37]
            test_frames = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
            
            if mode == 'train':
                frames = train_frames
            elif mode == 'valid':
                frames = valid_frames
            else:
                frames = test_frames
                
            for frame in frames:
                all_rgb.append(base_path + 'RGB/top_mosaic_09cm_area' + str(frame) + '.tif')
                all_dsm.append(base_path + 'NDSM/dsm_09cm_matching_area' + str(frame) + '.jpg')
                all_sem.append(base_path + 'SEM/top_mosaic_09cm_area' + str(frame) + '.tif')

        elif dataset_name == 'DFC2018':
            base_path = shortcut_path + 'DFC2018/'
            train_frames = ['UH_NAD83_272056_3289689', 'UH_NAD83_272652_3289689', 'UH_NAD83_273248_3289689']
            valid_frames = ['UH_NAD83_273844_3289689']
            test_frames = ['UH_NAD83_271460_3289689', 'UH_NAD83_271460_3290290', 'UH_NAD83_272056_3290290',
                           'UH_NAD83_272652_3290290', 'UH_NAD83_273248_3290290', 'UH_NAD83_273844_3290290',
                           'UH_NAD83_274440_3289689', 'UH_NAD83_274440_3290290', 'UH_NAD83_275036_3289689',
                           'UH_NAD83_275036_3290290']
            
            if mode == 'train':
                frames = train_frames
            elif mode == 'valid':
                frames = valid_frames
            else:
                frames = test_frames
                
            for frame in frames:
                all_rgb.append(base_path + 'RGB/' + frame + '.tif')
                all_dsm.append(base_path + 'DSM/' + frame + '.tif')
                all_dsm.append(base_path + 'DEM/' + frame + '.tif')
                all_sem.append(base_path + 'SEM/' + frame + '.tif')

    # Handle regular-sized datasets with standard folder structure
    else:
        base_path = shortcut_path + dataset_name + '/' + mode + '/'
        
        # Define subfolder mappings - all regular datasets use same structure
        subfolders = ['rgb', 'dsm', 'sem']
        if any(dataset_name.startswith(d) for d in sar_datasets):
            subfolders.append('sar')
        
        for subfolder in subfolders:
            folder_path = base_path + subfolder + '/'
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.tif', '.jpg', '.png')):
                        filepath = folder_path + filename
                        if subfolder == 'rgb':
                            all_rgb.append(filepath)
                        elif subfolder == 'sar':
                            all_sar.append(filepath)
                        elif subfolder == 'dsm':
                            all_dsm.append(filepath)
                        elif subfolder == 'sem':
                            all_sem.append(filepath)
        
    samples_no = len(all_rgb)
    return all_rgb, all_sar, all_dsm, all_sem, samples_no


def generate_training_batches(train_rgb, train_sar, train_dsm, train_sem, iter, mtl_flag):
    """
    Generate training batches for multi-task learning from RGB, SAR, DSM and semantic segmentation data.
    This function creates batches of training data by either:
    1. Randomly sampling patches from large input tiles (for tile_mode datasets like Vaihingen and DFC2018)
    2. Sequentially selecting patches based on iteration number (for other datasets like DFC2023)
    
    For multi-GPU training, this function generates batches of size equal to the GLOBAL batch size,
    which will be automatically distributed across GPUs by TensorFlow's MirroredStrategy.
    
    Parameters:
    ----------
    train_rgb : list
        List of paths to RGB image files
    train_sar : list
        List of paths to SAR image files (optional)
    train_dsm : list  
        List of paths to DSM (Digital Surface Model) files
    train_sem : list
        List of paths to semantic segmentation label files
    iter : int
        Current iteration number for sequential batch selection
    mtl_flag : bool
        Flag to enable multi-task learning outputs (semantic, normals, edges)
    Returns:
    -------
    tuple
        - rgb_batch : numpy.ndarray
            Batch of RGB (+ SAR if enabled) images
        - dsm_batch : numpy.ndarray 
            Batch of DSM values
        - sem_batch : numpy.ndarray
            Batch of one-hot encoded semantic labels (if mtl_flag=True)
        - norm_batch : numpy.ndarray
            Batch of surface normal maps (if mtl_flag=True)
        - edge_batch : numpy.ndarray
            Batch of edge maps (if mtl_flag=True)
    Notes:
    -----
    - Input images can be normalized based on normalize_flag
    - For DFC2018, DSM is computed as difference between surface and terrain models
    - Batch size is controlled by mtl_global_batch_size for proper multi-GPU distribution
    - Patch size is controlled by cropSize global variable
    """
    rgb_batch = []
    dsm_batch = []
    sem_batch = []
    norm_batch = []
    edge_batch = []

    # Determine the actual batch size to generate based on multi-GPU configuration and model type
    # IMPORTANT: For multi-GPU, we generate global_batch_size samples total, which TensorFlow
    # will automatically distribute across GPUs (e.g., 8 samples → 2 per GPU on 4 GPUs)
    # For single GPU: generate per-GPU batch size samples
    if mtl_flag:
        # MTL training
        actual_batch_size = mtl_global_batch_size if multi_gpu_enabled else mtl_batchSize
    else:
        # DAE training
        actual_batch_size = dae_global_batch_size if multi_gpu_enabled else dae_batchSize

    # Select and preprocess a random input tile for batch random selection, if the input image is large
    if large_tile_mode:
        idx = random.randint(0, len(train_rgb) - 1)
        if dataset_name == 'Vaihingen':
            rgb_tile = np.array(Image.open(train_rgb[idx])); 
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(train_dsm[idx])); 
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile

            if mtl_flag:
                sem_tile = np.array(Image.open(train_sem[idx]))
                if norm_flag:
                    norm_tile = genNormals(dsm_tile); 
                    norm_tile = norm_tile if normalize_flag else (norm_tile * 255).astype(np.uint8)
                if edge_flag:
                    edge_tile = genEdgeMap(dsm_tile); 
                    edge_tile = normalize_array(edge_tile, 0, 1) if normalize_flag else edge_tile

        elif dataset_name == 'DFC2018':
            rgb_tile = np.array(Image.open(train_rgb[idx])); 
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(train_dsm[2 * idx]))
            dem_tile = np.array(Image.open(train_dsm[2 * idx + 1]))
            dsm_tile = correctTile(dsm_tile)
            dem_tile = correctTile(dem_tile)
            dsm_tile = dsm_tile - dem_tile  # Caution! nDSM here could still contain negative values
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile

            if mtl_flag:
                sem_tile = np.array(Image.open(train_sem[idx]))
                if norm_flag:
                    norm_tile = genNormals(dsm_tile); 
                    norm_tile = norm_tile if normalize_flag else (norm_tile * 255).astype(np.uint8)
                if edge_flag:
                    edge_tile = genEdgeMap(dsm_tile); 
                    edge_tile = normalize_array(edge_tile, 0, 1) if normalize_flag else edge_tile

    # Generate or select random patches - now using actual_batch_size for proper multi-GPU distribution
    for i in range(actual_batch_size):
        # Generate random patches if it is tile_mode_data. This is like data augmentation process
        if large_tile_mode:
            h = rgb_tile.shape[0]
            w = rgb_tile.shape[1]
            r = random.randint(0, h - cropSize)
            c = random.randint(0, w - cropSize)
            rgb = rgb_tile[r:r + cropSize, c:c + cropSize]
            dsm = dsm_tile[r:r + cropSize, c:c + cropSize]
            
            if mtl_flag:
                sem = sem_tile[r:r + cropSize, c:c + cropSize]
                if (dataset_name == 'DFC2018'): sem = sem[..., np.newaxis]
                if norm_flag:
                    norm = norm_tile[r:r + cropSize, c:c + cropSize]
                if edge_flag:
                    edge = edge_tile[r:r + cropSize, c:c + cropSize]
        else:
            # Choose batch items in order based on every dataset specifics
            # For non-large-tile datasets, we need to calculate the correct sample index
            # based on the actual batch size being used
            sample_idx = (iter - 1) * actual_batch_size + i
            
            # Handle all regular-sized datasets with standard folder structure
            rgb = np.array(Image.open(train_rgb[sample_idx]))
            rgb = normalize_array(rgb, 0, 1) if normalize_flag else rgb
            
            # Add SAR channels if available and enabled
            if sar_mode and any(dataset_name.startswith(d) for d in sar_datasets):
                sar = np.array(Image.open(train_sar[sample_idx]))
                sar = normalize_array(sar, 0, 1) if normalize_flag else sar
                rgb = np.dstack((rgb, sar))
            
            dsm = np.array(Image.open(train_dsm[sample_idx]))
            dsm = normalize_array(dsm, 0, 1) if normalize_flag else dsm
            
            if mtl_flag:
                sem = np.array(Image.open(train_sem[sample_idx]))
                if norm_flag:
                    norm = genNormals(dsm); 
                    norm = norm if normalize_flag else (norm * 255).astype(np.uint8)
                if edge_flag:
                    edge = genEdgeMap(dsm); 
                    edge = normalize_array(edge, 0, 1) if normalize_flag else edge

        rgb_batch.append(rgb)
        dsm_batch.append(dsm)
        if mtl_flag:
            sem_batch.append(sem_to_onehot(sem))
            if norm_flag:
                norm_batch.append(norm)
            if edge_flag:
                edge_batch.append(edge)

    rgb_batch = np.array(rgb_batch)
    dsm_batch = np.array(dsm_batch)[..., np.newaxis]
    if mtl_flag:
        sem_batch = np.array(sem_batch)
        if norm_flag:
            norm_batch = np.array(norm_batch)
        if edge_flag:
            edge_batch = np.array(edge_batch)[..., np.newaxis]
    
    return rgb_batch, dsm_batch, sem_batch, norm_batch, edge_batch


def load_test_tiles(test_rgb, test_sar, test_dsm, test_sem, tile):
    """
    Load and preprocess test tiles from different datasets.
    This function loads RGB, SAR (optional), DSM, and semantic segmentation tiles from specified datasets
    and applies normalization if required.
    Args:
        test_rgb (list): List of paths to RGB image tiles
        test_sar (list): List of paths to SAR image tiles (used only for DFC2023 datasets)
        test_dsm (list): List of paths to DSM (Digital Surface Model) tiles
        test_sem (list): List of paths to semantic segmentation tiles
        tile (int): Index of the tile to load
    Returns:
        tuple: Contains:
            - rgb_tile (numpy.ndarray): RGB image tile (with SAR channels appended for DFC2023 if sar_mode=True)
            - dsm_tile (numpy.ndarray): DSM tile (normalized if normalize_flag=True)
            - sem_tile (numpy.ndarray): Semantic segmentation tile
    Notes:
        - For DFC2018 dataset, DSM is calculated as the difference between DSM and DEM tiles
        - Function behavior depends on global variables:
            - dataset_name: Determines which dataset is being processed
            - normalize_flag: Controls whether to normalize the data
            - sar_mode: Controls whether to include SAR data for DFC2023 datasets
    """
    # Handle large tile datasets
    if dataset_name in large_tile_datasets:
        if dataset_name == 'Vaihingen':
            rgb_tile = np.array(Image.open(test_rgb[tile])); 
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(test_dsm[tile])); 
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
            sem_tile = np.array(Image.open(test_sem[tile]))

        elif dataset_name == 'DFC2018':
            rgb_tile = np.array(Image.open(test_rgb[tile])); 
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(test_dsm[2 * tile]))
            dem_tile = np.array(Image.open(test_dsm[2 * tile + 1]))
            dsm_tile = correctTile(dsm_tile)
            dem_tile = correctTile(dem_tile)
            dsm_tile = dsm_tile - dem_tile  # Caution! nDSM here could still contain negative values
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
            sem_tile = np.array(Image.open(test_sem[tile]))

    # Handle regular-sized datasets with unified approach
    else:
        rgb_tile = np.array(Image.open(test_rgb[tile])); 
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        
        # Add SAR channels if available and enabled
        if sar_mode and any(dataset_name.startswith(d) for d in sar_datasets):
            sar_tile = np.array(Image.open(test_sar[tile])); 
            sar_tile = normalize_array(sar_tile, 0, 1) if normalize_flag else sar_tile
            rgb_tile = np.dstack((rgb_tile, sar_tile))
        
        dsm_tile = np.array(Image.open(test_dsm[tile]))
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))
        
    return rgb_tile, dsm_tile, sem_tile


def sem_to_onehot(sem_tensor):
    """
    Converts a semantic tensor containing class identities to a one-hot encoded representation based on the specified dataset and semantic_label_map.

    Parameters:
    - sem_tensor (numpy.ndarray): The input semantic tensor containing semantic labels represented as a NumPy array.

    Returns:
    - numpy.ndarray: A one-hot encoded representation of the input RGB image.
    """
    num_classes = len(semantic_label_map)
    shape = sem_tensor.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for cls_idx, color in semantic_label_map.items():
        if uses_rgb_labels:
            encoded_image[:, :, cls_idx] = np.all(sem_tensor.reshape((-1, 3)) == color, axis=1).reshape(shape[:2])
        else:
            encoded_image[:, :, cls_idx] = np.all(sem_tensor.reshape((-1, 1)) == color, axis=1).reshape(shape[:2])

    return encoded_image


def convert_sem_onehot_to_annotation(sem_onehot):
    """
    Converts the softmax output (probability values) into the corresponding semantic annotation in the format of the ground truth.

    Parameters:
    - sem_onehot (numpy.ndarray): The softmax output of the semantic segmentation (probability values), shape (H, W, num_classes).

    Returns:
    - numpy.ndarray: The semantic annotation in the format of the ground truth, either RGB image or single-channel class labels.
    """ 
    # Step 1: Convert the softmax probabilities into class predictions (one-hot encoded format)
    class_predictions = np.argmax(sem_onehot, axis=-1)  # Shape: (H, W), class with highest probability
    
    # Step 2: Map the one-hot predictions back to the original semantic annotation format (RGB or class labels)
    H, W = class_predictions.shape
    if uses_rgb_labels:
        # Initialize the annotation tensor with the same shape as the input RGB (H, W, 3)
        sem_tensor = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_idx, color in semantic_label_map.items():
            mask = class_predictions == cls_idx
            sem_tensor[mask] = color  # Assign the RGB color to each class
    else:
        # Initialize the annotation tensor as a single-channel image with class IDs (H, W)
        sem_tensor = np.zeros((H, W), dtype=np.uint8)
        for cls_idx, color in semantic_label_map.items():
            sem_tensor[class_predictions == cls_idx] = color  # Assign class labels

    return sem_tensor


def genNormals(dsm_tile, mode='sobel'):
    """
    Generates normal vectors for a given DSM tile based on gradient calculations.

    Parameters:
    - dsm_tile (numpy.ndarray): The input DSM tile for which normals are to be generated.
    - mode (str): The mode of gradient calculation. Can be either 'gradient' or 'sobel'. Default is 'sobel'.

    Returns:
    - numpy.ndarray: The normalized tile with normal vectors.

    Raises:
    - ValueError: If the mode is neither 'gradient' nor 'sobel'.
    """
    # Validate the mode parameter
    if mode not in ['gradient', 'sobel']:
        raise ValueError("Mode must be either 'gradient' or 'sobel'.")

    # Calculate gradients based on the mode
    if mode == 'gradient':
        zy, zx = np.gradient(dsm_tile)
    elif mode == 'sobel':
        zx = cv2.Sobel(dsm_tile, cv2.CV_64F, 1, 0, ksize=5)
        zy = cv2.Sobel(dsm_tile, cv2.CV_64F, 0, 1, ksize=5)

    # Stack the gradients along the third dimension to form a 3D array
    norm_tile = np.dstack((-zx, -zy, np.ones_like(dsm_tile)))

    # Normalize the gradients
    n = np.linalg.norm(norm_tile, axis=2)
    norm_tile[:, :, 0] /= n
    norm_tile[:, :, 1] /= n
    norm_tile[:, :, 2] /= n

    # Adjust the normalization values
    norm_tile += 1
    norm_tile /= 2

    return norm_tile


def genEdgeMap(DSM, roof_height_threshold=roof_height_threshold, canny_lt=canny_lt, canny_ht=canny_ht):
    """
    Generates an edge map from a Digital Surface Model (DSM) image.
    
    Parameters:
    - DSM (numpy.ndarray): The input Digital Surface Model (DSM) image.
    - roof_height_threshold (float): Threshold value for identifying roof heights.
    - canny_lt (float): Lower threshold for Canny edge detection.
    - canny_ht (float): Higher threshold for Canny edge detection.
    
    Returns:
    - numpy.ndarray: An edge map generated from the input DSM image.
    """
    # Normalize DSM to range (0, 255) if not already and convert to uint8 for thresholding and edge detection
    if (DSM.min() >= 0 and DSM.max() <= 1) or DSM.min() < 0 or DSM.max() > 255:
        DSM = cv2.normalize(DSM, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        DSM = DSM.astype(np.uint8)

    # Find the roof height edges
    _, roof_height_edges = cv2.threshold(DSM, roof_height_threshold, 255, cv2.THRESH_BINARY)

    # Apply Gaussian smoothing to reduce noise
    edges_smoothed = cv2.GaussianBlur(roof_height_edges, (5, 5), 0)

    # Apply morphological operations to remove small contours
    # Adjust the kernel size and iterations as needed
    kernel = np.ones((3, 3), np.uint8)
    edges_cleaned = cv2.morphologyEx(edges_smoothed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply Canny edge detection to enhance the edges
    edges = cv2.Canny(edges_cleaned, canny_lt, canny_ht)

    return edges


def normalize_array(arr, min, max):
    """
    Normalizes an array by scaling its values to the range [min, max].
    
    Parameters:
    - arr (numpy.ndarray): The input array to be normalized.
    
    Returns:
    - numpy.ndarray: The normalized array.
    """
    # norm_arr = cv2.normalize(arr, None, min, max, cv2.NORM_MINMAX).astype(np.uint8)
    norm_arr = cv2.normalize(arr, None, min, max, cv2.NORM_MINMAX).astype('float32')
    norm_arr = np.clip(norm_arr, min, max)

    return norm_arr


def correctTile(tile):
    """
    Corrects the values in a tile array based on specified thresholds.
    This is usually the case for datasets with both DSM and DEM available.
    
    Parameters:
    - tile (numpy.ndarray): The input tile array.
    
    Returns:
    - numpy.ndarray: The corrected tile array.
    """
    tile[tile > 1000] = -123456
    tile[tile == -123456] = np.max(tile)
    tile[tile < -1000] = 123456
    tile[tile == 123456] = np.min(tile)

    return tile


def gaussian_kernel(width, height, sigma=0.2, mu=0.0):
    """
    Generates a Gaussian kernel for the Gaussian smoohing procedure applied on the estimated DSM outputs.
    
    Parameters:
    - width (int): Width of the kernel.
    - height (int): Height of the kernel.
    - sigma (float): Standard deviation of the Gaussian distribution.
    - mu (float): Mean of the Gaussian distribution.
    
    Returns:
    - numpy.ndarray: The Gaussian kernel.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))
    d = np.sqrt(x * x + y * y)
    gaussian_k = (np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))) / np.sqrt(2 * np.pi * sigma ** 2)

    return gaussian_k / gaussian_k.sum()


def sliding_window(image, step, window_size):
    """
    Iterates over an image with a sliding window, yielding coordinates for each window position.
    
    Parameters:
    - image (numpy.ndarray): The input image.
    - step (int): Step size for moving the window.
    - window_size (tuple): Size of the window (width, height).
    
    Yields:
    - tuple: Coordinates of the top-left and bottom-right corners of the current window.
    """
    height, width = (image.shape[0], image.shape[1])
    h, w = (window_size[0], window_size[1])
    for x in range(0, width - w + step, step):
        if x + w >= width:
            x = width - w
        for y in range(0, height - h + step, step):
            if y + h >= height:
                y = height - h
            yield x, x + w, y, y + h



def handle_early_stopping(
    early_stop_flag: bool,
    current_metric: float,
    best_metric: float,
    patience_counter: int,
    early_stop_patience: int,
    early_stop_delta: float,
    model: tf.keras.Model,
    checkpoint_path: str,
    epoch: int,
    logger: logging.Logger,
) -> Tuple[bool, float, int]:
    """
    Handle early stopping logic during model training.
    
    Args:
        early_stop_flag (bool): Whether early stopping is enabled
        current_metric (float): Current validation metric
        best_metric (float): Best validation metric so far
        patience_counter (int): Counter for patience
        early_stop_patience (int): Maximum patience before stopping
        early_stop_delta (float): Minimum improvement threshold
        model (tf.keras.Model): Model instance (MTL or DAE)
        checkpoint_path (str): Path to save checkpoints
        epoch (int): Current epoch number
        logger (logging.Logger): Logger instance
        
    Returns:
        Tuple: (should_stop, best_metric, patience_counter)
    """
    should_stop = False
    
    if early_stop_flag:
        # Check if metric improved based on whether lower or higher is better
        if eval_metric_lower_better:
            improved = current_metric < best_metric - early_stop_delta
        else:
            improved = current_metric > best_metric + early_stop_delta
            
        if improved:
            best_metric = current_metric
            patience_counter = 0
            logger.info(f'Validation {eval_metric.upper()} improved to {current_metric:.6f}')
            # Save best model
            model.save_weights(checkpoint_path)
        else:
            patience_counter += 1
            logger.info(f'No improvement in validation {eval_metric.upper()} for {patience_counter} epochs')
            
        # Check if we should stop
        if patience_counter >= early_stop_patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch} after {patience_counter} epochs without improvement!")
            should_stop = True
            
    return should_stop, best_metric, patience_counter

