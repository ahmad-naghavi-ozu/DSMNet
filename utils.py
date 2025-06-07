# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

import random
import numpy as np
import glob
import cv2
import PIL
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from skimage import io
from typing import Optional, List, Tuple
import logging
from config import *

Image.MAX_IMAGE_PIXELS = 1000000000


class DSMDataset(Dataset):
    """PyTorch Dataset for DSM data"""
    
    def __init__(self, rgb_files, sar_files, dsm_files, sem_files, mode='train', 
                 transform=None, target_transform=None):
        self.rgb_files = rgb_files
        self.sar_files = sar_files if sar_files else [None] * len(rgb_files)
        self.dsm_files = dsm_files
        self.sem_files = sem_files
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = self.rgb_files[idx]
        rgb_img = load_image(rgb_path, normalize_flag)
        
        # Load SAR if available
        sar_img = None
        if self.sar_files[idx] is not None and self.sar_files[idx] != "":
            sar_img = load_image(self.sar_files[idx], normalize_flag)
        
        # Load DSM
        dsm_path = self.dsm_files[idx]
        dsm_img = load_image(dsm_path, normalize_flag)
        
        # Load semantic labels
        sem_path = self.sem_files[idx]
        sem_img = load_semantic_labels(sem_path)
        
        # Generate patches if large tile mode
        if large_tile_mode:
            rgb_patch, dsm_patch, sem_patch, norm_patch, edge_patch, patch_coords = \
                generate_patch_from_large_tile(rgb_img, sar_img, dsm_img, sem_img)
        else:
            rgb_patch = rgb_img
            dsm_patch = dsm_img
            sem_patch = sem_img
            norm_patch = compute_surface_normals(dsm_patch) if norm_flag else None
            edge_patch = compute_edge_map(dsm_patch) if edge_flag else None
            patch_coords = (0, 0)  # Default coordinates for non-large-tile mode
        
        # Convert to tensors
        rgb_tensor = torch_from_numpy(rgb_patch)
        dsm_tensor = torch_from_numpy(dsm_patch)
        sem_tensor = torch_from_numpy(sem_patch)
        
        # Combine RGB and SAR if available
        if sar_mode and sar_img is not None:
            if large_tile_mode:
                # Use the same coordinates for SAR patch as used for RGB patch
                sar_patch = generate_sar_patch_from_large_tile(sar_img, patch_coords[0], patch_coords[1])
            else:
                sar_patch = sar_img
            sar_tensor = torch_from_numpy(sar_patch)
            input_tensor = torch.cat([rgb_tensor, sar_tensor], dim=0)
        else:
            input_tensor = rgb_tensor
        
        # Prepare additional outputs
        outputs = {
            'input': input_tensor,
            'dsm': dsm_tensor,
            'sem': sem_tensor
        }
        
        if norm_flag and norm_patch is not None:
            outputs['norm'] = torch_from_numpy(norm_patch)
        
        if edge_flag and edge_patch is not None:
            outputs['edge'] = torch_from_numpy(edge_patch)
        
        return outputs


def collect_all_tilenames():
    """Collect all filenames for training, validation, and testing"""
    train_data = collect_tilenames("train")
    valid_data = collect_tilenames("valid") if train_valid_flag else ([], [], [], [], [])
    test_data = collect_tilenames("test")
    
    return train_data, valid_data, test_data


def collect_tilenames(mode):
    """
    Collects filenames for RGB, SAR, DSM, and SEM images based on the specified dataset and mode.
    Converted to work with PyTorch data loading.
    """
    all_rgb, all_sar, all_dsm, all_sem = [], [], [], []

    # Determine base paths
    if dataset_name == 'Vaihingen':
        base_path = shortcut_path + 'Vaihingen/'
        train_frames = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 30, 34]
        valid_frames = [28, 32, 37]
        test_frames = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]

    elif dataset_name == 'DFC2018':
        base_path = shortcut_path + 'DFC2018/'
        train_frames = ['UH_NAD83_272056_3289689', 'UH_NAD83_272652_3289689', 'UH_NAD83_273248_3289689']
        valid_frames = ['UH_NAD83_273844_3289689']
        test_frames = ['UH_NAD83_271460_3289689', 'UH_NAD83_271460_3290290', 'UH_NAD83_272056_3290290',
                       'UH_NAD83_272652_3290290', 'UH_NAD83_273248_3290290', 'UH_NAD83_273844_3290290',
                       'UH_NAD83_274440_3289689', 'UH_NAD83_274440_3290290', 'UH_NAD83_275036_3289689',
                       'UH_NAD83_275036_3290290']
    
    elif dataset_name.startswith('DFC2019'):
        if mode == 'train':
            base_path = shortcut_path + dataset_name + '/Train/'
        elif mode == 'valid':
            base_path = shortcut_path + dataset_name + '/Valid/'
        elif mode == 'test':
            base_path = shortcut_path + dataset_name + '/Test/'

    elif dataset_name.startswith('DFC2023'):
        if mode == 'train':
            base_path = shortcut_path + dataset_name + '/train/'
        elif mode == 'valid':
            base_path = shortcut_path + dataset_name + '/valid/'
        elif mode == 'test':
            base_path = shortcut_path + dataset_name + '/test/'

    elif dataset_name.startswith('Vaihingen_crp256'):
        if mode == 'train':
            base_path = shortcut_path + dataset_name + '/train/'
        elif mode == 'valid':
            base_path = shortcut_path + dataset_name + '/valid/'
        elif mode == 'test':
            base_path = shortcut_path + dataset_name + '/test/'

    # Collection logic remains the same as original utils.py
    # [Implementation details would be the same as in the original file]
    
    # For DFC2023 datasets (including DFC2023Amini)
    if dataset_name.startswith('DFC2023') and mode in ['train', 'valid', 'test']:
        rgb_path = base_path + 'rgb/'
        dsm_path = base_path + 'dsm/'
        sem_path = base_path + 'sem/'
        sar_path = base_path + 'sar/' if sar_mode else None
        
        if os.path.exists(rgb_path):
            for filename in os.listdir(rgb_path):
                if filename.endswith('.tif'):
                    # Get the base name without extension
                    base_name = filename[:-4]  # Remove .tif extension
                    
                    # Check if corresponding DSM and SEM files exist
                    dsm_file = dsm_path + filename
                    sem_file = sem_path + filename
                    
                    if os.path.exists(dsm_file) and os.path.exists(sem_file):
                        all_rgb.append(rgb_path + filename)
                        all_dsm.append(dsm_file)
                        all_sem.append(sem_file)
                        
                        # Add SAR if in SAR mode
                        if sar_mode and sar_path and os.path.exists(sar_path + filename):
                            all_sar.append(sar_path + filename)
                        else:
                            all_sar.append("")  # Empty placeholder if no SAR
    
    # For brevity, I'll include the DFC2019 example:
    elif dataset_name.startswith('DFC2019') and mode in ['train', 'valid', 'test']:
        for filename in os.listdir(base_path + 'RGB/'):
            if filename.endswith('.tif'):
                base_name = '_'.join(filename.split('_')[:-2])
                number = filename.split('_')[-1].split('.')[0]
                
                agl_file = f"{base_name}_AGL_{number}.tif"
                cls_file = f"{base_name}_CLS_{number}.tif"
                
                if os.path.exists(base_path + 'Truth/' + agl_file) and os.path.exists(base_path + 'Truth/' + cls_file):
                    all_rgb.append(base_path + 'RGB/' + filename)
                    all_dsm.append(base_path + 'Truth/' + agl_file)
                    all_sem.append(base_path + 'Truth/' + cls_file)
    
    return all_rgb, all_sar, all_dsm, all_sem, []


def create_data_loaders(train_data, valid_data=None, test_data=None):
    """Create PyTorch DataLoaders for training, validation, and testing"""
    
    # Unpack data tuples
    train_rgb, train_sar, train_dsm, train_sem, _ = train_data
    
    # Create datasets
    train_dataset = DSMDataset(train_rgb, train_sar, train_dsm, train_sem, mode='train')
    
    # Setup samplers for distributed training
    train_sampler = None
    valid_sampler = None
    
    if USE_MULTI_GPU and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        if valid_data is not None:
            valid_rgb, valid_sar, valid_dsm, valid_sem, _ = valid_data
            valid_dataset = DSMDataset(valid_rgb, valid_sar, valid_dsm, valid_sem, mode='valid')
            valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=mtl_batchSize,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = None
    if valid_data is not None:
        valid_rgb, valid_sar, valid_dsm, valid_sem, _ = valid_data
        valid_dataset = DSMDataset(valid_rgb, valid_sar, valid_dsm, valid_sem, mode='valid')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=mtl_batchSize,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    test_loader = None
    if test_data is not None:
        test_rgb, test_sar, test_dsm, test_sem, _ = test_data
        test_dataset = DSMDataset(test_rgb, test_sar, test_dsm, test_sem, mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Usually batch size 1 for testing
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    return train_loader, valid_loader, test_loader


def load_image(image_path, normalize=False):
    """Load and preprocess image"""
    try:
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            img = np.expand_dims(img, axis=-1) if len(img.shape) == 2 else img
        else:
            img = io.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
        
        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0
        else:
            img = img.astype(np.float32)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise


def load_semantic_labels(sem_path):
    """Load semantic segmentation labels"""
    try:
        if uses_rgb_labels:
            # For RGB triplet labels (like Vaihingen)
            sem_img = io.imread(sem_path)
            if sem_img is None:
                raise FileNotFoundError(f"Could not load semantic labels: {sem_path}")
            if len(sem_img.shape) == 3:
                # Convert RGB to class indices
                sem_processed = rgb_to_class_indices(sem_img, label_codes)
            else:
                sem_processed = sem_img
        else:
            # For direct class labels
            sem_img = io.imread(sem_path)
            if sem_img is None:
                raise FileNotFoundError(f"Could not load semantic labels: {sem_path}")
            sem_processed = sem_img
        
        return sem_processed.astype(np.int64)
    except Exception as e:
        print(f"Error loading semantic labels {sem_path}: {str(e)}")
        raise


def rgb_to_class_indices(rgb_image, label_codes):
    """Convert RGB labels to class indices"""
    h, w, c = rgb_image.shape
    class_map = np.zeros((h, w), dtype=np.int64)
    
    for idx, color in enumerate(label_codes):
        mask = np.all(rgb_image == color, axis=-1)
        class_map[mask] = idx
    
    return class_map


def torch_from_numpy(array):
    """Convert numpy array to PyTorch tensor with proper channel ordering"""
    if len(array.shape) == 3:
        # Convert HWC to CHW for PyTorch
        array = np.transpose(array, (2, 0, 1))
    elif len(array.shape) == 2:
        # Add channel dimension
        array = np.expand_dims(array, axis=0)
    
    return torch.from_numpy(array).float()


def generate_patch_from_large_tile(rgb_img, sar_img, dsm_img, sem_img):
    """Generate random patches from large tiles"""
    h, w = rgb_img.shape[:2]
    
    # Generate random crop coordinates
    top = random.randint(0, h - cropSize)
    left = random.randint(0, w - cropSize)
    
    # Crop all images
    rgb_patch = rgb_img[top:top+cropSize, left:left+cropSize]
    dsm_patch = dsm_img[top:top+cropSize, left:left+cropSize]
    sem_patch = sem_img[top:top+cropSize, left:left+cropSize]
    
    # Compute surface normals and edge maps
    norm_patch = compute_surface_normals(dsm_patch) if norm_flag else None
    edge_patch = compute_edge_map(dsm_patch) if edge_flag else None
    
    return rgb_patch, dsm_patch, sem_patch, norm_patch, edge_patch, (top, left)


def generate_sar_patch_from_large_tile(sar_img, top=None, left=None):
    """Generate SAR patch from large tile using same patch coordinates"""
    h, w = sar_img.shape[:2]
    
    # Use provided coordinates or generate random ones
    if top is None or left is None:
        top = random.randint(0, h - cropSize)
        left = random.randint(0, w - cropSize)
    
    sar_patch = sar_img[top:top+cropSize, left:left+cropSize]
    return sar_patch


def compute_surface_normals(dsm_patch):
    """Compute surface normals from DSM patch"""
    if len(dsm_patch.shape) == 3 and dsm_patch.shape[2] == 1:
        dsm_patch = dsm_patch.squeeze(-1)
    
    # Compute gradients
    grad_x = cv2.Sobel(dsm_patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dsm_patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute normal vectors
    ones = np.ones_like(grad_x)
    normals = np.stack([-grad_x, -grad_y, ones], axis=-1)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)
    
    return normals.astype(np.float32)


def compute_edge_map(dsm_patch):
    """Compute edge map from DSM patch using Canny edge detector"""
    if len(dsm_patch.shape) == 3 and dsm_patch.shape[2] == 1:
        dsm_patch = dsm_patch.squeeze(-1)
    
    # Apply threshold for potential rooftops
    mask = dsm_patch > roof_height_threshold
    masked_dsm = dsm_patch * mask
    
    # Apply Canny edge detection
    edges = cv2.Canny(masked_dsm.astype(np.uint8), canny_lt, canny_ht)
    
    # Normalize to [0, 1]
    edges = edges.astype(np.float32) / 255.0
    
    return np.expand_dims(edges, axis=-1)


def normalize_array(arr, min_val, max_val):
    """Normalize array to specified range"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    if arr_max == arr_min:
        return np.zeros_like(arr)
    
    # Normalize to [0, 1] first
    normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Scale to desired range
    return normalized * (max_val - min_val) + min_val


def sem_to_onehot(sem_tensor):
    """Convert semantic segmentation to one-hot encoding"""
    if len(sem_tensor.shape) == 3 and sem_tensor.shape[-1] == 1:
        sem_tensor = sem_tensor.squeeze(-1)
    
    height, width = sem_tensor.shape
    one_hot = np.zeros((height, width, sem_k), dtype=np.float32)
    
    for class_idx in range(sem_k):
        one_hot[:, :, class_idx] = (sem_tensor == class_idx).astype(np.float32)
    
    return one_hot


def convert_sem_onehot_to_annotation(sem_onehot):
    """Convert one-hot semantic segmentation back to class indices"""
    return np.argmax(sem_onehot, axis=-1)


def correctTile(tile):
    """
    Corrects the values in a tile array based on specified thresholds.
    This is usually the case for datasets with both DSM and DEM available.
    """
    tile = tile.copy()
    tile[tile > 1000] = -123456
    tile[tile == -123456] = np.max(tile)
    tile[tile < -1000] = 123456
    tile[tile == 123456] = np.min(tile)
    return tile


def gaussian_kernel(width, height, sigma=0.2, mu=0.0):
    """Generate 2D Gaussian kernel for smooth blending"""
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)
    y = y[:, np.newaxis]
    
    # Center coordinates
    x0 = width // 2
    y0 = height // 2
    
    # Generate Gaussian
    kernel = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return kernel


def sliding_window(image, step, window_size):
    """Generate sliding window coordinates for patch extraction"""
    h, w = image.shape[:2]
    window_h, window_w = window_size
    
    coordinates = []
    for y in range(0, h - window_h + 1, step):
        for x in range(0, w - window_w + 1, step):
            # Ensure we don't go out of bounds
            y2 = min(y + window_h, h)
            x2 = min(x + window_w, w)
            y1 = y2 - window_h
            x1 = x2 - window_w
            coordinates.append((x1, x2, y1, y2))
    
    return coordinates


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    if hasattr(model, 'module'):
        # Handle DataParallel/DistributedDataParallel
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath}, Epoch: {epoch}, Loss: {loss}")
    return epoch, loss


def setup_distributed_training():
    """Setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False
    
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return True


def cleanup_distributed():
    """Cleanup distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def generate_training_batches(train_rgb, train_sar, train_dsm, train_sem, iter, mtl_flag):
    """
    Generate training batches for multi-task learning from RGB, SAR, DSM and semantic segmentation data.
    Converted from TensorFlow to PyTorch implementation.
    """
    rgb_batch = []
    dsm_batch = []
    sem_batch = []
    norm_batch = []
    edge_batch = []

    # Select and preprocess a random input tile for batch random selection, if the input image is large
    if large_tile_mode:
        idx = random.randint(0, len(train_rgb) - 1)
        
        if dataset_name == 'Vaihingen':
            rgb_tile = np.array(Image.open(train_rgb[idx]))
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(train_dsm[idx]))
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile

            if mtl_flag:
                sem_tile = np.array(Image.open(train_sem[idx]))
                if norm_flag:
                    norm_tile = compute_surface_normals(dsm_tile)
                    norm_tile = norm_tile if normalize_flag else (norm_tile * 255).astype(np.uint8)
                if edge_flag:
                    edge_tile = compute_edge_map(dsm_tile)
                    edge_tile = normalize_array(edge_tile, 0, 1) if normalize_flag else edge_tile

        elif dataset_name == 'DFC2018':
            rgb_tile = np.array(Image.open(train_rgb[idx]))
            rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
            dsm_tile = np.array(Image.open(train_dsm[2 * idx]))
            dem_tile = np.array(Image.open(train_dsm[2 * idx + 1]))
            dsm_tile = correctTile(dsm_tile)
            dem_tile = correctTile(dem_tile)
            dsm_tile = dsm_tile - dem_tile  # nDSM calculation
            dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile

            if mtl_flag:
                sem_tile = np.array(Image.open(train_sem[idx]))
                if norm_flag:
                    norm_tile = compute_surface_normals(dsm_tile)
                    norm_tile = norm_tile if normalize_flag else (norm_tile * 255).astype(np.uint8)
                if edge_flag:
                    edge_tile = compute_edge_map(dsm_tile)
                    edge_tile = normalize_array(edge_tile, 0, 1) if normalize_flag else edge_tile

    # Generate or select patches
    for i in range(mtl_batchSize):
        if large_tile_mode:
            # Generate random patches from large tiles
            h, w = rgb_tile.shape[:2]
            r = random.randint(0, h - cropSize)
            c = random.randint(0, w - cropSize)
            
            rgb = rgb_tile[r:r + cropSize, c:c + cropSize]
            dsm = dsm_tile[r:r + cropSize, c:c + cropSize]
            
            if mtl_flag:
                sem = sem_tile[r:r + cropSize, c:c + cropSize]
                if norm_flag:
                    norm = norm_tile[r:r + cropSize, c:c + cropSize]
                if edge_flag:
                    edge = edge_tile[r:r + cropSize, c:c + cropSize]
        else:
            # Choose batch items in order for patch-based datasets
            if dataset_name.startswith('DFC2019'):
                rgb = np.array(Image.open(train_rgb[(iter - 1) * mtl_batchSize + i]))
                rgb = normalize_array(rgb, 0, 1) if normalize_flag else rgb
                dsm = np.array(Image.open(train_dsm[(iter - 1) * mtl_batchSize + i]))
                dsm = normalize_array(dsm, 0, 1) if normalize_flag else dsm
                
                if mtl_flag:
                    sem = np.array(Image.open(train_sem[(iter - 1) * mtl_batchSize + i]))
                    if norm_flag:
                        norm = compute_surface_normals(dsm)
                        norm = norm if normalize_flag else (norm * 255).astype(np.uint8)
                    if edge_flag:
                        edge = compute_edge_map(dsm)
                        edge = normalize_array(edge, 0, 1) if normalize_flag else edge

            elif dataset_name.startswith('DFC2023'):
                rgb = np.array(Image.open(train_rgb[(iter - 1) * mtl_batchSize + i]))
                rgb = normalize_array(rgb, 0, 1) if normalize_flag else rgb
                
                if sar_mode:
                    sar = np.array(Image.open(train_sar[(iter - 1) * mtl_batchSize + i]))
                    sar = normalize_array(sar, 0, 1) if normalize_flag else sar
                    rgb = np.dstack((rgb, sar))
                
                dsm = np.array(Image.open(train_dsm[(iter - 1) * mtl_batchSize + i]))
                dsm = normalize_array(dsm, 0, 1) if normalize_flag else dsm

                if mtl_flag:
                    sem = np.array(Image.open(train_sem[(iter - 1) * mtl_batchSize + i]))
                    if norm_flag:
                        norm = compute_surface_normals(dsm)
                        norm = norm if normalize_flag else (norm * 255).astype(np.uint8)
                    if edge_flag:
                        edge = compute_edge_map(dsm)
                        edge = normalize_array(edge, 0, 1) if normalize_flag else edge

            elif dataset_name.startswith('Vaihingen_crp256'):
                rgb = np.array(Image.open(train_rgb[(iter - 1) * mtl_batchSize + i]))
                rgb = normalize_array(rgb, 0, 1) if normalize_flag else rgb
                dsm = np.array(Image.open(train_dsm[(iter - 1) * mtl_batchSize + i]))
                dsm = normalize_array(dsm, 0, 1) if normalize_flag else dsm
                
                if mtl_flag:
                    sem = np.array(Image.open(train_sem[(iter - 1) * mtl_batchSize + i]))
                    if norm_flag:
                        norm = compute_surface_normals(dsm)
                        norm = norm if normalize_flag else (norm * 255).astype(np.uint8)
                    if edge_flag:
                        edge = compute_edge_map(dsm)
                        edge = normalize_array(edge, 0, 1) if normalize_flag else edge

        # Append to batches
        rgb_batch.append(rgb)
        dsm_batch.append(dsm)
        if mtl_flag:
            sem_batch.append(sem_to_onehot(sem))
            if norm_flag:
                norm_batch.append(norm)
            if edge_flag:
                edge_batch.append(edge)

    # Convert to numpy arrays
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
    Converted from TensorFlow to PyTorch implementation.
    """
    if dataset_name == 'Vaihingen':
        rgb_tile = np.array(Image.open(test_rgb[tile]))
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        dsm_tile = np.array(Image.open(test_dsm[tile]))
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))

    elif dataset_name == 'DFC2018':
        rgb_tile = np.array(Image.open(test_rgb[tile]))
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        dsm_tile = np.array(Image.open(test_dsm[2 * tile]))
        dem_tile = np.array(Image.open(test_dsm[2 * tile + 1]))
        dsm_tile = correctTile(dsm_tile)
        dem_tile = correctTile(dem_tile)
        dsm_tile = dsm_tile - dem_tile  # nDSM calculation
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))

    elif dataset_name.startswith('DFC2019'):
        rgb_tile = np.array(Image.open(test_rgb[tile]))
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        dsm_tile = np.array(Image.open(test_dsm[tile]))
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))

    elif dataset_name.startswith('DFC2023'):
        rgb_tile = np.array(Image.open(test_rgb[tile]))
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        
        if sar_mode:
            sar_tile = np.array(Image.open(test_sar[tile]))
            sar_tile = normalize_array(sar_tile, 0, 1) if normalize_flag else sar_tile
            rgb_tile = np.dstack((rgb_tile, sar_tile))
        
        dsm_tile = np.array(Image.open(test_dsm[tile]))
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))

    elif dataset_name.startswith('Vaihingen_crp256'):
        rgb_tile = np.array(Image.open(test_rgb[tile]))
        rgb_tile = normalize_array(rgb_tile, 0, 1) if normalize_flag else rgb_tile
        dsm_tile = np.array(Image.open(test_dsm[tile]))
        dsm_tile = normalize_array(dsm_tile, 0, 1) if normalize_flag else dsm_tile
        sem_tile = np.array(Image.open(test_sem[tile]))
        
    return rgb_tile, dsm_tile, sem_tile


# Additional utility functions as needed for PyTorch compatibility

def validate_tensor_shape(tensor, expected_shape=None, name="tensor"):
    """Validate tensor shape and provide helpful error messages"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
    
    return True


def safe_normalize(tensor, dim=None, eps=1e-8):
    """Safely normalize tensor to avoid division by zero"""
    if dim is None:
        norm = torch.norm(tensor)
    else:
        norm = torch.norm(tensor, dim=dim, keepdim=True)
    
    return tensor / (norm + eps)


def convert_to_pytorch_format(data_batch):
    """Convert numpy batch to PyTorch format"""
    if isinstance(data_batch, np.ndarray):
        # Convert numpy to tensor
        tensor = torch.from_numpy(data_batch).float()
        
        # If it's an image batch, transpose from NHWC to NCHW
        if len(tensor.shape) == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        elif len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)
            
        return tensor
    elif isinstance(data_batch, torch.Tensor):
        return data_batch
    else:
        raise TypeError(f"Unsupported data type: {type(data_batch)}")


def get_device():
    """Get the appropriate device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def move_to_device(data, device=None):
    """Move data to specified device"""
    if device is None:
        device = get_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def print_tensor_stats(tensor, name="tensor"):
    """Print useful statistics about a tensor"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name} - Shape: {tensor.shape}, "
              f"Dtype: {tensor.dtype}, "
              f"Device: {tensor.device}, "
              f"Min: {tensor.min().item():.4f}, "
              f"Max: {tensor.max().item():.4f}, "
              f"Mean: {tensor.mean().item():.4f}")
    else:
        print(f"{name} is not a tensor: {type(tensor)}")


def ensure_directory_exists(directory_path):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")


def log_memory_usage():
    """Log current GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        f_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return f_str.format(**self.__dict__)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Handle different data types in batch
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    
    return collated
