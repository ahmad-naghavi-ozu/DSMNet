# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

from config import *  # Must come first to make metric_names available

import numpy as np
import random
import logging
import os
import datetime
import time
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nets import *
from utils import *
from metrics import *  # Import the metrics module
from test_dsm import test_dsm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(mtl_log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def main():
    # Initialize distributed training if using multiple GPUs
    distributed = setup_distributed_training() if USE_MULTI_GPU else False
    
    # Setup device
    if distributed:
        device = torch.device(f'cuda:{torch.distributed.get_rank()}')
        torch.cuda.set_device(device)
    else:
        device = setup_device()
    
    # Keep track of computation time
    total_start = time.time()
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'\nMTL PyTorch training on {dataset_name} started at {current_datetime}!\n')
    
    # Collect training and validation data
    train_data = collect_tilenames("train")
    if train_valid_flag:
        valid_data = collect_tilenames("valid")
    else:
        # Merge train and valid sets if no validation
        valid_data = collect_tilenames("valid")
        train_rgb, train_sar, train_dsm, train_sem, _ = train_data
        valid_rgb, valid_sar, valid_dsm, valid_sem, _ = valid_data
        train_rgb.extend(valid_rgb)
        train_sar.extend(valid_sar)
        train_dsm.extend(valid_dsm)
        train_sem.extend(valid_sem)
        train_data = (train_rgb, train_sar, train_dsm, train_sem, [])
        valid_data = None
    
    # Create data loaders
    train_loader, valid_loader, _ = create_data_loaders(train_data, valid_data)
    
    NUM_TRAIN_IMAGES = len(train_data[0])
    logger.info(f"Number of training samples: {NUM_TRAIN_IMAGES}\\n")
    
    # Create model
    input_channels = 4 if sar_mode else 3
    backbone_net = create_densenet_backbone(input_channels)
    mtl_model = MTL(backbone_net, sem_flag=sem_flag, norm_flag=norm_flag, edge_flag=edge_flag)
    mtl_model = mtl_model.to(device)
    
    # Load pretrained weights if specified
    if mtl_preload and os.path.exists(predCheckPointPath + '.pth'):
        checkpoint = torch.load(predCheckPointPath + '.pth', map_location=device)
        mtl_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded pretrained MTL weights")
    
    # Freeze backbone if specified
    if mtl_bb_freeze:
        for param in backbone_net.parameters():
            param.requires_grad = False
        logger.info("Backbone weights frozen")
    
    # Setup model for multi-GPU training
    if distributed:
        mtl_model = DDP(mtl_model, device_ids=[device])
    elif USE_MULTI_GPU and len(GPU_IDS) > 1:
        mtl_model = nn.DataParallel(mtl_model, device_ids=GPU_IDS)
    
    # Define loss functions
    if reg_loss == 'mse':
        dsm_loss_fn = nn.MSELoss()
    elif reg_loss == 'huber':
        dsm_loss_fn = nn.HuberLoss(delta=huber_delta)
    
    sem_loss_fn = nn.CrossEntropyLoss()
    norm_loss_fn = nn.MSELoss()
    edge_loss_fn = nn.MSELoss()
    
    # Setup optimizer
    optimizer = optim.Adam(mtl_model.parameters(), lr=mtl_lr, betas=(0.9, 0.999))
    
    # Initialize metrics tracking
    if train_valid_flag:
        all_metrics = metric_names.copy()
        for metric in segmentation_scalar_metrics:
            if metric not in all_metrics:
                all_metrics.append(metric)
        for metric in segmentation_class_metrics:
            for class_idx in range(len(semantic_label_map)):
                all_metrics.append(f"{metric}_class{class_idx}")
        
        valid_metrics = {metric: [] for metric in all_metrics}
        if plot_train_error:
            train_metrics = {metric: [] for metric in all_metrics}
        
        fig, axes, lines = plot_train_valid_metrics(
            epoch=0,
            train_metrics={metric: [] for metric in plot_metrics} if plot_train_error else None,
            valid_metrics={metric: [] for metric in plot_metrics},
            plot_train=plot_train_error,
            model_type='MTL'
        )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, mtl_numEpochs + 1):
        epoch_start = time.time()
        logger.info(f'\\nepoch {epoch}/{mtl_numEpochs} started!\\n')
        
        # Update learning rate if decay is enabled
        if mtl_lr_decay and epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            logger.info(f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")
        
        # Training phase
        mtl_model.train()
        train_losses = train_epoch(mtl_model, train_loader, optimizer, 
                                 dsm_loss_fn, sem_loss_fn, norm_loss_fn, edge_loss_fn,
                                 device, epoch, logger)
        
        # Validation phase
        if train_valid_flag and valid_loader is not None:
            mtl_model.eval()
            with torch.no_grad():
                val_losses, val_metrics_dict = validate_epoch(mtl_model, valid_loader,
                                                            dsm_loss_fn, sem_loss_fn, norm_loss_fn, edge_loss_fn,
                                                            device, epoch, logger)
            
            # Update metrics tracking
            for metric_name, value in val_metrics_dict.items():
                if metric_name in valid_metrics:
                    valid_metrics[metric_name].append(value)
            
            # Early stopping check
            current_metric = val_metrics_dict.get(eval_metric, float('inf'))
            if eval_metric_lower_better:
                improved = current_metric < best_loss - early_stop_delta
            else:
                improved = current_metric > best_loss + early_stop_delta
            
            if improved:
                best_loss = current_metric
                patience_counter = 0
                # Save best model
                save_checkpoint(mtl_model, optimizer, epoch, train_losses['total'], 
                              f"{predCheckPointPath}_best.pth")
            else:
                patience_counter += 1
            
            logger.info(f"Best {eval_metric}: {best_loss:.6f}, Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping
            if early_stop_flag and patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
            save_checkpoint(mtl_model, optimizer, epoch, train_losses['total'], 
                          f"{predCheckPointPath}_epoch_{epoch}.pth")
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
        
        # Update plots if validation is enabled
        if train_valid_flag:
            plot_train_valid_metrics(
                epoch=epoch,
                train_metrics=train_metrics if plot_train_error else None,
                valid_metrics=valid_metrics,
                plot_train=plot_train_error,
                model_type='MTL',
                fig=fig, axes=axes, lines=lines
            )
    
    # Final save
    save_checkpoint(mtl_model, optimizer, epoch, train_losses['total'], 
                  f"{predCheckPointPath}_final.pth")
    
    total_time = time.time() - total_start
    logger.info(f"\\nTraining completed in {total_time:.2f} seconds")
    
    # Cleanup distributed training
    if distributed:
        cleanup_distributed()


def train_epoch(model, data_loader, optimizer, dsm_loss_fn, sem_loss_fn, norm_loss_fn, edge_loss_fn, 
                device, epoch, logger):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    dsm_loss_sum = 0.0
    sem_loss_sum = 0.0
    norm_loss_sum = 0.0
    edge_loss_sum = 0.0
    rmse_sum = 0.0
    
    num_batches = len(data_loader)
    
    for batch_idx, batch_data in enumerate(data_loader):
        # Move data to device
        inputs = batch_data['input'].to(device)
        dsm_targets = batch_data['dsm'].to(device)
        sem_targets = batch_data['sem'].to(device)
        
        # Optional targets
        norm_targets = batch_data.get('norm', None)
        edge_targets = batch_data.get('edge', None)
        if norm_targets is not None:
            norm_targets = norm_targets.to(device)
        if edge_targets is not None:
            edge_targets = edge_targets.to(device)
        
        # Forward pass
        dsm_out, sem_out, norm_out, edge_out = model(inputs, mtl_head_mode, True)
        
        # Compute losses
        L1 = dsm_loss_fn(dsm_out.squeeze(), dsm_targets.squeeze())
        
        # RMSE computation for logging
        with torch.no_grad():
            mse_per_sample = F.mse_loss(dsm_out.squeeze(), dsm_targets.squeeze(), reduction='none')
            if len(mse_per_sample.shape) > 1:
                mse_per_sample = mse_per_sample.view(mse_per_sample.size(0), -1).mean(dim=1)
            rmse_per_sample = torch.sqrt(mse_per_sample)
            batch_rmse = rmse_per_sample.mean()
        
        L2 = sem_loss_fn(sem_out, sem_targets.squeeze().long()) if sem_flag and sem_out is not None else torch.tensor(0.0, device=device)
        L3 = norm_loss_fn(norm_out, norm_targets) if norm_flag and norm_out is not None and norm_targets is not None else torch.tensor(0.0, device=device)
        L4 = edge_loss_fn(edge_out.squeeze(), edge_targets.squeeze()) if edge_flag and edge_out is not None and edge_targets is not None else torch.tensor(0.0, device=device)
        
        # Check for NaN values
        if torch.isnan(L1) or torch.isnan(L2) or torch.isnan(L3) or torch.isnan(L4):
            logger.error(f"NaN detected in loss values at batch {batch_idx}")
            logger.error(f"L1: {L1}, L2: {L2}, L3: {L3}, L4: {L4}")
            raise ValueError("Training halted due to NaN values")
        
        # Total weighted loss
        loss = w1 * L1 + w2 * L2 + w3 * L3 + w4 * L4
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        dsm_loss_sum += L1.item()
        sem_loss_sum += L2.item()
        norm_loss_sum += L3.item()
        edge_loss_sum += L4.item()
        rmse_sum += batch_rmse.item()
        
        # Log progress - handle small datasets safely
        log_freq = max(1, num_batches // 5)  # Ensure log_freq is at least 1
        if batch_idx % log_freq == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, '
                       f'Loss: {loss.item():.6f}, RMSE: {batch_rmse.item():.6f}')
    
    # Return average losses
    return {
        'total': total_loss / num_batches,
        'dsm': dsm_loss_sum / num_batches,
        'sem': sem_loss_sum / num_batches,
        'norm': norm_loss_sum / num_batches,
        'edge': edge_loss_sum / num_batches,
        'rmse': rmse_sum / num_batches
    }


def validate_epoch(model, data_loader, dsm_loss_fn, sem_loss_fn, norm_loss_fn, edge_loss_fn, 
                   device, epoch, logger):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    dsm_loss_sum = 0.0
    sem_loss_sum = 0.0
    norm_loss_sum = 0.0
    edge_loss_sum = 0.0
    
    # Metrics accumulators
    all_dsm_preds = []
    all_dsm_targets = []
    all_sem_preds = []
    all_sem_targets = []
    
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # Move data to device
            inputs = batch_data['input'].to(device)
            dsm_targets = batch_data['dsm'].to(device)
            sem_targets = batch_data['sem'].to(device)
            
            norm_targets = batch_data.get('norm', None)
            edge_targets = batch_data.get('edge', None)
            if norm_targets is not None:
                norm_targets = norm_targets.to(device)
            if edge_targets is not None:
                edge_targets = edge_targets.to(device)
            
            # Forward pass
            dsm_out, sem_out, norm_out, edge_out = model(inputs, mtl_head_mode, False)
            
            # Compute losses
            L1 = dsm_loss_fn(dsm_out.squeeze(), dsm_targets.squeeze())
            L2 = sem_loss_fn(sem_out, sem_targets.squeeze().long()) if sem_flag and sem_out is not None else torch.tensor(0.0, device=device)
            L3 = norm_loss_fn(norm_out, norm_targets) if norm_flag and norm_out is not None and norm_targets is not None else torch.tensor(0.0, device=device)
            L4 = edge_loss_fn(edge_out.squeeze(), edge_targets.squeeze()) if edge_flag and edge_out is not None and edge_targets is not None else torch.tensor(0.0, device=device)
            
            loss = w1 * L1 + w2 * L2 + w3 * L3 + w4 * L4
            
            # Accumulate losses
            total_loss += loss.item()
            dsm_loss_sum += L1.item()
            sem_loss_sum += L2.item()
            norm_loss_sum += L3.item()
            edge_loss_sum += L4.item()
            
            # Collect predictions for metrics
            all_dsm_preds.append(dsm_out.cpu())
            all_dsm_targets.append(dsm_targets.cpu())
            if sem_out is not None:
                all_sem_preds.append(sem_out.cpu())
                all_sem_targets.append(sem_targets.cpu())
    
    # Compute metrics
    dsm_preds = torch.cat(all_dsm_preds, dim=0)
    dsm_targets = torch.cat(all_dsm_targets, dim=0)
    
    # Compute height metrics
    metrics_dict = compute_height_metrics_pytorch(dsm_preds, dsm_targets)
    
    # Compute segmentation metrics if available
    if all_sem_preds:
        sem_preds = torch.cat(all_sem_preds, dim=0)
        sem_targets = torch.cat(all_sem_targets, dim=0)
        seg_metrics = compute_segmentation_metrics_pytorch(sem_preds, sem_targets)
        metrics_dict.update(seg_metrics)
    
    # Add loss metrics
    metrics_dict.update({
        'total_loss': total_loss / num_batches,
        'dsm_loss': dsm_loss_sum / num_batches,
        'sem_loss': sem_loss_sum / num_batches,
        'norm_loss': norm_loss_sum / num_batches,
        'edge_loss': edge_loss_sum / num_batches
    })
    
    logger.info(f"Validation Epoch {epoch} - RMSE: {metrics_dict.get('rmse', 0):.6f}, "
               f"Total Loss: {metrics_dict['total_loss']:.6f}")
    
    return {
        'total': total_loss / num_batches,
        'dsm': dsm_loss_sum / num_batches,
        'sem': sem_loss_sum / num_batches,
        'norm': norm_loss_sum / num_batches,
        'edge': edge_loss_sum / num_batches
    }, metrics_dict


if __name__ == "__main__":
    main()
