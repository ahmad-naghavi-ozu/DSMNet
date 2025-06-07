# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

from config import *
import numpy as np
import random
import logging
import os
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nets import UNet, setup_model_for_multi_gpu
from utils import *
from metrics import *

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(dae_log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


class DAEDataset(torch.utils.data.Dataset):
    """Dataset for Denoising AutoEncoder training"""
    
    def __init__(self, dsm_files, noise_std=0.1):
        self.dsm_files = dsm_files
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.dsm_files)
    
    def __getitem__(self, idx):
        # Load clean DSM
        dsm_path = self.dsm_files[idx]
        clean_dsm = load_image(dsm_path, normalize_flag)
        
        # Add noise to create noisy version
        noise = np.random.normal(0, self.noise_std, clean_dsm.shape).astype(np.float32)
        noisy_dsm = clean_dsm + noise
        
        # Convert to tensors
        clean_tensor = torch.from_numpy(clean_dsm).unsqueeze(0).float()  # Add channel dimension
        noisy_tensor = torch.from_numpy(noisy_dsm).unsqueeze(0).float()  # Add channel dimension
        
        return {
            'clean': clean_tensor,
            'noisy': noisy_tensor
        }
        
        # Convert to tensors
        clean_tensor = torch_from_numpy(clean_dsm)
        noisy_tensor = torch_from_numpy(noisy_dsm)
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor
        }


def create_dae_data_loaders(train_dsm_files, valid_dsm_files=None):
    """Create data loaders for DAE training"""
    
    # Create datasets
    train_dataset = DAEDataset(train_dsm_files)
    
    # Setup samplers for distributed training
    train_sampler = None
    valid_sampler = None
    
    if USE_MULTI_GPU and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        if valid_dsm_files is not None:
            valid_dataset = DAEDataset(valid_dsm_files)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=dae_batchSize,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = None
    if valid_dsm_files is not None:
        valid_dataset = DAEDataset(valid_dsm_files)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=dae_batchSize,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True
        )
    
    return train_loader, valid_loader


def main():
    """Main DAE training function"""
    
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
    logger.info(f'\\nDAE PyTorch training on {dataset_name} started at {current_datetime}!\\n')
    
    # Collect training and validation DSM data
    train_data = collect_tilenames("train")
    train_dsm_files = train_data[2]  # DSM files are at index 2
    
    valid_dsm_files = None
    if train_valid_flag:
        valid_data = collect_tilenames("valid")
        valid_dsm_files = valid_data[2]
    else:
        # Merge train and valid sets if no validation
        valid_data = collect_tilenames("valid")
        train_dsm_files.extend(valid_data[2])
    
    # Create data loaders
    train_loader, valid_loader = create_dae_data_loaders(train_dsm_files, valid_dsm_files)
    
    NUM_TRAIN_IMAGES = len(train_dsm_files)
    logger.info(f"Number of training DSM samples: {NUM_TRAIN_IMAGES}\\n")
    
    # Create U-Net model for denoising
    unet_model = UNet(in_channels=1, out_channels=1)
    unet_model = unet_model.to(device)
    
    # Load pretrained weights if specified
    if dae_preload and os.path.exists(corrCheckPointPath + '.pth'):
        checkpoint = torch.load(corrCheckPointPath + '.pth', map_location=device)
        unet_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded pretrained DAE weights")
    
    # Setup model for multi-GPU training
    if distributed:
        unet_model = DDP(unet_model, device_ids=[device])
    elif USE_MULTI_GPU and len(GPU_IDS) > 1:
        unet_model = nn.DataParallel(unet_model, device_ids=GPU_IDS)
    
    # Define loss function
    if reg_loss == 'mse':
        loss_fn = nn.MSELoss()
    elif reg_loss == 'huber':
        loss_fn = nn.HuberLoss(delta=huber_delta)
    
    # Setup optimizer
    optimizer = optim.Adam(unet_model.parameters(), lr=dae_lr, betas=(0.9, 0.999))
    
    # Initialize metrics tracking
    if train_valid_flag:
        valid_metrics = {metric: [] for metric in metric_names}
        if plot_train_error:
            train_metrics = {metric: [] for metric in metric_names}
        
        fig, axes, lines = plot_train_valid_metrics(
            epoch=0,
            train_metrics={metric: [] for metric in plot_metrics} if plot_train_error else None,
            valid_metrics={metric: [] for metric in plot_metrics},
            plot_train=plot_train_error,
            model_type='DAE'
        )
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, dae_numEpochs + 1):
        epoch_start = time.time()
        logger.info(f'\\nepoch {epoch}/{dae_numEpochs} started!\\n')
        
        # Update learning rate if decay is enabled
        if dae_lr_decay and epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            logger.info(f"Learning rate updated to: {optimizer.param_groups[0]['lr']}")
        
        # Training phase
        unet_model.train()
        train_losses = train_dae_epoch(unet_model, train_loader, optimizer, 
                                     loss_fn, device, epoch, logger)
        
        # Validation phase
        if train_valid_flag and valid_loader is not None:
            unet_model.eval()
            with torch.no_grad():
                val_losses, val_metrics_dict = validate_dae_epoch(unet_model, valid_loader,
                                                                loss_fn, device, epoch, logger)
            
            # Update metrics tracking
            for metric_name, value in val_metrics_dict.items():
                if metric_name in valid_metrics:
                    valid_metrics[metric_name].append(value)
            
            # Early stopping check
            current_metric = val_metrics_dict.get(eval_metric, val_losses['total'])
            if eval_metric_lower_better:
                improved = current_metric < best_loss - early_stop_delta
            else:
                improved = current_metric > best_loss + early_stop_delta
            
            if improved:
                best_loss = current_metric
                patience_counter = 0
                # Save best model
                save_checkpoint(unet_model, optimizer, epoch, train_losses['total'], 
                              f"{corrCheckPointPath}_best.pth")
            else:
                patience_counter += 1
            
            logger.info(f"Best {eval_metric}: {best_loss:.6f}, Patience: {patience_counter}/{early_stop_patience}")
            
            # Early stopping
            if early_stop_flag and patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save checkpoint periodically
        if epoch % 10 == 0:
            save_checkpoint(unet_model, optimizer, epoch, train_losses['total'], 
                          f"{corrCheckPointPath}_epoch_{epoch}.pth")
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
        
        # Update plots if validation is enabled
        if train_valid_flag:
            plot_train_valid_metrics(
                epoch=epoch,
                train_metrics=train_metrics if plot_train_error else None,
                valid_metrics=valid_metrics,
                plot_train=plot_train_error,
                model_type='DAE',
                fig=fig, axes=axes, lines=lines
            )
    
    # Final save
    save_checkpoint(unet_model, optimizer, epoch, train_losses['total'], 
                  f"{corrCheckPointPath}_final.pth")
    
    total_time = time.time() - total_start
    logger.info(f"\\nDAE training completed in {total_time:.2f} seconds")
    
    # Cleanup distributed training
    if distributed:
        cleanup_distributed()


def train_dae_epoch(model, data_loader, optimizer, loss_fn, device, epoch, logger):
    """Train DAE for one epoch"""
    model.train()
    
    total_loss = 0.0
    rmse_sum = 0.0
    mae_sum = 0.0
    
    num_batches = len(data_loader)
    
    for batch_idx, batch_data in enumerate(data_loader):
        # Move data to device
        noisy_inputs = batch_data['noisy'].to(device)
        clean_targets = batch_data['clean'].to(device)
        
        # Forward pass
        denoised_outputs = model(noisy_inputs)
        
        # Compute loss
        loss = loss_fn(denoised_outputs, clean_targets)
        
        # Compute additional metrics
        with torch.no_grad():
            mse = nn.MSELoss()(denoised_outputs, clean_targets)
            rmse = torch.sqrt(mse)
            mae = nn.L1Loss()(denoised_outputs, clean_targets)
        
        # Check for NaN values
        if torch.isnan(loss):
            logger.error(f"NaN detected in loss at batch {batch_idx}")
            raise ValueError("Training halted due to NaN values")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        rmse_sum += rmse.item()
        mae_sum += mae.item()
        
        # Log progress - handle small datasets safely
        log_freq = max(1, num_batches // 5)  # Ensure log_freq is at least 1
        if batch_idx % log_freq == 0:
            logger.info(f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, '
                       f'Loss: {loss.item():.6f}, RMSE: {rmse.item():.6f}')
    
    # Return average losses
    return {
        'total': total_loss / num_batches,
        'rmse': rmse_sum / num_batches,
        'mae': mae_sum / num_batches
    }


def validate_dae_epoch(model, data_loader, loss_fn, device, epoch, logger):
    """Validate DAE for one epoch"""
    model.eval()
    
    total_loss = 0.0
    
    # Metrics accumulators
    all_preds = []
    all_targets = []
    
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            # Move data to device
            noisy_inputs = batch_data['noisy'].to(device)
            clean_targets = batch_data['clean'].to(device)
            
            # Forward pass
            denoised_outputs = model(noisy_inputs)
            
            # Compute loss
            loss = loss_fn(denoised_outputs, clean_targets)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Collect predictions for metrics
            all_preds.append(denoised_outputs.cpu())
            all_targets.append(clean_targets.cpu())
    
    # Compute metrics
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute height metrics
    metrics_dict = compute_height_metrics_pytorch(preds, targets)
    
    # Add loss metric
    metrics_dict['total_loss'] = total_loss / num_batches
    
    logger.info(f"DAE Validation Epoch {epoch} - RMSE: {metrics_dict.get('rmse', 0):.6f}, "
               f"Total Loss: {metrics_dict['total_loss']:.6f}")
    
    return {
        'total': total_loss / num_batches
    }, metrics_dict


if __name__ == "__main__":
    main()
