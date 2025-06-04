# MAHDI ELHOUSNI, WPI 2020
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

from config import *
import numpy as np
import os
import time
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets import *
from utils import *
from metrics import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io


def test_dsm(mtl_model=None, dae_model=None, save_predictions=True, save_visualizations=True):
    """
    Test the DSM estimation model on test data
    
    Args:
        mtl_model: Trained MTL model (optional, will load if None)
        dae_model: Trained DAE model for correction (optional)
        save_predictions: Whether to save prediction images
        save_visualizations: Whether to save comparison visualizations
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    
    # Setup device
    device = setup_device()
    
    # Setup logging
    logger = logging.getLogger()
    logger.info(f"\\n{'='*60}")
    logger.info(f"STARTING DSM TESTING ON {dataset_name}")
    logger.info(f"{'='*60}")
    
    # Collect test data
    test_data = collect_tilenames("test")
    test_rgb, test_sar, test_dsm, test_sem, _ = test_data
    
    if not test_rgb:
        logger.warning("No test data found!")
        return {}
    
    logger.info(f"Number of test samples: {len(test_rgb)}")
    
    # Load MTL model if not provided
    if mtl_model is None:
        logger.info("Loading MTL model...")
        input_channels = 4 if sar_mode else 3
        backbone_net = create_densenet_backbone(input_channels)
        mtl_model = MTL(backbone_net, sem_flag=sem_flag, norm_flag=norm_flag, edge_flag=edge_flag)
        
        # Load trained weights
        checkpoint_path = f"{predCheckPointPath}_best.pth"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"{predCheckPointPath}_final.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if hasattr(mtl_model, 'module'):
                mtl_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                mtl_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded MTL weights from {checkpoint_path}")
        else:
            logger.warning(f"No MTL checkpoint found at {checkpoint_path}")
            return {}
    
    mtl_model = mtl_model.to(device)
    mtl_model.eval()
    
    # Load DAE model if correction is enabled
    if correction and dae_model is None:
        logger.info("Loading DAE model for correction...")
        dae_model = UNet(in_channels=1, out_channels=1)
        
        checkpoint_path = f"{corrCheckPointPath}_best.pth"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f"{corrCheckPointPath}_final.pth"
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if hasattr(dae_model, 'module'):
                dae_model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                dae_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded DAE weights from {checkpoint_path}")
            dae_model = dae_model.to(device)
            dae_model.eval()
        else:
            logger.warning(f"No DAE checkpoint found, skipping correction")
            dae_model = None
    
    # Create test dataset and data loader
    test_dataset = DSMDataset(test_rgb, test_sar, test_dsm, test_sem, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Initialize metrics accumulators
    all_dsm_preds = []
    all_dsm_targets = []
    all_sem_preds = []
    all_sem_targets = []
    
    # Create output directories
    if save_predictions or save_visualizations:
        output_dir = f"./output/{dataset_name}/test_results/"
        os.makedirs(output_dir, exist_ok=True)
        if save_predictions:
            os.makedirs(f"{output_dir}/predictions/", exist_ok=True)
        if save_visualizations:
            os.makedirs(f"{output_dir}/visualizations/", exist_ok=True)
    
    # Test loop
    logger.info("Starting inference on test data...")
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            start_time = time.time()
            
            # Move data to device
            inputs = batch_data['input'].to(device)
            dsm_targets = batch_data['dsm'].to(device)
            sem_targets = batch_data['sem'].to(device)
            
            # MTL forward pass
            dsm_pred, sem_pred, norm_pred, edge_pred = mtl_model(inputs, 'full', training=False)
            
            # Apply DAE correction if enabled
            if correction and dae_model is not None:
                dsm_pred = dae_model(dsm_pred)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert to CPU for metrics computation
            dsm_pred_cpu = dsm_pred.cpu()
            dsm_targets_cpu = dsm_targets.cpu()
            
            # Accumulate predictions
            all_dsm_preds.append(dsm_pred_cpu)
            all_dsm_targets.append(dsm_targets_cpu)
            
            if sem_pred is not None:
                all_sem_preds.append(sem_pred.cpu())
                all_sem_targets.append(sem_targets.cpu())
            
            # Save predictions if requested
            if save_predictions:
                save_prediction_images(
                    dsm_pred_cpu.squeeze().numpy(),
                    dsm_targets_cpu.squeeze().numpy(),
                    batch_idx,
                    f"{output_dir}/predictions/"
                )
            
            # Save visualizations if requested
            if save_visualizations and batch_idx < 10:  # Save first 10 for visualization
                save_comparison_visualization(
                    inputs.cpu().squeeze().numpy(),
                    dsm_pred_cpu.squeeze().numpy(),
                    dsm_targets_cpu.squeeze().numpy(),
                    sem_pred.cpu() if sem_pred is not None else None,
                    batch_idx,
                    f"{output_dir}/visualizations/"
                )
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} samples")
    
    # Compute overall metrics
    logger.info("Computing final metrics...")
    
    # Concatenate all predictions
    dsm_preds = torch.cat(all_dsm_preds, dim=0)
    dsm_targets = torch.cat(all_dsm_targets, dim=0)
    
    # Compute height metrics
    height_metrics = compute_height_metrics_pytorch(dsm_preds, dsm_targets)
    
    # Compute segmentation metrics if available
    segmentation_metrics = {}
    if all_sem_preds:
        sem_preds = torch.cat(all_sem_preds, dim=0)
        sem_targets = torch.cat(all_sem_targets, dim=0)
        segmentation_metrics = compute_segmentation_metrics_pytorch(sem_preds, sem_targets)
    
    # Combine all metrics
    all_metrics = {**height_metrics, **segmentation_metrics}
    
    # Add timing information
    avg_inference_time = np.mean(inference_times)
    all_metrics['avg_inference_time'] = avg_inference_time
    all_metrics['total_samples'] = len(test_loader)
    
    # Print results
    print_test_results(all_metrics, logger)
    
    # Save metrics to file
    save_test_metrics(all_metrics, f"{output_dir}/test_metrics.txt")
    
    logger.info(f"Testing completed successfully!")
    logger.info(f"Average inference time: {avg_inference_time:.4f} seconds per sample")
    
    return all_metrics


def save_prediction_images(pred_dsm, target_dsm, sample_idx, output_dir):
    """Save prediction and target DSM images"""
    
    # Normalize for visualization
    pred_norm = (pred_dsm - pred_dsm.min()) / (pred_dsm.max() - pred_dsm.min() + 1e-8)
    target_norm = (target_dsm - target_dsm.min()) / (target_dsm.max() - target_dsm.min() + 1e-8)
    
    # Save as images
    pred_path = f"{output_dir}/pred_dsm_{sample_idx:04d}.png"
    target_path = f"{output_dir}/target_dsm_{sample_idx:04d}.png"
    
    plt.imsave(pred_path, pred_norm, cmap='viridis')
    plt.imsave(target_path, target_norm, cmap='viridis')


def save_comparison_visualization(input_rgb, pred_dsm, target_dsm, sem_pred, sample_idx, output_dir):
    """Save side-by-side comparison visualization"""
    
    # Prepare RGB input for visualization
    if input_rgb.shape[0] >= 3:  # If has RGB channels
        rgb_viz = np.transpose(input_rgb[:3], (1, 2, 0))
        # Normalize RGB to [0, 1]
        rgb_viz = (rgb_viz - rgb_viz.min()) / (rgb_viz.max() - rgb_viz.min() + 1e-8)
    else:
        rgb_viz = np.zeros((input_rgb.shape[1], input_rgb.shape[2], 3))
    
    # Normalize DSM predictions and targets
    pred_norm = (pred_dsm - pred_dsm.min()) / (pred_dsm.max() - pred_dsm.min() + 1e-8)
    target_norm = (target_dsm - target_dsm.min()) / (target_dsm.max() - target_dsm.min() + 1e-8)
    
    # Create error map
    error_map = np.abs(pred_dsm - target_dsm)
    error_norm = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    
    # Create subplot visualization
    n_plots = 4 if sem_pred is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # RGB input
    axes[0].imshow(rgb_viz)
    axes[0].set_title('RGB Input')
    axes[0].axis('off')
    
    # Target DSM
    im1 = axes[1].imshow(target_norm, cmap='viridis')
    axes[1].set_title('Target DSM')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Predicted DSM
    im2 = axes[2].imshow(pred_norm, cmap='viridis')
    axes[2].set_title('Predicted DSM')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Error map
    if n_plots > 3:
        im3 = axes[3].imshow(error_norm, cmap='hot')
        axes[3].set_title('Error Map')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_{sample_idx:04d}.png", dpi=150, bbox_inches='tight')
    plt.close()


def print_test_results(metrics, logger):
    """Print formatted test results"""
    
    logger.info(f"\\n{'='*60}")
    logger.info(f"TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    
    # Height estimation metrics
    logger.info(f"\\nHeight Estimation Metrics:")
    logger.info(f"{'-'*30}")
    for metric in height_error_metrics + height_accuracy_metrics:
        if metric in metrics:
            logger.info(f"{metric.upper():>8}: {metrics[metric]:.6f}")
    
    # Segmentation metrics
    seg_metrics = [k for k in metrics.keys() if any(seg in k for seg in segmentation_scalar_metrics)]
    if seg_metrics:
        logger.info(f"\\nSegmentation Metrics:")
        logger.info(f"{'-'*25}")
        for metric in segmentation_scalar_metrics:
            if metric in metrics:
                logger.info(f"{metric.upper():>8}: {metrics[metric]:.6f}")
    
    # Performance metrics
    if 'avg_inference_time' in metrics:
        logger.info(f"\\nPerformance Metrics:")
        logger.info(f"{'-'*20}")
        logger.info(f"Avg Inference Time: {metrics['avg_inference_time']:.4f} seconds")
        logger.info(f"Total Samples: {metrics['total_samples']}")
    
    logger.info(f"\\n{'='*60}")


def save_test_metrics(metrics, output_path):
    """Save test metrics to file"""
    
    with open(output_path, 'w') as f:
        f.write(f"DSMNet Test Results - {dataset_name}\\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"{'='*60}\\n\\n")
        
        # Height metrics
        f.write(f"Height Estimation Metrics:\\n")
        f.write(f"{'-'*30}\\n")
        for metric in height_error_metrics + height_accuracy_metrics:
            if metric in metrics:
                f.write(f"{metric.upper():>8}: {metrics[metric]:.6f}\\n")
        
        # Segmentation metrics
        seg_metrics = [k for k in metrics.keys() if any(seg in k for seg in segmentation_scalar_metrics)]
        if seg_metrics:
            f.write(f"\\nSegmentation Metrics:\\n")
            f.write(f"{'-'*25}\\n")
            for metric in segmentation_scalar_metrics:
                if metric in metrics:
                    f.write(f"{metric.upper():>8}: {metrics[metric]:.6f}\\n")
        
        # Performance metrics
        if 'avg_inference_time' in metrics:
            f.write(f"\\nPerformance Metrics:\\n")
            f.write(f"{'-'*20}\\n")
            f.write(f"Avg Inference Time: {metrics['avg_inference_time']:.4f} seconds\\n")
            f.write(f"Total Samples: {metrics['total_samples']}\\n")
        
        # All metrics (for reference)
        f.write(f"\\nAll Metrics (Key: Value):\\n")
        f.write(f"{'-'*30}\\n")
        for key, value in sorted(metrics.items()):
            f.write(f"{key}: {value}\\n")


def main():
    """Main testing function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{dataset_name}_test_pytorch_output.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Run testing
    test_metrics = test_dsm(
        mtl_model=None,
        dae_model=None,
        save_predictions=True,
        save_visualizations=True
    )
    
    return test_metrics


if __name__ == "__main__":
    main()
