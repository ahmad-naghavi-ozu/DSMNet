# Metrics computation for PyTorch DSMNet implementation
# Ahmad Naghavi, OzU 2025

import torch
import torch.nn.functional as F
import numpy as np
import os
import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from config import *

def save_checkpoint(model, optimizer, epoch, metric_value, checkpoint_path):
    """
    Save model checkpoint with state information
    
    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        epoch: Current training epoch
        metric_value: Current metric value (loss or accuracy)
        checkpoint_path: Path where to save the checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_value': metric_value,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)

def compute_height_metrics_pytorch(predictions, targets):
    """
    Compute height estimation metrics for PyTorch tensors
    
    Args:
        predictions: Predicted DSM tensor (B, C, H, W) or (B, H, W)
        targets: Target DSM tensor (B, C, H, W) or (B, H, W)
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Ensure tensors are squeezed to remove unnecessary dimensions
    pred = predictions.squeeze()
    targ = targets.squeeze()
    
    # Flatten for computation
    pred_flat = pred.view(-1)
    targ_flat = targ.view(-1)
    
    # Remove invalid values (if any)
    valid_mask = torch.isfinite(pred_flat) & torch.isfinite(targ_flat)
    pred_valid = pred_flat[valid_mask]
    targ_valid = targ_flat[valid_mask]
    
    if len(pred_valid) == 0:
        return {metric: 0.0 for metric in metric_names}
    
    # Compute basic error metrics
    diff = pred_valid - targ_valid
    abs_diff = torch.abs(diff)
    squared_diff = diff ** 2
    
    # MSE (Mean Squared Error)
    mse = torch.mean(squared_diff).item()
    
    # MAE (Mean Absolute Error)  
    mae = torch.mean(abs_diff).item()
    
    # RMSE (Root Mean Squared Error)
    rmse = torch.sqrt(torch.mean(squared_diff)).item()
    
    # Delta metrics (accuracy within relative thresholds)
    # delta1: percentage of pixels where max(pred/targ, targ/pred) < 1.25
    # delta2: percentage of pixels where max(pred/targ, targ/pred) < 1.25^2
    # delta3: percentage of pixels where max(pred/targ, targ/pred) < 1.25^3
    
    # Avoid division by zero
    targ_valid_safe = torch.clamp(targ_valid, min=1e-8)
    pred_valid_safe = torch.clamp(pred_valid, min=1e-8)
    
    ratio1 = pred_valid_safe / targ_valid_safe
    ratio2 = targ_valid_safe / pred_valid_safe
    max_ratio = torch.max(ratio1, ratio2)
    
    delta1 = (max_ratio < 1.25).float().mean().item()
    delta2 = (max_ratio < 1.25**2).float().mean().item()
    delta3 = (max_ratio < 1.25**3).float().mean().item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3
    }


def compute_segmentation_metrics_pytorch(predictions, targets):
    """
    Compute segmentation metrics for PyTorch tensors
    
    Args:
        predictions: Predicted segmentation tensor (B, C, H, W) - logits or probabilities
        targets: Target segmentation tensor (B, H, W) - class indices
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Convert predictions to class indices
    if len(predictions.shape) == 4:
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        pred_classes = predictions
    
    # Flatten for computation
    pred_flat = pred_classes.view(-1).cpu().numpy()
    targ_flat = targets.view(-1).cpu().numpy()
    
    # Remove invalid class indices
    valid_mask = (targ_flat >= 0) & (targ_flat < len(semantic_label_map))
    pred_valid = pred_flat[valid_mask]
    targ_valid = targ_flat[valid_mask]
    
    if len(pred_valid) == 0:
        return {}
    
    # Compute confusion matrix
    num_classes = len(semantic_label_map)
    cm = confusion_matrix(targ_valid, pred_valid, labels=range(num_classes))
    
    # Per-class metrics
    metrics = {}
    
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # IoU (Intersection over Union)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[f'iou_class{class_idx}'] = iou
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics[f'precision_class{class_idx}'] = precision
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f'recall_class{class_idx}'] = recall
        
        # F1-score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f'f1_score_class{class_idx}'] = f1
    
    # Overall metrics
    # Mean IoU (mIoU)
    class_ious = [metrics[f'iou_class{i}'] for i in range(num_classes)]
    metrics['miou'] = np.mean(class_ious)
    
    # Overall Accuracy (OA)
    correct_pixels = np.trace(cm)
    total_pixels = cm.sum()
    metrics['oa'] = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Frequency Weighted IoU (FWIoU)
    class_frequencies = cm.sum(axis=1)
    total_freq = class_frequencies.sum()
    if total_freq > 0:
        weighted_ious = [(class_frequencies[i] / total_freq) * class_ious[i] 
                        for i in range(num_classes)]
        metrics['fwiou'] = sum(weighted_ious)
    else:
        metrics['fwiou'] = 0.0
    
    return metrics


def plot_train_valid_metrics(epoch, train_metrics=None, valid_metrics=None, 
                           plot_train=True, model_type='MTL', fig=None, axes=None, lines=None):
    """
    Plot training and validation metrics
    
    Args:
        epoch: Current epoch
        train_metrics: Dictionary of training metrics
        valid_metrics: Dictionary of validation metrics  
        plot_train: Whether to plot training metrics
        model_type: Type of model ('MTL' or 'DAE')
        fig, axes, lines: Existing plot objects (optional)
    
    Returns:
        fig, axes, lines: Updated plot objects
    """
    if not valid_metrics:
        return None, None, None
    
    # Filter to only plot selected metrics
    metrics_to_plot = [m for m in plot_metrics if m in valid_metrics]
    if not metrics_to_plot:
        return None, None, None
    
    # Create figure if not provided
    if fig is None:
        n_metrics = len(metrics_to_plot)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        lines = {}
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        epochs = list(range(1, len(valid_metrics[metric]) + 1))
        
        # Plot validation metrics
        if f'valid_{metric}' not in lines:
            line_valid, = ax.plot(epochs, valid_metrics[metric], 'b-', label=f'Validation {metric}')
            lines[f'valid_{metric}'] = line_valid
        else:
            lines[f'valid_{metric}'].set_data(epochs, valid_metrics[metric])
        
        # Plot training metrics if available
        if plot_train and train_metrics and metric in train_metrics:
            if f'train_{metric}' not in lines:
                line_train, = ax.plot(epochs, train_metrics[metric], 'r-', label=f'Training {metric}')
                lines[f'train_{metric}'] = line_train
            else:
                lines[f'train_{metric}'].set_data(epochs, train_metrics[metric])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{model_type} {metric.upper()} Progress')
        ax.legend()
        ax.grid(True)
        
        # Auto-adjust axis limits
        ax.relim()
        ax.autoscale_view()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'./plots/{model_type.lower()}_metrics_{dataset_name}_{timestamp}.png'
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    return fig, axes, lines


def should_compute_metrics(iteration, total_iterations, log_frequency):
    """Check if metrics should be computed at current iteration"""
    return iteration % log_frequency == 0 or iteration == total_iterations


def log_metrics_and_save_model(model, optimizer, epoch, iteration, total_iterations,
                              losses_dict, metrics_dict, min_loss, last_epoch_saved, 
                              checkpoint_path, logger, eval_metric='rmse'):
    """Log metrics and save model if improved"""
    
    # Log current metrics
    logger.info(f"Epoch {epoch}, Iter {iteration}/{total_iterations}")
    logger.info(f"Losses - Total: {losses_dict['total']:.6f}, DSM: {losses_dict['dsm']:.6f}")
    
    if metrics_dict:
        for metric_name, value in metrics_dict.items():
            if metric_name in metric_names:
                logger.info(f"{metric_name.upper()}: {value:.6f}")
    
    # Check if model should be saved (improved performance)
    current_metric = metrics_dict.get(eval_metric, losses_dict['total'])
    
    if eval_metric_lower_better:
        improved = current_metric < min_loss
    else:
        improved = current_metric > min_loss
    
    if improved:
        min_loss = current_metric
        last_epoch_saved = epoch
        
        # Save model checkpoint
        save_checkpoint(model, optimizer, epoch, current_metric, checkpoint_path)
        logger.info(f"Model saved at epoch {epoch} with {eval_metric}: {current_metric:.6f}")
    
    return min_loss, last_epoch_saved


def compute_delta_metrics(pred, target, thresholds=[1.25, 1.25**2, 1.25**3]):
    """
    Compute delta accuracy metrics
    
    Args:
        pred: Predictions tensor
        target: Targets tensor  
        thresholds: List of threshold values
    
    Returns:
        List of delta accuracies
    """
    # Avoid division by zero
    target_safe = torch.clamp(target, min=1e-8)
    pred_safe = torch.clamp(pred, min=1e-8)
    
    ratio1 = pred_safe / target_safe
    ratio2 = target_safe / pred_safe
    max_ratio = torch.max(ratio1, ratio2)
    
    deltas = []
    for threshold in thresholds:
        delta = (max_ratio < threshold).float().mean().item()
        deltas.append(delta)
    
    return deltas


def print_metrics_summary(metrics_dict, dataset_split='validation'):
    """Print a formatted summary of metrics"""
    print(f"\\n{'='*50}")
    print(f"{dataset_split.upper()} METRICS SUMMARY")
    print(f"{'='*50}")
    
    # Height metrics
    if any(metric in metrics_dict for metric in height_error_metrics + height_accuracy_metrics):
        print("\\nHeight Estimation Metrics:")
        print("-" * 30)
        for metric in height_error_metrics:
            if metric in metrics_dict:
                print(f"{metric.upper():>8}: {metrics_dict[metric]:.6f}")
        for metric in height_accuracy_metrics:
            if metric in metrics_dict:
                print(f"{metric.upper():>8}: {metrics_dict[metric]:.6f}")
    
    # Segmentation metrics
    seg_metrics = [k for k in metrics_dict.keys() if any(seg in k for seg in segmentation_scalar_metrics)]
    if seg_metrics:
        print("\\nSegmentation Metrics:")
        print("-" * 25)
        for metric in segmentation_scalar_metrics:
            if metric in metrics_dict:
                print(f"{metric.upper():>8}: {metrics_dict[metric]:.6f}")
    
    print(f"\\n{'='*50}")


# Additional utility functions for metrics computation
def masked_metrics(pred, target, mask=None):
    """Compute metrics with optional masking"""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    
    return compute_height_metrics_pytorch(pred.unsqueeze(0), target.unsqueeze(0))


def per_class_metrics_summary(metrics_dict, num_classes):
    """Print per-class metrics summary"""
    print("\\nPer-Class Metrics:")
    print("-" * 40)
    print(f"{'Class':>5} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 40)
    
    for class_idx in range(num_classes):
        iou = metrics_dict.get(f'iou_class{class_idx}', 0.0)
        precision = metrics_dict.get(f'precision_class{class_idx}', 0.0)
        recall = metrics_dict.get(f'recall_class{class_idx}', 0.0)
        f1 = metrics_dict.get(f'f1_score_class{class_idx}', 0.0)
        
        print(f"{class_idx:>5} {iou:>8.4f} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f}")


class HeightMetrics:
    """Class for computing height estimation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics accumulation"""
        self.total_mse = 0.0
        self.total_mae = 0.0
        self.total_rmse = 0.0
        self.total_samples = 0
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions, targets):
        """Update metrics with new batch"""
        metrics = compute_height_metrics_pytorch(predictions, targets)
        batch_size = predictions.shape[0] if len(predictions.shape) > 2 else 1
        
        self.total_mse += metrics.get('mse', 0.0) * batch_size
        self.total_mae += metrics.get('mae', 0.0) * batch_size
        self.total_rmse += metrics.get('rmse', 0.0) * batch_size
        self.total_samples += batch_size
        
        # Store for detailed analysis
        self.all_predictions.append(predictions.detach().cpu())
        self.all_targets.append(targets.detach().cpu())
    
    def compute(self):
        """Compute final averaged metrics"""
        if self.total_samples == 0:
            return {}
        
        return {
            'mse': self.total_mse / self.total_samples,
            'mae': self.total_mae / self.total_samples,
            'rmse': self.total_rmse / self.total_samples,
            'total_samples': self.total_samples
        }
    
    def compute_detailed(self):
        """Compute detailed metrics on all accumulated data"""
        if not self.all_predictions:
            return {}
        
        all_pred = torch.cat(self.all_predictions, dim=0)
        all_targ = torch.cat(self.all_targets, dim=0)
        
        return compute_height_metrics_pytorch(all_pred, all_targ)


class SegmentationMetrics:
    """Class for computing semantic segmentation metrics"""
    
    def __init__(self, num_classes=None):
        self.num_classes = num_classes if num_classes is not None else sem_k
        self.reset()
    
    def reset(self):
        """Reset metrics accumulation"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, predictions, targets):
        """Update metrics with new batch"""
        # Convert predictions to class indices if they are logits/probabilities
        if len(predictions.shape) == 4:  # (B, C, H, W)
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
        
        # Convert to numpy for confusion matrix computation
        pred_np = pred_classes.detach().cpu().numpy().flatten()
        targ_np = targets.detach().cpu().numpy().flatten()
        
        # Update confusion matrix
        cm = confusion_matrix(targ_np, pred_np, labels=range(self.num_classes))
        self.confusion_matrix += cm
        self.total_samples += predictions.shape[0] if len(predictions.shape) > 2 else 1
    
    def compute(self):
        """Compute final metrics from confusion matrix"""
        if self.confusion_matrix.sum() == 0:
            return {}
        
        # Compute per-class metrics
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Avoid division by zero
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, 
                      out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
        
        # IoU computation
        union = tp + fp + fn
        iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union != 0)
        
        # Overall accuracy
        accuracy = np.sum(tp) / np.sum(self.confusion_matrix) if np.sum(self.confusion_matrix) > 0 else 0.0
        
        # Mean metrics
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_f1 = np.mean(f1)
        mean_iou = np.mean(iou)
        
        metrics = {
            'accuracy': accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou,
            'total_samples': self.total_samples
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'precision_class{i}'] = precision[i]
            metrics[f'recall_class{i}'] = recall[i]
            metrics[f'f1_score_class{i}'] = f1[i]
            metrics[f'iou_class{i}'] = iou[i]
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get the current confusion matrix"""
        return self.confusion_matrix.copy()
    
    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', save_path=None):
        """Plot confusion matrix"""
        cm = self.confusion_matrix.copy()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
