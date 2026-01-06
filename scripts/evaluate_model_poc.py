#!/usr/bin/env python3
"""
Phase 8: Evaluate Model on Test Set (PoC)

This script evaluates the trained model on the temporally independent test set
(February 2021 data) and generates visualizations and metrics.

The test set uses only 1 temporal date (6 channels) vs training's 3 dates (18 channels).
We handle this by either:
1. Expanding test data to 18 channels (repeat the single date 3x)
2. Adapting the model to accept 6-channel input

We use approach 1 for simplicity.

Author: Agri AI Project
Date: 2024
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_PATH = BASE_DIR / "models" / "prithvi_sugarcane_poc_best.pth"
EVAL_DIR = BASE_DIR / "evaluation"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=18, out_channels=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict:
    """Compute comprehensive segmentation metrics."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    
    tp = ((preds_flat == 1) & (targets_flat == 1)).sum()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum()
    tn = ((preds_flat == 0) & (targets_flat == 0)).sum()
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU (Intersection over Union)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def compute_per_patch_metrics(all_preds: np.ndarray, all_targets: np.ndarray) -> Dict:
    """Compute metrics per patch for distribution analysis."""
    n_patches = len(all_preds)
    metrics_list = []
    
    for i in range(n_patches):
        m = compute_metrics(all_preds[i], all_targets[i])
        metrics_list.append(m)
    
    # Aggregate
    agg_metrics = {
        'accuracy_mean': np.mean([m['accuracy'] for m in metrics_list]),
        'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
        'f1_mean': np.mean([m['f1'] for m in metrics_list]),
        'f1_std': np.std([m['f1'] for m in metrics_list]),
        'iou_mean': np.mean([m['iou'] for m in metrics_list]),
        'iou_std': np.std([m['iou'] for m in metrics_list]),
    }
    
    return agg_metrics, metrics_list


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    num_samples: int = 10
) -> None:
    """Create visualization of predictions vs ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(num_samples, len(X))
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    for i in range(n_samples):
        # RGB composite (bands 2, 1, 0 = Red, Green, Blue)
        rgb = X[i, :3, :, :].transpose(1, 2, 0)  # Use first 3 channels
        rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        # Ground truth
        gt = y_true[i]
        
        # Prediction
        pred = y_pred[i]
        
        # Overlay (TP=green, FP=red, FN=yellow, TN=transparent)
        overlay = np.zeros((*gt.shape, 3))
        tp_mask = (pred == 1) & (gt == 1)
        fp_mask = (pred == 1) & (gt == 0)
        fn_mask = (pred == 0) & (gt == 1)
        overlay[tp_mask] = [0, 1, 0]  # Green - correct sugarcane
        overlay[fp_mask] = [1, 0, 0]  # Red - false positive
        overlay[fn_mask] = [1, 1, 0]  # Yellow - missed sugarcane
        
        axes[i, 0].imshow(rgb_norm)
        axes[i, 0].set_title('RGB Composite')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt, cmap='Greens')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='Greens')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(rgb_norm)
        axes[i, 3].imshow(overlay, alpha=0.5)
        axes[i, 3].set_title('Overlay (G=TP, R=FP, Y=FN)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_dir / 'predictions_visualization.png'}")


def plot_metrics_distribution(metrics_list: List[Dict], output_dir: Path) -> None:
    """Plot distribution of per-patch metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    accuracies = [m['accuracy'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    ious = [m['iou'] for m in metrics_list]
    
    axes[0].hist(accuracies, bins=20, edgecolor='black')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Accuracy Distribution\nMean: {np.mean(accuracies):.3f}')
    
    axes[1].hist(f1s, bins=20, edgecolor='black')
    axes[1].set_xlabel('F1 Score')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'F1 Score Distribution\nMean: {np.mean(f1s):.3f}')
    
    axes[2].hist(ious, bins=20, edgecolor='black')
    axes[2].set_xlabel('IoU')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'IoU Distribution\nMean: {np.mean(ious):.3f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics distribution to {output_dir / 'metrics_distribution.png'}")


def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_model():
    """Main evaluation function."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 8: EVALUATE MODEL (PoC)")
    print_separator()
    print()
    
    # Create output directory
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = UNet(in_channels=18, out_channels=2)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"  Loaded model from: {MODEL_PATH}")
    print(f"  Best training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Training val loss: {checkpoint['val_loss']:.4f}")
    print()
    
    # Load test data
    print("Loading test data...")
    data = np.load(DATA_FILE)
    X_test = data['X_test']  # Shape: (N, 6, 224, 224) - single date
    y_test = data['y_test']  # Shape: (N, 224, 224)
    
    print(f"  Test patches: {len(X_test)}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    # Expand 6-channel input to 18-channel by repeating
    print("\nExpanding test data from 6 to 18 channels (repeat 3x)...")
    X_test_expanded = np.concatenate([X_test, X_test, X_test], axis=1)
    print(f"  X_test_expanded shape: {X_test_expanded.shape}")
    print()
    
    # Run inference
    print("Running inference...")
    all_preds = []
    
    X_test_tensor = torch.from_numpy(X_test_expanded).float()
    
    batch_size = 4
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size].to(DEVICE)
        
        with torch.no_grad(), autocast(enabled=True):
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        all_preds.append(preds)
        
        if (i + batch_size) % 50 == 0 or (i + batch_size) >= len(X_test_tensor):
            print(f"  Processed {min(i + batch_size, len(X_test_tensor))}/{len(X_test_tensor)} patches")
    
    all_preds = np.concatenate(all_preds, axis=0)
    print(f"\nPredictions shape: {all_preds.shape}")
    
    # Compute overall metrics
    print_separator("-")
    print("Computing metrics...")
    print_separator("-")
    
    overall_metrics = compute_metrics(all_preds, y_test)
    per_patch_agg, per_patch_metrics = compute_per_patch_metrics(all_preds, y_test)
    
    print("\nOVERALL METRICS (on test set):")
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1 Score:  {overall_metrics['f1']:.4f}")
    print(f"  IoU:       {overall_metrics['iou']:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    TP: {overall_metrics['tp']:,} | FP: {overall_metrics['fp']:,}")
    print(f"    FN: {overall_metrics['fn']:,} | TN: {overall_metrics['tn']:,}")
    
    print("\nPER-PATCH METRICS:")
    print(f"  Accuracy: {per_patch_agg['accuracy_mean']:.3f} ± {per_patch_agg['accuracy_std']:.3f}")
    print(f"  F1 Score: {per_patch_agg['f1_mean']:.3f} ± {per_patch_agg['f1_std']:.3f}")
    print(f"  IoU:      {per_patch_agg['iou_mean']:.3f} ± {per_patch_agg['iou_std']:.3f}")
    
    # Save metrics to JSON
    metrics_file = EVAL_DIR / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'overall': overall_metrics,
            'per_patch_aggregate': per_patch_agg,
            'model_path': str(MODEL_PATH),
            'test_patches': len(X_test),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_predictions(X_test, y_test, all_preds, EVAL_DIR)
    plot_metrics_distribution(per_patch_metrics, EVAL_DIR)
    
    # Summary
    total_time = time.time() - start_time
    
    print()
    print_separator()
    print("EVALUATION COMPLETE")
    print_separator()
    print()
    print(f"Test set accuracy: {overall_metrics['accuracy']:.1%}")
    print(f"Test set F1 score: {overall_metrics['f1']:.3f}")
    print(f"Test set IoU:      {overall_metrics['iou']:.3f}")
    print()
    print(f"Results saved to: {EVAL_DIR}")
    print(f"Total time: {total_time:.1f} seconds")
    print()
    print("Phase 8 complete. PoC is complete!")
    print("\nNote: The test set uses temporally independent data (Feb 2021)")
    print("while training used Mar, Sep, Dec 2020 data.")


if __name__ == "__main__":
    evaluate_model()
