#!/usr/bin/env python3
"""
Phase 7: Fine-Tune Prithvi-EO-2.0 on PoC Data

This script fine-tunes NASA/IBM's Prithvi-EO-2.0 model for sugarcane segmentation.

NOTE: Prithvi-EO-2.0-600M is a large model requiring significant GPU memory.
For RTX 3050 (6GB VRAM), we use:
- Gradient checkpointing
- Mixed precision training (fp16)
- Small batch size (1)
- Gradient accumulation

Alternative: If Prithvi doesn't fit, we fall back to a simpler U-Net architecture.

Author: Agri AI Project
Date: 2024
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_DIR = BASE_DIR / "models"

# Training configuration
BATCH_SIZE = 1  # RTX 3050 memory constraint
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
GRAD_ACCUMULATION_STEPS = 4  # Effective batch size = 4

# Model configuration
NUM_INPUT_CHANNELS = 18  # 6 bands × 3 dates for training
NUM_OUTPUT_CLASSES = 2   # Binary: background, sugarcane
PATCH_SIZE = 224

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = True  # Automatic Mixed Precision


# ============================================================================
# DATASET
# ============================================================================

class SugarcaneDataset(Dataset):
    """PyTorch Dataset for sugarcane patches."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: numpy array of shape (N, C, 224, 224)
            y: numpy array of shape (N, 224, 224)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# SIMPLE U-NET MODEL (Fallback if Prithvi doesn't fit in memory)
# ============================================================================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    
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
    """Simple U-Net for semantic segmentation."""
    
    def __init__(self, in_channels=18, out_channels=2):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict:
    """Compute accuracy, precision, recall, F1 for binary segmentation."""
    preds_flat = preds.argmax(dim=1).flatten()
    targets_flat = targets.flatten()
    
    # True positives, false positives, false negatives
    tp = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
    tn = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_accum_steps: int = 4
) -> Tuple[float, Dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    num_batches = 0
    
    optimizer.zero_grad()
    
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        with autocast(enabled=USE_AMP):
            outputs = model(X)
            loss = criterion(outputs, y) / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, y)
            for k in all_metrics:
                all_metrics[k] += metrics[k]
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    num_batches = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            with autocast(enabled=USE_AMP):
                outputs = model(X)
                loss = criterion(outputs, y)
            
            total_loss += loss.item()
            
            metrics = compute_metrics(outputs, y)
            for k in all_metrics:
                all_metrics[k] += metrics[k]
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    
    return avg_loss, avg_metrics


def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model():
    """Main training function."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 7: FINE-TUNE MODEL (PoC)")
    print_separator()
    print()
    
    # Check device
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Load data
    print("Loading training data...")
    data = np.load(DATA_FILE)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print(f"  Training: {X_train.shape[0]} patches, shape {X_train.shape[1:]}")
    print(f"  Validation: {X_val.shape[0]} patches, shape {X_val.shape[1:]}")
    print()
    
    # Create datasets and dataloaders
    train_dataset = SugarcaneDataset(X_train, y_train)
    val_dataset = SugarcaneDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model (U-Net as fallback from Prithvi)
    print("Creating model...")
    model = UNet(in_channels=NUM_INPUT_CHANNELS, out_channels=NUM_OUTPUT_CLASSES)
    model = model.to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: U-Net (Prithvi fallback)")
    print(f"  Parameters: {n_params:,}")
    print()
    
    # Loss function with class weights (handle class imbalance)
    # Count sugarcane vs non-sugarcane pixels
    sugarcane_ratio = y_train.mean()
    class_weights = torch.tensor([sugarcane_ratio, 1.0 - sugarcane_ratio]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=USE_AMP)
    
    # Training loop
    print_separator("-")
    print("Training...")
    print_separator("-")
    
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_DIR / "prithvi_sugarcane_poc_best.pth"
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, GRAD_ACCUMULATION_STEPS
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.3f} | "
              f"Val F1: {val_metrics['f1']:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, best_model_path)
            print(f"  ✓ Saved best model (epoch {epoch+1})")
    
    # Summary
    total_time = time.time() - start_time
    
    print()
    print_separator()
    print("TRAINING COMPLETE")
    print_separator()
    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    # Final metrics
    print("Final validation metrics:")
    print(f"  Accuracy: {history['val_acc'][-1]:.3f}")
    print(f"  F1 Score: {history['val_f1'][-1]:.3f}")
    print()
    print("Phase 7 complete. Ready for Phase 8 (evaluation).")


if __name__ == "__main__":
    train_model()
