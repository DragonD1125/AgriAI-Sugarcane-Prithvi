#!/usr/bin/env python3
"""
Prithvi Fine-Tuning with TerraTorch

Official approach using IBM/NASA TerraTorch library for Prithvi models.
Uses:
- prithvi_vit_100 (100M params - fits in 6GB VRAM)
- UperNet decoder for segmentation
- Mixed precision (fp16)

Based on official config from:
https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification-demo

Author: Agri AI Project
Date: 2026
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model configuration - use smaller Prithvi for 6GB VRAM
# Options: prithvi_vit_100 (100M), prithvi_eo_v2_300 (300M), prithvi_eo_v2_600 (600M)
BACKBONE_NAME = "prithvi_vit_100"  # 100M params - fits in 6GB
NUM_CLASSES = 2  # background, sugarcane
NUM_FRAMES = 3   # temporal frames

# Training
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# SHOW GPU INFO
# ============================================================================

print("=" * 70)
print("PRITHVI FINE-TUNING WITH TERRATORCH")
print("=" * 70)
print()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU found!")
print()

# ============================================================================
# CUSTOM DATASET
# ============================================================================

class SugarcaneSegmentationDataset(Dataset):
    """Dataset for sugarcane segmentation compatible with TerraTorch."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: (N, 18, 224, 224) - 6 bands × 3 dates
            y: (N, 224, 224) - ground truth masks
        """
        self.N = X.shape[0]
        # Reshape to (N, T, C, H, W) for temporal models
        # 18 channels = 6 bands × 3 temporal frames
        self.X = X.reshape(self.N, 3, 6, 224, 224)  # (N, T=3, C=6, H=224, W=224)
        self.y = y
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # x: (T, C, H, W) - temporal first format for Prithvi
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).long()
        return {"image": x, "mask": y}


# ============================================================================
# BUILD MODEL USING TERRATORCH
# ============================================================================

def build_prithvi_model():
    """Build Prithvi model using TerraTorch factory."""
    print(f"Building Prithvi model: {BACKBONE_NAME}")
    print("-" * 70)
    
    try:
        from terratorch.models import EncoderDecoderFactory
        
        # Select indices based on backbone
        if BACKBONE_NAME == "prithvi_vit_100":
            select_indices = [2, 5, 8, 11]  # For 100M model
        elif BACKBONE_NAME == "prithvi_eo_v2_300":
            select_indices = [5, 11, 17, 23]  # For 300M model
        else:  # prithvi_eo_v2_600
            select_indices = [7, 15, 23, 31]  # For 600M model
        
        model_args = {
            "backbone": BACKBONE_NAME,
            "backbone_pretrained": True,
            "decoder": "UperNetDecoder",
            "decoder_channels": 256,
            "num_classes": NUM_CLASSES,
            "backbone_num_frames": NUM_FRAMES,
            "backbone_bands": [
                "BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"
            ],
            "head_dropout": 0.1,
            "rescale": True,
            "necks": [
                {"name": "SelectIndices", "indices": select_indices},
                {"name": "ReshapeTokensToImage", "effective_time_dim": NUM_FRAMES},
            ]
        }
        
        # Build model
        factory = EncoderDecoderFactory()
        model = factory.build_model(**model_args)
        
        print(f"  Model built successfully!")
        print(f"  Backbone: {BACKBONE_NAME}")
        print(f"  Decoder: UperNetDecoder")
        print(f"  Classes: {NUM_CLASSES}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"  TerraTorch factory failed: {e}")
        print("  Trying alternative approach...")
        
        # Fallback: Use pretrained Prithvi encoder with custom decoder
        return build_prithvi_fallback()


def build_prithvi_fallback():
    """Fallback model building if TerraTorch factory fails."""
    print("\nBuilding fallback model...")
    
    try:
        # Try loading Prithvi encoder from HuggingFace
        from transformers import AutoModel
        
        # Use the Temporal version
        encoder = AutoModel.from_pretrained(
            "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Simple decoder head
        class PrithviWithDecoder(nn.Module):
            def __init__(self, encoder, num_classes=2, hidden_dim=768):
                super().__init__()
                self.encoder = encoder
                self.decoder = nn.Sequential(
                    nn.Conv2d(hidden_dim, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=4, mode='bilinear'),
                    nn.Conv2d(256, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=4, mode='bilinear'),
                    nn.Conv2d(64, num_classes, 1)
                )
            
            def forward(self, x):
                features = self.encoder(x)
                if isinstance(features, dict):
                    features = features['last_hidden_state']
                # Reshape and decode
                B, N, C = features.shape
                h = w = int(N ** 0.5)
                features = features.reshape(B, h, w, C).permute(0, 3, 1, 2)
                return self.decoder(features)
        
        model = PrithviWithDecoder(encoder, num_classes=NUM_CLASSES)
        print("  Fallback model built!")
        return model
        
    except Exception as e2:
        print(f"  Fallback also failed: {e2}")
        raise RuntimeError("Could not build Prithvi model. Check TerraTorch/transformers installation.")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def compute_metrics(preds, targets):
    """Compute segmentation metrics."""
    preds_flat = preds.argmax(dim=1).flatten()
    targets_flat = targets.flatten()
    
    tp = ((preds_flat == 1) & (targets_flat == 1)).sum().float()
    fp = ((preds_flat == 1) & (targets_flat == 0)).sum().float()
    fn = ((preds_flat == 0) & (targets_flat == 1)).sum().float()
    tn = ((preds_flat == 0) & (targets_flat == 0)).sum().float()
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    
    return {'acc': acc.item(), 'prec': prec.item(), 'rec': rec.item(), 'f1': f1.item()}


def train():
    """Main training function."""
    start_time = time.time()
    
    # Load data
    print("\nLoading data...")
    data = np.load(DATA_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    print(f"  Training: {len(X_train)} patches")
    print(f"  Validation: {len(X_val)} patches")
    
    # Create datasets
    train_ds = SugarcaneSegmentationDataset(X_train, y_train)
    val_ds = SugarcaneSegmentationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Build model
    print()
    model = build_prithvi_model()
    model = model.to(DEVICE)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Check memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"\nGPU Memory after model load: {mem:.2f} GB")
    
    # Loss function with class weights
    sugar_ratio = y_train.mean()
    weights = torch.tensor([sugar_ratio, 1 - sugar_ratio]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.35)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")
    
    best_val_loss = float('inf')
    best_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            
            try:
                with autocast():
                    outputs = model(images)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
                    
                    # Ensure correct shape
                    if outputs.shape[2:] != masks.shape:
                        outputs = nn.functional.interpolate(outputs, size=masks.shape, mode='bilinear')
                    
                    loss = criterion(outputs, masks) / GRAD_ACCUMULATION
                
                scaler.scale(loss).backward()
                
                if (i + 1) % GRAD_ACCUMULATION == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * GRAD_ACCUMULATION
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at batch {i}! Clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise e
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_metrics = {'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                
                with autocast():
                    outputs = model(images)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs.get('output', list(outputs.values())[0]))
                    if outputs.shape[2:] != masks.shape:
                        outputs = nn.functional.interpolate(outputs, size=masks.shape, mode='bilinear')
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                m = compute_metrics(outputs, masks)
                for k in all_metrics:
                    all_metrics[k] += m[k]
        
        val_loss /= len(val_loader)
        for k in all_metrics:
            all_metrics[k] /= len(val_loader)
        
        # Log
        improved = ""
        if all_metrics['f1'] > best_f1:
            best_f1 = all_metrics['f1']
            improved = " ⬆️"
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': all_metrics,
                'backbone': BACKBONE_NAME
            }, MODEL_DIR / "prithvi_sugarcane_best.pth")
            improved += " ✓"
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {all_metrics['acc']:.3f} | F1: {all_metrics['f1']:.3f}{improved}")
        
        # Memory check
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU: {mem:.2f} GB")
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best F1: {best_f1:.3f}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Saved: {MODEL_DIR / 'prithvi_sugarcane_best.pth'}")


if __name__ == "__main__":
    train()
