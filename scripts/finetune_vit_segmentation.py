#!/usr/bin/env python3
"""
ViT Segmentation Model - Multi-temporal Crop Detection

Vision Transformer (ViT) based segmentation model for crop detection.
Designed for 6GB VRAM GPUs with mixed precision training.

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

print("=" * 70)
print("VIT SEGMENTATION MODEL TRAINING")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ============================================================================
# DATASET
# ============================================================================

class SugarcaneDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # (N, 18, 224, 224)
        self.y = y  # (N, 224, 224)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (18, 224, 224)
        y = torch.from_numpy(self.y[idx]).long()   # (224, 224)
        return x, y


# ============================================================================
# PRITHVI-INSPIRED MODEL (Simpler approach)
# ============================================================================

class ViTSegmentation(nn.Module):
    """
    Vision Transformer for Semantic Segmentation.
    Optimized for 6GB VRAM GPUs.
    """
    
    def __init__(self, 
                 in_channels=18,  # 6 bands x 3 temporal
                 embed_dim=384,   # Smaller than full Prithvi
                 depth=12,
                 num_heads=6,
                 patch_size=16,
                 img_size=224,
                 num_classes=2):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196
        self.grid_size = img_size // patch_size  # 14
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Segmentation decoder (UperNet-style)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 28
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 56
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 112
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 224
            
            nn.Conv2d(32, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove CLS token and reshape
        x = x[:, 1:]  # (B, 196, embed_dim)
        x = x.reshape(B, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, 14, 14)
        
        # Decode
        out = self.decoder(x)  # (B, num_classes, 224, 224)
        
        return out


def try_load_pretrained_weights(model):
    """Optionally try to load pretrained ViT weights."""
    print("\nUsing random initialization for ViT model")
    return model


# ============================================================================
# TRAINING
# ============================================================================

def compute_metrics(preds, targets):
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
    start_time = time.time()
    
    # Load data
    print("\nLoading data...")
    data = np.load(DATA_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    print(f"  Training: {len(X_train)} patches")
    print(f"  Validation: {len(X_val)} patches")
    
    train_ds = SugarcaneDataset(X_train, y_train)
    val_ds = SugarcaneDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Build model
    print("\nBuilding ViT Segmentation model...")
    model = ViTSegmentation(
        in_channels=18,
        embed_dim=384,   # Optimized for 6GB VRAM
        depth=12,
        num_heads=6,
        num_classes=NUM_CLASSES
    )
    
    # Initialize weights
    model = try_load_pretrained_weights(model)
    model = model.to(DEVICE)
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {total_params:,} total, {trainable_params:,} trainable")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {mem:.2f} GB")
    
    # Loss
    sugar_ratio = y_train.mean()
    weights = torch.tensor([sugar_ratio, 1 - sugar_ratio]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING VIT SEGMENTATION MODEL")
    print("=" * 70 + "\n")
    
    best_val_loss = float('inf')
    best_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            try:
                with autocast():
                    output = model(X)
                    loss = criterion(output, y) / GRAD_ACCUMULATION
                
                scaler.scale(loss).backward()
                
                if (i + 1) % GRAD_ACCUMULATION == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * GRAD_ACCUMULATION
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at batch {i}! Clearing...")
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
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                with autocast():
                    output = model(X)
                    loss = criterion(output, y)
                
                val_loss += loss.item()
                m = compute_metrics(output, y)
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
                'model_type': 'ViTSegmentation'
            }, MODEL_DIR / "vit_sugarcane_best.pth")
            improved += " ✓"
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {all_metrics['acc']:.3f} | F1: {all_metrics['f1']:.3f}{improved}")
        
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU: {mem:.2f} GB")
    
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best F1: {best_f1:.3f}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Saved: {MODEL_DIR / 'vit_sugarcane_best.pth'}")


if __name__ == "__main__":
    train()
