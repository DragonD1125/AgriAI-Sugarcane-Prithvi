#!/usr/bin/env python3
"""
Prithvi-EO-2.0 Fine-tuning with LoRA

Memory-optimized fine-tuning for 6GB VRAM using:
- 8-bit quantization (bitsandbytes)
- LoRA adapters (PEFT)
- Gradient checkpointing
- Batch size 1 with gradient accumulation

Model: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL
Task: Sugarcane field segmentation

Author: Agri AI Project
Date: 2026
"""

import os
import sys
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Check GPU
print("=" * 70)
print("PRITHVI-EO-2.0 FINE-TUNING WITH LoRA")
print("=" * 70)
print()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU found, will use CPU (very slow)")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
USE_8BIT = True  # 8-bit quantization
USE_LORA = True  # LoRA adapters

# LoRA configuration
LORA_R = 16          # LoRA rank
LORA_ALPHA = 32      # LoRA scaling
LORA_DROPOUT = 0.1   # Dropout

# Training configuration
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATASET
# ============================================================================

class SugarcaneDataset(Dataset):
    """Dataset for sugarcane segmentation."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, 18, 224, 224) - 6 bands x 3 dates
        # For Prithvi, we need to reshape to (N, T, C, H, W)
        # T = 3 temporal frames, C = 6 channels
        self.N = X.shape[0]
        self.X = X.reshape(self.N, 3, 6, 224, 224)  # (N, T, C, H, W)
        self.y = y
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(self.y[idx]).long()
        return x, y


# ============================================================================
# PRITHVI MODEL WITH SEGMENTATION HEAD
# ============================================================================

class PrithviSegmentationHead(nn.Module):
    """Segmentation head for Prithvi encoder outputs."""
    
    def __init__(self, embed_dim=768, num_classes=2, patch_size=16, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size  # 14
        
        # Decoder
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
    
    def forward(self, x):
        # x: (B, num_patches, embed_dim)
        B = x.shape[0]
        
        # Reshape to spatial: (B, grid, grid, embed_dim) -> (B, embed_dim, grid, grid)
        x = x.reshape(B, self.grid_size, self.grid_size, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, 14, 14)
        
        # Decode to full resolution
        out = self.decoder(x)  # (B, num_classes, 224, 224)
        
        return out


class PrithviForSegmentation(nn.Module):
    """Prithvi encoder with segmentation head."""
    
    def __init__(self, encoder, embed_dim=768, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.seg_head = PrithviSegmentationHead(embed_dim, num_classes)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        # Prithvi expects (B, C, T, H, W) - channel first then temporal
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Encode
        features = self.encoder(x)
        
        # Handle different output formats
        if isinstance(features, dict):
            features = features.get('last_hidden_state', features.get('hidden_states', None))
            if features is None:
                features = list(features.values())[0]
        
        if isinstance(features, tuple):
            features = features[0]
        
        # features should be (B, num_patches, embed_dim) or (B, embed_dim)
        if features.dim() == 2:
            # Global pooled - need to expand for segmentation
            # This shouldn't happen for segmentation tasks
            raise ValueError("Got global pooled features, need spatial features")
        
        # Remove CLS token if present
        if features.shape[1] == 197:  # 14*14 + 1 CLS token
            features = features[:, 1:, :]  # Remove CLS
        
        # Segment
        out = self.seg_head(features)
        return out


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_prithvi_model():
    """Load Prithvi model with memory optimizations."""
    from transformers import AutoModel
    
    print("Loading Prithvi-EO-2.0-300M...")
    print(f"  8-bit quantization: {USE_8BIT}")
    print(f"  LoRA adapters: {USE_LORA}")
    
    # Load with 8-bit quantization
    if USE_8BIT and torch.cuda.is_available():
        try:
            encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            print("  Loaded with 8-bit quantization")
        except Exception as e:
            print(f"  8-bit failed: {e}")
            print("  Falling back to fp16...")
            encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(DEVICE)
    else:
        encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(DEVICE)
    
    # Enable gradient checkpointing
    if hasattr(encoder, 'gradient_checkpointing_enable'):
        encoder.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
    
    # Freeze encoder initially
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Apply LoRA if enabled
    if USE_LORA:
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=["q_proj", "v_proj", "k_proj"],  # Attention layers
                lora_dropout=LORA_DROPOUT,
                bias="none"
            )
            encoder = get_peft_model(encoder, lora_config)
            encoder.print_trainable_parameters()
            print("  LoRA adapters applied")
        except Exception as e:
            print(f"  LoRA failed: {e}")
            print("  Training without LoRA")
            # Unfreeze some layers for training
            for name, param in encoder.named_parameters():
                if 'layer' in name and any(x in name for x in ['11', '10', '9']):
                    param.requires_grad = True
    
    # Get embedding dimension
    if hasattr(encoder, 'config'):
        embed_dim = getattr(encoder.config, 'hidden_size', 768)
    else:
        embed_dim = 768
    
    # Create full model
    model = PrithviForSegmentation(encoder, embed_dim=embed_dim, num_classes=2)
    
    # Move segmentation head to device
    model.seg_head = model.seg_head.to(DEVICE)
    
    return model


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


def train_prithvi():
    """Main training function."""
    start_time = time.time()
    
    # Load data
    print("\nLoading training data...")
    data = np.load(DATA_FILE)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print(f"  Training: {len(X_train)} patches")
    print(f"  Validation: {len(X_val)} patches")
    
    # Create datasets
    train_ds = SugarcaneDataset(X_train, y_train)
    val_ds = SugarcaneDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load model
    print("\n" + "-" * 70)
    model = load_prithvi_model()
    print("-" * 70)
    
    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory after loading: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Loss and optimizer
    sugar_ratio = y_train.mean()
    weights = torch.tensor([sugar_ratio, 1 - sugar_ratio]).float().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING PRITHVI")
    print("=" * 70)
    print()
    
    best_val_loss = float('inf')
    best_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (X, y) in enumerate(train_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            
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
                    print(f"\nOOM at batch {i}! Clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise e
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_metrics = {'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0}
        
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                
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
        
        scheduler.step(val_loss)
        
        # Log
        improved = ""
        if all_metrics['f1'] > best_f1:
            best_f1 = all_metrics['f1']
            improved = " ⬆️"
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': all_metrics
            }, MODEL_DIR / "prithvi_sugarcane_best.pth")
            improved += " ✓"
        
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {all_metrics['acc']:.3f} | F1: {all_metrics['f1']:.3f}{improved}")
        
        # Memory check
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory: {mem:.2f} GB")
    
    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best F1 score: {best_f1:.3f}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Model saved to: {MODEL_DIR / 'prithvi_sugarcane_best.pth'}")


if __name__ == "__main__":
    train_prithvi()
