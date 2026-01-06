"""
PRITHVI-100M MULTICLASS FINE-TUNING (SUGARCANE + RICE)
======================================================
Copy to Kaggle notebook. Settings: GPU T4 x2
Classes: 0=Background, 1=Sugarcane, 2=Rice
Dataset: Upload "training_data_multiclass.npz"
"""

# CELL 1: INSTALL
!pip install timm einops huggingface_hub --quiet
print("✅ Installed!")

# CELL 2: IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from huggingface_hub import hf_hub_download
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# Config - MULTICLASS
NUM_CLASSES = 3  # Background, Sugarcane, Rice
BATCH_SIZE = 4
LR = 1.5e-5
EPOCHS = 80
DATA_PATH = Path("/kaggle/input/multiclass-agri-data/training_data_multiclass.npz")
OUTPUT = Path("/kaggle/working")

# CELL 3: DATASET
class MulticlassDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 18, 224, 224) -> (N, 6, 3, 224, 224)
        N = X.shape[0]
        X = X.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
        
        # CRITICAL: Normalize Sentinel-2 data to [0, 1]
        # Sentinel-2 reflectance values are typically 0-10000
        X = X.astype(np.float32)
        X = np.clip(X, 0, 10000) / 10000.0  # Normalize to [0, 1]
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)  # Handle NaN/Inf
        
        self.X = X
        self.y = y.astype(np.int64)
        print(f"  Data range: [{self.X.min():.3f}, {self.X.max():.3f}]")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# CELL 4: EXACT PRITHVI-100M ARCHITECTURE (Reused from Run1/Run2)
# Using the 6-layer architecture (Run1) since it performed better
# But configured for 3 classes output

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(6, 768, kernel_size=(1, 16, 16), stride=(1, 16, 16))
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, dim=768, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

class PrithviMulticlass(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))
        # Using 6 layers as it outperformed 12 layers on our small dataset
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        self.norm = nn.LayerNorm(768)
        
        # Decoder for NUM_CLASSES
        self.decoder = nn.Sequential(
            nn.Conv2d(768 * 3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1)  # Output 3 channels
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))[:, 1:]
        x = x.reshape(B, 3, 14, 14, 768).permute(0, 1, 4, 2, 3)
        x = x.reshape(B, 3 * 768, 14, 14)
        return self.decoder(x)

def load_weights(model):
    print("\nDownloading Prithvi-100M weights...")
    path = hf_hub_download("ibm-nasa-geospatial/Prithvi-100M", "Prithvi_100M.pt")
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    
    own = model.state_dict()
    loaded = 0
    # Loose matching as we are using the simplified 6-layer architecture
    for k, v in state.items():
        k2 = k.replace('encoder.', '')
        if k2 in own and v.shape == own[k2].shape:
            own[k2] = v
            loaded += 1
    model.load_state_dict(own, strict=False)
    print(f"✅ Loaded {loaded} pretrained weights (Simple Arch)")
    return model

# CELL 5: TRAIN
def compute_metrics(pred, target):
    # Flatten
    p = pred.flatten()
    t = target.flatten()
    
    metrics = {}
    
    # Per-class IoU
    for cls_idx, cls_name in enumerate(['Bg', 'Sugar', 'Rice']):
        tp = ((p == cls_idx) & (t == cls_idx)).sum().item()
        fp = ((p == cls_idx) & (t != cls_idx)).sum().item()
        fn = ((p != cls_idx) & (t == cls_idx)).sum().item()
        
        iou = tp / (tp + fp + fn + 1e-8)
        metrics[f'IoU_{cls_name}'] = iou
        
    # Mean IoU (excluding background usually, but here all 3)
    metrics['mIoU'] = np.mean([metrics['IoU_Sugar'], metrics['IoU_Rice']])
    return metrics

def train():
    print("\n" + "="*50)
    print("PRITHVI MULTICLASS FINE-TUNING")
    print("="*50)
    
    # Data
    data = np.load(DATA_PATH)
    train_ds = MulticlassDataset(data['X_train'], data['y_train'])
    val_ds = MulticlassDataset(data['X_val'], data['y_val'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = PrithviMulticlass(num_classes=3)
    model = load_weights(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Loss: Weigh classes to handle imbalance
    # Assuming Rice/Sugar roughly equal but smaller than Bg
    # Weights: Bg=0.2, Sugar=1.0, Rice=1.0
    cw = torch.tensor([0.2, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    scaler = GradScaler()
    
    best_miou = 0
    print("\nTraining...")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass (no autocast for stability)
            out = model(X)
            loss = criterion(out, y)
            
            # Skip if NaN
            if torch.isnan(loss):
                print(f"  Warning: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X).argmax(1)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            val_metrics = compute_metrics(
                np.concatenate(all_preds), 
                np.concatenate(all_targets)
            )
            miou = val_metrics['mIoU']
            
            marker = ""
            if miou > best_miou:
                best_miou = miou
                marker = " ⭐ BEST"
                torch.save(model.state_dict(), OUTPUT / "prithvi_multiclass_best.pth")
            
            print(f"Epoch {epoch+1:2d} | Loss: {loss_sum/len(train_loader):.4f} | mIoU: {miou:.3f} (S:{val_metrics['IoU_Sugar']:.2f} R:{val_metrics['IoU_Rice']:.2f}){marker}")
        else:
            print(f"Epoch {epoch+1:2d} | Loss: {loss_sum/len(train_loader):.4f}")
    
    print(f"\n✅ Done! Best mIoU: {best_miou:.3f}")
    print(f"Model: {OUTPUT}/prithvi_multiclass_best.pth")

# CELL 6: RUN
if __name__ == "__main__":
    if torch.cuda.is_available():
        train()
    else:
        print("❌ Enable GPU!")
