"""
PRITHVI-100M MULTICLASS FINE-TUNING v3 (FULL ARCHITECTURE)
============================================================
FIXES:
- Uses FULL 12-layer Prithvi architecture (not 6)
- Combined qkv attention (matches pretrained weights)
- Stronger decoder with more upsampling stages
- Focal Loss for better class handling
- Higher learning rate + longer training
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
from huggingface_hub import hf_hub_download
from pathlib import Path
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# Config
NUM_CLASSES = 3
BATCH_SIZE = 4
LR = 5e-5  # Higher LR
EPOCHS = 150
PATIENCE = 30
DATA_PATH = Path("/kaggle/input/multiclass-agri-data-v2/training_data_multiclass_v2.npz")
OUTPUT = Path("/kaggle/working")

# CELL 3: DATASET WITH AUGMENTATION
class MulticlassDataset(Dataset):
    def __init__(self, X, y, augment=False):
        N = X.shape[0]
        X = X.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
        X = X.astype(np.float32)
        X = np.clip(X, 0, 10000) / 10000.0
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        self.X = X
        self.y = y.astype(np.int64)
        self.augment = augment
        print(f"  Data range: [{self.X.min():.3f}, {self.X.max():.3f}], Augment: {augment}")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx].copy()
        if self.augment:
            if random.random() > 0.5:
                x = np.flip(x, axis=-1).copy()
                y = np.flip(y, axis=-1).copy()
            if random.random() > 0.5:
                x = np.flip(x, axis=-2).copy()
                y = np.flip(y, axis=-2).copy()
            k = random.randint(0, 3)
            if k > 0:
                x = np.rot90(x, k, axes=(-2, -1)).copy()
                y = np.rot90(y, k, axes=(-2, -1)).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

# CELL 4: FULL PRITHVI-100M ARCHITECTURE (12 LAYERS)
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=6, embed_dim=768, patch_size=16, tubelet_size=1):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class Attention(nn.Module):
    """Combined qkv attention - matches Prithvi weights"""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)  # Combined Q, K, V
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Prithvi100MMulticlass(nn.Module):
    """FULL Prithvi-100M with 12 transformer blocks"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Encoder - EXACT match to Prithvi-100M
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))
        self.blocks = nn.ModuleList([Block() for _ in range(12)])  # 12 blocks!
        self.norm = nn.LayerNorm(768)
        
        # Stronger decoder 
        self.decoder = nn.Sequential(
            nn.Conv2d(768 * 3, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)[:, 1:]  # Remove CLS token
        x = x.reshape(B, 3, 14, 14, 768).permute(0, 1, 4, 2, 3)
        x = x.reshape(B, 3 * 768, 14, 14)
        return self.decoder(x)

def load_prithvi_weights(model):
    print("\nDownloading Prithvi-100M weights...")
    path = hf_hub_download("ibm-nasa-geospatial/Prithvi-100M", "Prithvi_100M.pt")
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    
    own = model.state_dict()
    loaded = 0
    
    # Direct weight mapping
    for k, v in state.items():
        # Remove 'encoder.' prefix if present
        k_clean = k.replace('encoder.', '')
        
        if k_clean in own and v.shape == own[k_clean].shape:
            own[k_clean] = v
            loaded += 1
    
    model.load_state_dict(own, strict=False)
    total = sum(1 for k in own.keys() if 'decoder' not in k)
    print(f"✅ Loaded {loaded}/{total} encoder weights ({100*loaded/total:.1f}%)")
    return model

# CELL 5: FOCAL LOSS (better for imbalanced classes)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Class weights
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# CELL 6: METRICS
def compute_metrics(pred, target):
    p = pred.flatten()
    t = target.flatten()
    metrics = {}
    for cls_idx, cls_name in enumerate(['Bg', 'Sugar', 'Rice']):
        tp = ((p == cls_idx) & (t == cls_idx)).sum().item()
        fp = ((p == cls_idx) & (t != cls_idx)).sum().item()
        fn = ((p != cls_idx) & (t == cls_idx)).sum().item()
        iou = tp / (tp + fp + fn + 1e-8)
        metrics[f'IoU_{cls_name}'] = iou
    metrics['mIoU'] = np.mean([metrics['IoU_Sugar'], metrics['IoU_Rice']])
    return metrics

# CELL 7: TRAIN
def train():
    print("\n" + "="*60)
    print("PRITHVI-100M MULTICLASS v3 (FULL 12-LAYER)")
    print("="*60)
    
    data = np.load(DATA_PATH)
    train_ds = MulticlassDataset(data['X_train'], data['y_train'], augment=True)
    val_ds = MulticlassDataset(data['X_val'], data['y_val'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = Prithvi100MMulticlass(num_classes=3)
    model = load_prithvi_weights(model)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Focal loss with class weights
    alpha = torch.tensor([0.1, 1.0, 1.5]).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
    
    best_miou = 0
    epochs_no_improve = 0
    print("\nTraining...")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X).argmax(1)
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            val_metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
            miou = val_metrics['mIoU']
            
            marker = ""
            if miou > best_miou:
                best_miou = miou
                epochs_no_improve = 0
                marker = " ⭐ BEST"
                torch.save(model.state_dict(), OUTPUT / "prithvi_multiclass_v3_best.pth")
            else:
                epochs_no_improve += 5
            
            print(f"Epoch {epoch+1:3d} | Loss: {loss_sum/len(train_loader):.4f} | mIoU: {miou:.3f} (S:{val_metrics['IoU_Sugar']:.2f} R:{val_metrics['IoU_Rice']:.2f}){marker}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {loss_sum/len(train_loader):.4f}")
    
    print(f"\n✅ Done! Best mIoU: {best_miou:.3f}")
    print(f"Model: {OUTPUT}/prithvi_multiclass_v3_best.pth")

# CELL 8: RUN
if __name__ == "__main__":
    if torch.cuda.is_available():
        train()
    else:
        print("❌ Enable GPU!")
