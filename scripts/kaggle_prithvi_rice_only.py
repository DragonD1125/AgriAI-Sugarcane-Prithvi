"""
PRITHVI RICE-ONLY TRAINING (BINARY: Rice vs Background)
=========================================================
Diagnostic test to isolate rice data quality from multiclass complexity.
If this also plateaus at ~30% -> rice data quality issue
If this gets 60%+ -> multiclass modeling is the bottleneck
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

# Config - BINARY RICE
NUM_CLASSES = 2  # Background, Rice
BATCH_SIZE = 4
LR = 3e-5
EPOCHS = 100
PATIENCE = 20
DATA_PATH = Path("/kaggle/input/rice-only-data/training_data_rice_only.npz")
OUTPUT = Path("/kaggle/working")

# CELL 3: DATASET
class RiceDataset(Dataset):
    def __init__(self, X, y, augment=False):
        N = X.shape[0]
        X = X.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
        X = X.astype(np.float32)
        X = np.clip(X, 0, 10000) / 10000.0
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        self.X = X
        self.y = y.astype(np.int64)
        self.augment = augment
        print(f"  Data range: [{self.X.min():.3f}, {self.X.max():.3f}]")
    
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

# CELL 4: MODEL (6-layer Prithvi - matches what worked for Sugarcane)
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

class PrithviRice(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        self.norm = nn.LayerNorm(768)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(768 * 3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1)
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
    for k, v in state.items():
        k2 = k.replace('encoder.', '')
        if k2 in own and v.shape == own[k2].shape:
            own[k2] = v
            loaded += 1
    model.load_state_dict(own, strict=False)
    print(f"✅ Loaded {loaded} pretrained weights")
    return model

# CELL 5: METRICS
def compute_metrics(pred, target):
    p = pred.flatten()
    t = target.flatten()
    
    # Rice class metrics (class 1)
    tp = ((p == 1) & (t == 1)).sum().item()
    fp = ((p == 1) & (t != 1)).sum().item()
    fn = ((p != 1) & (t == 1)).sum().item()
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {'IoU': iou, 'F1': f1, 'P': precision, 'R': recall}

# CELL 6: TRAIN
def train():
    print("\n" + "="*60)
    print("PRITHVI RICE-ONLY TRAINING (BINARY)")
    print("="*60)
    
    data = np.load(DATA_PATH)
    train_ds = RiceDataset(data['X_train'], data['y_train'], augment=True)
    val_ds = RiceDataset(data['X_val'], data['y_val'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = PrithviRice(num_classes=2)
    model = load_weights(model)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    # Binary cross-entropy with class weight for imbalance
    cw = torch.tensor([0.3, 1.0]).to(DEVICE)  # Weight rice more
    criterion = nn.CrossEntropyLoss(weight=cw)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    best_iou = 0
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
            
            metrics = compute_metrics(
                np.concatenate(all_preds), 
                np.concatenate(all_targets)
            )
            
            marker = ""
            if metrics['IoU'] > best_iou:
                best_iou = metrics['IoU']
                epochs_no_improve = 0
                marker = " ⭐ BEST"
                torch.save(model.state_dict(), OUTPUT / "prithvi_rice_only_best.pth")
            else:
                epochs_no_improve += 5
            
            print(f"Epoch {epoch+1:3d} | Loss: {loss_sum/len(train_loader):.4f} | Rice IoU: {metrics['IoU']:.3f} F1: {metrics['F1']:.3f} (P:{metrics['P']:.2f} R:{metrics['R']:.2f}){marker}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {loss_sum/len(train_loader):.4f}")
    
    print(f"\n✅ Done! Best Rice IoU: {best_iou:.3f}")
    print(f"Model: {OUTPUT}/prithvi_rice_only_best.pth")

# CELL 7: RUN
if __name__ == "__main__":
    if torch.cuda.is_available():
        train()
    else:
        print("❌ Enable GPU!")
