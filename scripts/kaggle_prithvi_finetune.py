"""
PRITHVI-100M FINE-TUNING - FULL WEIGHT LOADING
===============================================
Copy to Kaggle notebook. Settings: GPU T4 x2
Matches exact Prithvi-100M architecture for 100% weight loading.
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

# Config
BATCH_SIZE = 4
LR = 1.5e-5
EPOCHS = 80
DATA_PATH = Path("/kaggle/input/sugarcane-sentinel2-training/training_data_poc_randomized.npz")
OUTPUT = Path("/kaggle/working")

# CELL 3: DATASET
class SugarcaneDataset(Dataset):
    def __init__(self, X, y):
        N = X.shape[0]
        X = X.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# CELL 4: EXACT PRITHVI-100M ARCHITECTURE
class PatchEmbed(nn.Module):
    """Matches encoder.patch_embed.proj"""
    def __init__(self, in_chans=6, embed_dim=768, tubelet_size=1, patch_size=16):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=(tubelet_size, patch_size, patch_size),
                              stride=(tubelet_size, patch_size, patch_size))
    
    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class Attention(nn.Module):
    """Matches encoder.blocks.X.attn with combined qkv"""
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
    """Matches encoder.blocks.X.mlp"""
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
    """Matches encoder.blocks.X"""
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

class Prithvi100M(nn.Module):
    """
    EXACT Prithvi-100M Encoder Architecture
    - 12 transformer blocks (not 6)
    - Combined qkv attention
    - Matches all weight names from checkpoint
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Encoder - matches encoder.* weights
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))  # 1 + 3*14*14
        self.blocks = nn.ModuleList([Block() for _ in range(12)])  # 12 blocks!
        self.norm = nn.LayerNorm(768)
        
        # Segmentation decoder (custom - not from pretrained)
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(768 * 3, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, num_classes, 1)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Encoder forward
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Reshape for segmentation
        x = x[:, 1:]  # Remove CLS token
        x = x.reshape(B, 3, 14, 14, 768).permute(0, 1, 4, 2, 3)
        x = x.reshape(B, 3 * 768, 14, 14)
        
        return self.seg_decoder(x)

def load_pretrained_weights(model):
    """Load ALL encoder weights from Prithvi-100M checkpoint"""
    print("\n" + "="*50)
    print("LOADING PRITHVI-100M PRETRAINED WEIGHTS")
    print("="*50)
    
    path = hf_hub_download("ibm-nasa-geospatial/Prithvi-100M", "Prithvi_100M.pt")
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    
    # Build mapping from checkpoint keys to our model keys
    own = model.state_dict()
    loaded = 0
    skipped = 0
    
    for ckpt_key, ckpt_val in state.items():
        # Only load encoder weights (skip decoder - we have custom segmentation head)
        if not ckpt_key.startswith('encoder.'):
            skipped += 1
            continue
        
        # Remove 'encoder.' prefix
        our_key = ckpt_key.replace('encoder.', '')
        
        if our_key in own:
            if ckpt_val.shape == own[our_key].shape:
                own[our_key] = ckpt_val
                loaded += 1
            else:
                print(f"  Shape mismatch: {our_key} - ckpt:{ckpt_val.shape} vs model:{own[our_key].shape}")
        else:
            # Try without 'encoder.' for nested paths
            pass
    
    model.load_state_dict(own, strict=False)
    
    # Count total encoder weights in checkpoint
    encoder_weights = len([k for k in state.keys() if k.startswith('encoder.')])
    
    print(f"\n✅ Loaded {loaded}/{encoder_weights} encoder weights!")
    print(f"   (Skipped {skipped} decoder weights - using custom segmentation head)")
    
    return model

# CELL 5: TRAINING
def train():
    print("\n" + "="*50)
    print("PRITHVI-100M FINE-TUNING (FULL WEIGHTS)")
    print("="*50)
    
    # Data
    data = np.load(DATA_PATH)
    train_ds = SugarcaneDataset(data['X_train'], data['y_train'])
    val_ds = SugarcaneDataset(data['X_val'], data['y_val'])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model with full pretrained weights
    model = Prithvi100M(num_classes=2)
    model = load_pretrained_weights(model)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    scaler = GradScaler()
    
    best_f1 = 0
    print("\nTraining...")
    
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        scheduler.step()
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            tp, fp, fn = 0, 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X).argmax(1)
                    tp += ((pred == 1) & (y == 1)).sum().item()
                    fp += ((pred == 1) & (y == 0)).sum().item()
                    fn += ((pred == 0) & (y == 1)).sum().item()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            
            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                marker = " ⭐ BEST"
                torch.save(model.state_dict(), OUTPUT / "prithvi_full_best.pth")
            
            print(f"Epoch {epoch+1:2d} | Loss: {loss_sum/len(train_loader):.4f} | F1: {f1:.3f}{marker}")
        else:
            print(f"Epoch {epoch+1:2d} | Loss: {loss_sum/len(train_loader):.4f}")
    
    print(f"\n✅ Done! Best F1: {best_f1:.3f}")
    print(f"Model: {OUTPUT}/prithvi_full_best.pth")

# CELL 6: RUN
if __name__ == "__main__":
    if torch.cuda.is_available():
        train()
    else:
        print("❌ Enable GPU!")
