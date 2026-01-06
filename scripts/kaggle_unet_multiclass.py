"""
U-NET MULTICLASS SEGMENTATION (Alternative to Prithvi)
=======================================================
Uses proven U-Net architecture with pretrained ResNet34 backbone.
If this also plateaus at ~32%, the issue is DATA QUALITY not MODEL.
"""

# CELL 1: INSTALL
!pip install segmentation-models-pytorch --quiet
print("✅ Installed!")

# CELL 2: IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import segmentation_models_pytorch as smp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# Config
NUM_CLASSES = 3
BATCH_SIZE = 8  # U-Net is lighter, can use larger batch
LR = 1e-4
EPOCHS = 100
PATIENCE = 20
DATA_PATH = Path("/kaggle/input/multiclass-agri-data-v2/training_data_multiclass_v2.npz")
OUTPUT = Path("/kaggle/working")

# CELL 3: DATASET
class MulticlassDataset(Dataset):
    def __init__(self, X, y, augment=False):
        N = X.shape[0]
        # Keep as (N, 18, 224, 224) - 18 channels = 6 bands × 3 times
        X = X.astype(np.float32)
        X = np.clip(X, 0, 10000) / 10000.0
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        self.X = X
        self.y = y.astype(np.int64)
        self.augment = augment
        print(f"  Data shape: {self.X.shape}, range: [{self.X.min():.3f}, {self.X.max():.3f}]")
    
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

# CELL 4: MODEL
def create_model():
    """U-Net with ResNet34 backbone, adapted for 18 input channels"""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # Pretrained on ImageNet
        in_channels=18,              # 6 bands × 3 timestamps
        classes=NUM_CLASSES,
    )
    return model

# CELL 5: METRICS
def compute_iou(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(0)
    return ious

# CELL 6: TRAIN
def train():
    print("\n" + "="*60)
    print("U-NET MULTICLASS SEGMENTATION")
    print("="*60)
    
    # Data
    data = np.load(DATA_PATH)
    train_ds = MulticlassDataset(data['X_train'], data['y_train'], augment=True)
    val_ds = MulticlassDataset(data['X_val'], data['y_val'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = create_model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Loss with class weights
    weights = torch.tensor([0.1, 1.0, 1.5]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item()
        
        avg_loss = loss_sum / len(train_loader)
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_ious = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    pred = model(X).argmax(1).cpu().numpy()
                    target = y.cpu().numpy()
                    for i in range(len(pred)):
                        all_ious.append(compute_iou(pred[i], target[i]))
            
            mean_ious = np.mean(all_ious, axis=0)
            miou = (mean_ious[1] + mean_ious[2]) / 2  # Sugar + Rice
            
            scheduler.step(avg_loss)
            
            marker = ""
            if miou > best_miou:
                best_miou = miou
                epochs_no_improve = 0
                marker = " ⭐ BEST"
                torch.save(model.state_dict(), OUTPUT / "unet_multiclass_best.pth")
            else:
                epochs_no_improve += 5
            
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | mIoU: {miou:.3f} (Bg:{mean_ious[0]:.2f} S:{mean_ious[1]:.2f} R:{mean_ious[2]:.2f}){marker}")
            
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
    
    print(f"\n✅ Done! Best mIoU: {best_miou:.3f}")
    print(f"Model: {OUTPUT}/unet_multiclass_best.pth")

# CELL 7: RUN
if __name__ == "__main__":
    if torch.cuda.is_available():
        train()
    else:
        print("❌ Enable GPU!")
