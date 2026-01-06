"""
Prithvi Model Comparison - Evaluation Script
============================================

Compares two Prithvi models:
- prithvi_run1_F1_77.pth (6-layer, ~25M params)
- prithvi_run2_F1_77.pth (12-layer, ~98M params)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL1_PATH = BASE_DIR / "models" / "prithvi_run1_F1_77.pth"
MODEL2_PATH = BASE_DIR / "models" / "prithvi_run2_F1_77.pth"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

# Model 1: 6-layer Prithvi (Run 1)
class PatchEmbed6(nn.Module):
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

class Prithvi6Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed6()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(6)])
        self.norm = nn.LayerNorm(768)
        self.decoder = nn.Sequential(
            nn.Conv2d(768*3, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 2, 1)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))[:, 1:]
        x = x.reshape(B, 3, 14, 14, 768).permute(0, 1, 4, 2, 3).reshape(B, 3*768, 14, 14)
        return self.decoder(x)


# Model 2: 12-layer Prithvi (Run 2)
class Attention12(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block12(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention12()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class PatchEmbed12(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(6, 768, kernel_size=(1, 16, 16), stride=(1, 16, 16))
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Prithvi12Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed12()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 589, 768))
        self.blocks = nn.ModuleList([Block12() for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(768*3, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 2, 1)
        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 1:]
        x = x.reshape(B, 3, 14, 14, 768).permute(0, 1, 4, 2, 3).reshape(B, 3*768, 14, 14)
        return self.seg_decoder(x)


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(pred, target):
    """Compute segmentation metrics"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def evaluate_model(model, X_val, y_val, name):
    """Evaluate a model on validation data"""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for i in range(len(X_val)):
            x = torch.from_numpy(X_val[i:i+1]).float().to(DEVICE)
            out = model(x)
            prob = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_probs.append(prob[:, 1].cpu().numpy())  # Sugarcane probability
    
    preds = np.concatenate(all_preds, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(torch.tensor(preds), torch.tensor(y_val))
    
    print(f"\n{name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    
    return preds, probs, metrics


def visualize_comparison(X_val, y_val, preds1, preds2, probs1, probs2, indices):
    """Create comparison visualization"""
    fig, axes = plt.subplots(len(indices), 6, figsize=(18, 3*len(indices)))
    
    for row, idx in enumerate(indices):
        # RGB composite (bands 2,1,0 = R,G,B from first temporal frame)
        # X_val shape is (N, 6, 3, 224, 224) - (bands, time, h, w)
        # Take bands 2,1,0 (R,G,B) from first temporal frame (0)
        rgb = X_val[idx, :3, 0, :, :].transpose(1, 2, 0)  # (224, 224, 3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title('RGB Input')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(y_val[idx], cmap='Greens', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground Truth')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(preds1[idx], cmap='Greens', vmin=0, vmax=1)
        axes[row, 2].set_title('Run1 (6-layer) Pred')
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(probs1[idx], cmap='RdYlGn', vmin=0, vmax=1)
        axes[row, 3].set_title('Run1 Confidence')
        axes[row, 3].axis('off')
        
        axes[row, 4].imshow(preds2[idx], cmap='Greens', vmin=0, vmax=1)
        axes[row, 4].set_title('Run2 (12-layer) Pred')
        axes[row, 4].axis('off')
        
        axes[row, 5].imshow(probs2[idx], cmap='RdYlGn', vmin=0, vmax=1)
        axes[row, 5].set_title('Run2 Confidence')
        axes[row, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prithvi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {OUTPUT_DIR / 'prithvi_comparison.png'}")


def create_error_map(y_val, preds, title):
    """Create TP/TN/FP/FN error map"""
    error = np.zeros_like(y_val, dtype=np.float32)
    error[(preds == 1) & (y_val == 1)] = 1  # TP - Green
    error[(preds == 0) & (y_val == 0)] = 2  # TN - Gray
    error[(preds == 1) & (y_val == 0)] = 3  # FP - Red
    error[(preds == 0) & (y_val == 1)] = 4  # FN - Yellow
    return error


def main():
    print("="*60)
    print("PRITHVI MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("\nLoading validation data...")
    data = np.load(DATA_FILE)
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Reshape for Prithvi: (N, 18, 224, 224) -> (N, 6, 3, 224, 224)
    N = X_val.shape[0]
    X_val = X_val.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
    
    print(f"Validation samples: {N}")
    print(f"Input shape: {X_val.shape}")
    
    # Load Model 1 (6-layer)
    print("\n" + "-"*40)
    print("Loading Model 1 (Run1 - 6 layers)")
    print("-"*40)
    model1 = Prithvi6Layer()
    state1 = torch.load(MODEL1_PATH, map_location='cpu')
    # Handle DataParallel state dict
    if any(k.startswith('module.') for k in state1.keys()):
        state1 = {k.replace('module.', ''): v for k, v in state1.items()}
    model1.load_state_dict(state1, strict=False)
    model1 = model1.to(DEVICE)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"Parameters: {params1:,}")
    
    # Load Model 2 (12-layer)
    print("\n" + "-"*40)
    print("Loading Model 2 (Run2 - 12 layers)")
    print("-"*40)
    model2 = Prithvi12Layer()
    state2 = torch.load(MODEL2_PATH, map_location='cpu')
    if any(k.startswith('module.') for k in state2.keys()):
        state2 = {k.replace('module.', ''): v for k, v in state2.items()}
    model2.load_state_dict(state2, strict=False)
    model2 = model2.to(DEVICE)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters: {params2:,}")
    
    # Evaluate both models
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    preds1, probs1, metrics1 = evaluate_model(model1, X_val, y_val, "Run1 (6-layer)")
    preds2, probs2, metrics2 = evaluate_model(model2, X_val, y_val, "Run2 (12-layer)")
    
    # Comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<15} {'Run1 (6-layer)':<18} {'Run2 (12-layer)':<18} {'Diff':<10}")
    print("-"*60)
    for key in ['accuracy', 'precision', 'recall', 'f1', 'iou']:
        v1 = metrics1[key]
        v2 = metrics2[key]
        diff = v2 - v1
        print(f"{key:<15} {v1:<18.4f} {v2:<18.4f} {diff:+.4f}")
    
    print(f"\n{'Parameters':<15} {params1:,} {'':>8} {params2:,}")
    print(f"{'Model Size':<15} {MODEL1_PATH.stat().st_size/1e6:.1f} MB {'':>8} {MODEL2_PATH.stat().st_size/1e6:.1f} MB")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Select indices with interesting cases
    indices = list(range(min(6, N)))  # First 6 samples
    visualize_comparison(X_val, y_val, preds1, preds2, probs1, probs2, indices)
    
    print("\n✅ Evaluation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
