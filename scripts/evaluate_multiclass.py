"""
Evaluate Prithvi Multiclass Model (Sugarcane + Rice)
=====================================================
Generates visualizations showing predictions vs ground truth.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Config
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
MODEL_PATH = BASE_DIR / "models" / "prithvi_multiclass_best.pth"
DATA_PATH = BASE_DIR / "data" / "training_data_multiclass.npz"
OUTPUT_DIR = BASE_DIR / "outputs" / "multiclass_eval"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color map for 3 classes
COLORS = {
    0: [0, 0, 0],       # Background - Black
    1: [0, 255, 0],     # Sugarcane - Green
    2: [0, 100, 255],   # Rice - Blue
}

# Model Architecture (must match training)
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

def mask_to_rgb(mask):
    """Convert class mask to RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLORS.items():
        rgb[mask == cls_id] = color
    return rgb

def compute_metrics(pred, target):
    """Compute per-class IoU"""
    metrics = {}
    for cls_idx, cls_name in enumerate(['Background', 'Sugarcane', 'Rice']):
        p = (pred == cls_idx)
        t = (target == cls_idx)
        intersection = (p & t).sum()
        union = (p | t).sum()
        iou = intersection / (union + 1e-8)
        metrics[cls_name] = iou
    return metrics

def evaluate():
    print("="*60)
    print("MULTICLASS PRITHVI EVALUATION")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH.name}...")
    data = np.load(DATA_PATH)
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Reshape and normalize (match training)
    N = X_val.shape[0]
    X_val = X_val.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
    X_val = np.clip(X_val.astype(np.float32), 0, 10000) / 10000.0
    
    print(f"  Validation samples: {N}")
    print(f"  Classes in labels: {np.unique(y_val)}")
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH.name}...")
    model = PrithviMulticlass(num_classes=3)
    
    # Handle DataParallel saved weights
    state = torch.load(MODEL_PATH, map_location='cpu')
    new_state = {}
    for k, v in state.items():
        new_state[k.replace('module.', '')] = v
    model.load_state_dict(new_state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    print("  ✅ Model loaded")
    
    # Run inference
    print("\nRunning inference...")
    all_preds = []
    with torch.no_grad():
        for i in range(N):
            x = torch.from_numpy(X_val[i:i+1]).to(DEVICE)
            out = model(x)
            pred = out.argmax(1).cpu().numpy()[0]
            all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    
    # Compute metrics
    print("\nMetrics:")
    overall_metrics = compute_metrics(all_preds.flatten(), y_val.flatten())
    for cls_name, iou in overall_metrics.items():
        print(f"  {cls_name}: {iou:.3f} IoU")
    
    mIoU = (overall_metrics['Sugarcane'] + overall_metrics['Rice']) / 2
    print(f"  Mean IoU (Sugar+Rice): {mIoU:.3f}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    num_viz = min(12, N)
    
    fig, axes = plt.subplots(num_viz, 4, figsize=(16, 4*num_viz))
    if num_viz == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_viz):
        # RGB composite (use middle timestep, bands 2,1,0 for RGB)
        rgb = X_val[i, 1, :3, :, :].transpose(1, 2, 0)  # (224, 224, 3)
        rgb = np.clip(rgb * 3, 0, 1)  # Brighten for visibility
        
        gt_rgb = mask_to_rgb(y_val[i])
        pred_rgb = mask_to_rgb(all_preds[i])
        
        # Difference map
        diff = np.zeros_like(gt_rgb)
        diff[y_val[i] != all_preds[i]] = [255, 0, 0]  # Red for errors
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Sample {i+1}: RGB")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(diff)
        axes[i, 3].set_title("Errors (Red)")
        axes[i, 3].axis('off')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='black', label='Background'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='Sugarcane'),
        plt.Rectangle((0,0),1,1, facecolor='blue', label='Rice'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    
    out_path = OUTPUT_DIR / "multiclass_predictions.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {out_path}")
    
    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Sugarcane IoU: {overall_metrics['Sugarcane']:.1%}")
    print(f"  Rice IoU:      {overall_metrics['Rice']:.1%}")
    print(f"  Mean IoU:      {mIoU:.1%}")
    print(f"\n  Visualization saved!")
    
    plt.show()

if __name__ == "__main__":
    evaluate()
