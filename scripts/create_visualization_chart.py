#!/usr/bin/env python3
"""
Create enhanced visualization chart showing:
- RGB Image
- Ground Truth
- Prediction
- Evaluation overlay (TP, TN, FP, FN)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import torch.nn as nn

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
MODEL_PATH = BASE_DIR / "models" / "prithvi_sugarcane_poc_best.pth"
OUTPUT_DIR = BASE_DIR / "evaluation"


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=18, out_channels=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def create_evaluation_chart():
    """Create chart with RGB, Ground Truth, Prediction, and TP/TN/FP/FN overlay."""
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=18, out_channels=2)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Loading test data...")
    data = np.load(DATA_FILE)
    X_test = data['X_test']  # (N, 6, 224, 224)
    y_test = data['y_test']  # (N, 224, 224)
    
    # Expand to 18 channels
    X_test_expanded = np.concatenate([X_test, X_test, X_test], axis=1)
    
    # Select 6 sample patches with diverse outcomes
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), 6, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(6, 4, figsize=(16, 24))
    
    # Column titles
    col_titles = ['RGB Composite', 'Ground Truth', 'Prediction', 'Evaluation (TP/TN/FP/FN)']
    
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight='bold')
    
    for row, idx in enumerate(sample_indices):
        # Get data
        X = X_test_expanded[idx:idx+1]
        y = y_test[idx]
        
        # Run prediction
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            output = model(X_tensor)
            pred = output.argmax(dim=1).cpu().numpy()[0]
        
        # Create RGB composite from first 3 channels (B4, B3, B2 = R, G, B)
        rgb = X_test[idx, [2, 1, 0], :, :]  # R, G, B
        rgb = rgb.transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = np.clip(rgb * 2.5, 0, 1)  # Enhance brightness
        
        # Create evaluation overlay with colors
        eval_overlay = np.zeros((*y.shape, 3))
        tp_mask = (pred == 1) & (y == 1)
        tn_mask = (pred == 0) & (y == 0)
        fp_mask = (pred == 1) & (y == 0)
        fn_mask = (pred == 0) & (y == 1)
        
        eval_overlay[tp_mask] = [0.2, 0.8, 0.2]    # Green - TP
        eval_overlay[tn_mask] = [0.9, 0.9, 0.9]    # Light gray - TN
        eval_overlay[fp_mask] = [0.9, 0.2, 0.2]    # Red - FP
        eval_overlay[fn_mask] = [1.0, 0.8, 0.0]    # Yellow/Orange - FN
        
        # Calculate metrics for this patch
        tp = tp_mask.sum()
        tn = tn_mask.sum()
        fp = fp_mask.sum()
        fn = fn_mask.sum()
        total = tp + tn + fp + fn
        acc = (tp + tn) / total * 100
        
        # Plot RGB
        axes[row, 0].imshow(rgb)
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f'Patch {idx}', fontsize=12)
        
        # Plot Ground Truth
        axes[row, 1].imshow(y, cmap='Greens', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        gt_pct = y.mean() * 100
        axes[row, 1].text(5, 20, f'{gt_pct:.1f}% sugarcane', 
                         color='white', fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot Prediction
        axes[row, 2].imshow(pred, cmap='Greens', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        pred_pct = pred.mean() * 100
        axes[row, 2].text(5, 20, f'{pred_pct:.1f}% predicted', 
                         color='white', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Plot Evaluation overlay
        axes[row, 3].imshow(eval_overlay)
        axes[row, 3].axis('off')
        axes[row, 3].text(5, 20, f'Acc: {acc:.1f}%', 
                         color='black', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='TP (True Positive)'),
        mpatches.Patch(color=[0.9, 0.9, 0.9], label='TN (True Negative)'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='FP (False Positive)'),
        mpatches.Patch(color=[1.0, 0.8, 0.0], label='FN (False Negative)')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=12,
               bbox_to_anchor=(0.5, -0.01))
    
    plt.suptitle('Sugarcane Detection - Model Evaluation\n(Randomized Test Dates Approach)', 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'evaluation_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved evaluation chart to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_evaluation_chart()
