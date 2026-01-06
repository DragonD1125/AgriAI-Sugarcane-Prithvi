"""
Visualize Rice Patches - RGB + Ground Truth
============================================
Samples random rice patches and displays Sentinel-2 RGB alongside the mask.
Helps diagnose data quality issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Config
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
RICE_DIR_2018 = BASE_DIR / "data" / "rice_patches"
RICE_DIR_NEW = BASE_DIR / "data" / "rice_patches_v2"
OUTPUT_DIR = BASE_DIR / "outputs" / "rice_visualization"
NUM_SAMPLES = 16

def visualize_patches():
    print("="*60)
    print("RICE PATCH VISUALIZATION")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gather all rice patches
    files = list(RICE_DIR_2018.glob("rice_*.npz")) + list(RICE_DIR_NEW.glob("rice_*.npz"))
    print(f"Found {len(files)} rice patches")
    
    if len(files) == 0:
        print("No patches found!")
        return
    
    # Random sample
    samples = random.sample(files, min(NUM_SAMPLES, len(files)))
    
    # Create figure
    fig, axes = plt.subplots(NUM_SAMPLES // 4, 8, figsize=(24, NUM_SAMPLES // 4 * 3))
    axes = axes.flatten()
    
    for i, f in enumerate(samples):
        try:
            d = np.load(f)
            X = d['X']  # (6, 3, 224, 224) - bands, times, H, W
            y = d['y']  # (224, 224)
            
            # Get RGB from middle timestamp (index 1)
            # Bands: B02, B03, B04, B8A, B11, B12
            # RGB = B04 (Red), B03 (Green), B02 (Blue) = indices 2, 1, 0
            rgb = X[:3, 1, :, :]  # First 3 bands at middle time
            rgb = rgb.transpose(1, 2, 0)  # (H, W, 3)
            rgb = np.clip(rgb / 3000, 0, 1)  # Normalize for display
            
            # Create mask overlay
            mask_rgb = np.zeros((224, 224, 3))
            mask_rgb[y == 2] = [0, 0, 1]  # Rice = Blue
            mask_rgb[y == 0] = [0, 0, 0]  # Background = Black
            
            # Plot RGB
            ax_rgb = axes[i * 2]
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f.stem[:20], fontsize=8)
            ax_rgb.axis('off')
            
            # Plot mask
            ax_mask = axes[i * 2 + 1]
            ax_mask.imshow(mask_rgb)
            rice_pct = 100 * (y == 2).sum() / y.size
            ax_mask.set_title(f"Rice: {rice_pct:.1f}%", fontsize=8)
            ax_mask.axis('off')
            
        except Exception as e:
            print(f"Error with {f.name}: {e}")
    
    plt.suptitle("Rice Patches: RGB (left) | Ground Truth Mask (right, Blue=Rice)", fontsize=14)
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "rice_patches_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize_patches()
