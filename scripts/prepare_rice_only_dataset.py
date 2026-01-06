"""
Prepare Rice-Only Binary Dataset
=================================
Creates a separate dataset for binary Rice classification (0=BG, 1=Rice).
This helps diagnose if the problem is rice data quality or multiclass modeling.
"""

import numpy as np
from pathlib import Path

# Config
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
RICE_DIR_2018 = BASE_DIR / "data" / "rice_patches"
RICE_DIR_NEW = BASE_DIR / "data" / "rice_patches_v2"
OUTPUT_FILE = BASE_DIR / "data" / "training_data_rice_only.npz"

def load_rice_patches(rice_dir):
    """Load rice patches and convert to binary (0=BG, 1=Rice)."""
    files = list(rice_dir.glob("rice_*.npz"))
    if not files:
        print(f"  No patches found in {rice_dir}")
        return None, None
    
    X_list, y_list = [], []
    for f in files:
        try:
            d = np.load(f)
            x_raw = d['X']  # (6, 3, 224, 224)
            x_flat = x_raw.transpose(1, 0, 2, 3).reshape(18, 224, 224)
            y_raw = d['y']  # Values 0 and 2
            
            # Convert to binary: 0=BG, 1=Rice (was 2)
            y_binary = (y_raw > 0).astype(np.uint8)
            
            X_list.append(x_flat)
            y_list.append(y_binary)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    
    if X_list:
        return np.stack(X_list), np.stack(y_list)
    return None, None

def main():
    print("="*60)
    print("PREPARING RICE-ONLY BINARY DATASET")
    print("="*60)
    
    # Load 2018 rice
    print(f"\n1. Loading from rice_patches/...")
    X1, y1 = load_rice_patches(RICE_DIR_2018)
    if X1 is not None:
        print(f"   Loaded: {len(X1)} patches, Labels: {np.unique(y1)}")
    
    # Load 2020+2022 rice
    print(f"\n2. Loading from rice_patches_v2/...")
    X2, y2 = load_rice_patches(RICE_DIR_NEW)
    if X2 is not None:
        print(f"   Loaded: {len(X2)} patches, Labels: {np.unique(y2)}")
    
    # Combine
    datasets_X = []
    datasets_y = []
    if X1 is not None:
        datasets_X.append(X1)
        datasets_y.append(y1)
    if X2 is not None:
        datasets_X.append(X2)
        datasets_y.append(y2)
    
    X = np.concatenate(datasets_X)
    y = np.concatenate(datasets_y)
    
    # Shuffle
    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    
    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Stats
    print("\n" + "="*60)
    print("RICE-ONLY DATASET SUMMARY")
    print("="*60)
    print(f"   Total: {len(X)} patches")
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"   Labels: 0=Background, 1=Rice")
    
    # Rice pixel coverage
    rice_pixels = (y == 1).sum()
    total_pixels = y.size
    print(f"   Rice coverage: {100*rice_pixels/total_pixels:.1f}%")
    
    # Save
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    main()
