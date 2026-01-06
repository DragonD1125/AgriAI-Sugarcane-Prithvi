"""
Merge ALL datasets into training_data_multiclass_v2.npz
=======================================================
Combines:
- Existing Sugarcane patches (192)
- 2018 Rice patches (165) from rice_patches/
- 2020+2022 Rice patches (~300) from rice_patches_v2/
"""

import numpy as np
from pathlib import Path
import random

# Config
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
SUGAR_DATA = BASE_DIR / "data" / "training_data_poc_randomized.npz"
RICE_DIR_2018 = BASE_DIR / "data" / "rice_patches"
RICE_DIR_NEW = BASE_DIR / "data" / "rice_patches_v2"
OUTPUT_FILE = BASE_DIR / "data" / "training_data_multiclass_v2.npz"

def load_rice_patches(rice_dir, label=2):
    """Load rice patches from a directory."""
    files = list(rice_dir.glob("rice_*.npz"))
    if not files:
        print(f"  No patches found in {rice_dir}")
        return None, None
    
    X_list, y_list = [], []
    for f in files:
        try:
            d = np.load(f)
            x_raw = d['X']  # (6, 3, 224, 224)
            # Flatten to (18, 224, 224) to match sugarcane format
            x_flat = x_raw.transpose(1, 0, 2, 3).reshape(18, 224, 224)
            X_list.append(x_flat)
            y_list.append(d['y'])
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")
    
    if X_list:
        return np.stack(X_list), np.stack(y_list)
    return None, None

def merge_data():
    print("="*60)
    print("MERGING ALL DATASETS (v2)")
    print("="*60)
    
    # 1. Load Sugarcane
    print(f"\n1. Loading Sugarcane from {SUGAR_DATA.name}...")
    sugar = np.load(SUGAR_DATA)
    X_sugar = np.concatenate([sugar['X_train'], sugar['X_val']])
    y_sugar = np.concatenate([sugar['y_train'], sugar['y_val']])
    print(f"   Shape: {X_sugar.shape}, Labels: {np.unique(y_sugar)}")
    
    # 2. Load 2018 Rice
    print(f"\n2. Loading 2018 Rice from {RICE_DIR_2018.name}/...")
    X_rice_2018, y_rice_2018 = load_rice_patches(RICE_DIR_2018)
    if X_rice_2018 is not None:
        print(f"   Shape: {X_rice_2018.shape}, Labels: {np.unique(y_rice_2018)}")
    
    # 3. Load 2020+2022 Rice
    print(f"\n3. Loading 2020+2022 Rice from {RICE_DIR_NEW.name}/...")
    X_rice_new, y_rice_new = load_rice_patches(RICE_DIR_NEW)
    if X_rice_new is not None:
        print(f"   Shape: {X_rice_new.shape}, Labels: {np.unique(y_rice_new)}")
    
    # 4. Combine all
    print("\n4. Merging all datasets...")
    datasets_X = [X_sugar]
    datasets_y = [y_sugar]
    
    if X_rice_2018 is not None:
        datasets_X.append(X_rice_2018)
        datasets_y.append(y_rice_2018)
    
    if X_rice_new is not None:
        datasets_X.append(X_rice_new)
        datasets_y.append(y_rice_new)
    
    X_final = np.concatenate(datasets_X)
    y_final = np.concatenate(datasets_y)
    
    # Shuffle
    idx = np.arange(len(X_final))
    np.random.seed(42)
    np.random.shuffle(idx)
    X_final = X_final[idx]
    y_final = y_final[idx]
    
    # 5. Train/Val Split (80/20)
    split = int(len(X_final) * 0.8)
    X_train, X_val = X_final[:split], X_final[split:]
    y_train, y_val = y_final[:split], y_final[split:]
    
    print(f"\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"   Total Samples: {len(X_final)}")
    print(f"   - Sugarcane: {len(X_sugar)}")
    if X_rice_2018 is not None:
        print(f"   - Rice 2018: {len(X_rice_2018)}")
    if X_rice_new is not None:
        print(f"   - Rice 2020+2022: {len(X_rice_new)}")
    print(f"\n   Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"   Classes: {np.unique(y_final)} (0=Bg, 1=Sugar, 2=Rice)")
    
    # 6. Save
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    print(f"   File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    merge_data()
