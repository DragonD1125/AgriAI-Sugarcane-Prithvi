
import numpy as np
from pathlib import Path
import random

# Config
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
SUGAR_DATA = BASE_DIR / "data" / "training_data_poc_randomized.npz"
RICE_DIR = BASE_DIR / "data" / "rice_patches"
OUTPUT_FILE = BASE_DIR / "data" / "training_data_multiclass.npz"

def merge_data():
    print("="*60)
    print("MERGING SUGARCANE AND RICE DATASETS")
    print("="*60)
    
    # 1. Load Sugarcane Data
    print(f"Loading Sugarcane data from {SUGAR_DATA.name}...")
    sugar = np.load(SUGAR_DATA)
    X_sugar = sugar['X_train']  # (N, 18, 224, 224) - merged train/val? No, existing is split
    y_sugar = sugar['y_train']
    
    # We need to combine train and val from previous split to re-split properly
    X_sugar_all = np.concatenate([sugar['X_train'], sugar['X_val']])
    y_sugar_all = np.concatenate([sugar['y_train'], sugar['y_val']])
    
    # Ensure y is 0 (bg) and 1 (sugar)
    # Existing y might be 0/1 float or int.
    print(f"  Shape: {X_sugar_all.shape}")
    print(f"  Labels in y: {np.unique(y_sugar_all)}")
    
    # 2. Load Rice Data
    print(f"\nLoading Rice patches from {RICE_DIR}...")
    rice_files = list(RICE_DIR.glob("rice_*.npz"))
    if not rice_files:
        print("  ❌ No rice patches found! Run create_rice_dataset.py first.")
        # Create dummy for testing if needed, or exit
        # For script completeness, I'll allow it to fail if no files
        return

    X_rice_list = []
    y_rice_list = []
    
    for f in rice_files:
        try:
            d = np.load(f)
            # Rice data saved as X=(6, 3, 224, 224) -> need flat 18
            # Prithvi script expects (18, 224, 224) input or handles logic.
            # My Kaggle script expects (18, 224, 224) then reshapes.
            # So let's flatten here to match sugarcane format
            
            x_raw = d['X'] # (6, 3, 224, 224)
            # Transpose to (3, 6, 224, 224) then flatten to (18, 224, 224)
            # Wait, sugarcane data is (18, 224, 224). 
            # In sugarcane script: 
            # X = X.reshape(N, 3, 6, 224, 224).transpose(0, 2, 1, 3, 4)
            # So input was (N, 18, ...) representing (6*3).
            # Let's align: 
            # X_rice is (6, 3, ...). Transpose to (3, 6, ...) -> reshape (18, ...)
            x_flat = x_raw.transpose(1, 0, 2, 3).reshape(18, 224, 224)
            
            X_rice_list.append(x_flat)
            y_rice_list.append(d['y']) # Already 2 (Rice) and 0 (BG)
        except Exception as e:
            print(f"  Error loading {f.name}: {e}")

    X_rice_all = np.stack(X_rice_list)
    y_rice_all = np.stack(y_rice_list)
    
    print(f"  Loaded {len(X_rice_all)} rice patches")
    print(f"  Labels in rice y: {np.unique(y_rice_all)}")

    # 3. Combine
    print("\nMerging...")
    X_final = np.concatenate([X_sugar_all, X_rice_all])
    y_final = np.concatenate([y_sugar_all, y_rice_all])
    
    # Shuffle
    idx = np.arange(len(X_final))
    np.random.shuffle(idx)
    X_final = X_final[idx]
    y_final = y_final[idx]
    
    # 4. Split Train/Val (80/20)
    split = int(len(X_final) * 0.8)
    X_train, X_val = X_final[:split], X_final[split:]
    y_train, y_val = y_final[:split], y_final[split:]
    
    print(f"\nFinal Split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Classes: {np.unique(y_final)} (0=Bg, 1=Sugar, 2=Rice)")
    
    # 5. Save
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )
    print(f"\n✅ Saved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")

if __name__ == "__main__":
    merge_data()
