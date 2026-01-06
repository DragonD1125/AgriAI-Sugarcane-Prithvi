#!/usr/bin/env python3
"""
Prepare Training Data for Kaggle Upload

Creates a zip file containing the training data that can be
uploaded as a Kaggle dataset for Prithvi fine-tuning.

Run: python scripts/prepare_kaggle_data.py
"""

import zipfile
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Configuration
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
DATA_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"
OUTPUT_DIR = BASE_DIR / "kaggle_upload"
OUTPUT_ZIP = OUTPUT_DIR / "sugarcane_training_data.zip"

def main():
    print("=" * 60)
    print("PREPARE DATA FOR KAGGLE UPLOAD")
    print("=" * 60)
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load and inspect data
    print(f"Loading: {DATA_FILE}")
    data = np.load(DATA_FILE)
    
    print("\nDataset contents:")
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Sugarcane ratio: {y_train.mean():.2%}")
    
    # Create metadata
    metadata = {
        "dataset_name": "Sugarcane Detection Training Data",
        "created": datetime.now().isoformat(),
        "source": "Di Tommaso et al. (2024) + Sentinel-2",
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "image_shape": list(X_train.shape[1:]),   # [18, 224, 224]
        "mask_shape": list(y_train.shape[1:]),    # [224, 224]
        "num_bands": 6,
        "num_temporal_frames": 3,
        "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
        "dates": ["Mar 2020", "Sep 2020", "Dec 2020"],
        "test_date": "Random per patch (1 of 4)",
        "class_labels": {
            "0": "Non-sugarcane",
            "1": "Sugarcane"
        },
        "sugarcane_ratio": float(y_train.mean()),
        "notes": "Randomized test date approach - each patch uses 3 dates for training, 1 random date for testing"
    }
    
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")
    
    # Create zip file
    print(f"\nCreating zip file: {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add training data
        zf.write(DATA_FILE, "training_data_poc_randomized.npz")
        # Add metadata
        zf.write(metadata_path, "metadata.json")
    
    # Report size
    zip_size = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\nZip file created: {zip_size:.1f} MB")
    
    print()
    print("=" * 60)
    print("UPLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Go to: https://www.kaggle.com/datasets
2. Click "+ New Dataset"
3. Upload: {}
4. Name: "sugarcane-sentinel2-training"
5. Set visibility to Private (recommended)
6. Click "Create"

After upload, add the dataset to your notebook!
""".format(OUTPUT_ZIP))

if __name__ == "__main__":
    main()
