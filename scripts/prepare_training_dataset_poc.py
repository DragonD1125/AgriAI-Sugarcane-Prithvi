#!/usr/bin/env python3
"""
Phase 6 (v2): Prepare Training Dataset with Randomized Test Dates

RANDOMIZED TEST DATE APPROACH:
- For each patch, randomly select 1 of 4 dates as the test date
- Use remaining 3 dates for training
- This ensures model learns temporal patterns from ALL seasons

This creates a leave-one-date-out cross-validation style split per patch.

Author: Agri AI Project
Date: 2024
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
PATCHES_DIR = BASE_DIR / "data" / "patches_poc"
OUTPUT_FILE = BASE_DIR / "data" / "training_data_poc_randomized.npz"

# Dates available (indices 0-3)
NUM_DATES = 4
DATE_NAMES = ['2020-03-15', '2020-09-28', '2020-12-15', '2021-02-25']

# Train/val split from training patches
VALIDATION_SPLIT = 0.15


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


def load_all_patches(patches_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load all patch files from directory.
    
    Returns:
        X: numpy array of shape (N, 6, 4, 224, 224) - all patches
        y: numpy array of shape (N, 224, 224) - all ground truth
        metadata: list of dicts with patch metadata
    """
    patch_files = sorted(patches_dir.glob("patch_*.npz"))
    
    if not patch_files:
        raise ValueError(f"No patches found in {patches_dir}")
    
    print(f"Loading {len(patch_files)} patches...")
    
    X_list = []
    y_list = []
    metadata = []
    
    for i, pf in enumerate(patch_files):
        with np.load(pf) as data:
            X_list.append(data['X'])
            y_list.append(data['y'])
            metadata.append({
                'file': pf.name,
                'latitude': float(data['latitude']),
                'longitude': float(data['longitude']),
                'valid_pct': float(data['valid_pct']),
                'sugarcane_pct': float(data['sugarcane_pct'])
            })
        
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(patch_files)} patches")
    
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    print(f"  Total patches loaded: {len(X)}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    return X, y, metadata


def create_randomized_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create randomized per-patch test date split.
    
    For each patch:
    - Randomly select 1 of 4 dates as test
    - Use remaining 3 dates as training
    
    X shape: (N, 6_bands, 4_dates, 224, 224)
    
    Returns:
        X_train: (N_train, 18, 224, 224) - flattened 3 training dates
        y_train: (N_train, 224, 224)
        X_val: (N_val, 18, 224, 224)
        y_val: (N_val, 224, 224)
        X_test: (N, 6, 224, 224) - single test date per patch
        y_test: (N, 224, 224)
        test_date_indices: (N,) - which date was used for test for each patch
    """
    np.random.seed(random_seed)
    
    n_patches = X.shape[0]
    n_bands = X.shape[1]  # 6
    n_dates = X.shape[2]  # 4
    patch_size = X.shape[3]  # 224
    
    print(f"\nCreating randomized per-patch test date split...")
    print(f"  Each patch will have a random date held out for testing")
    
    # Randomly assign test date for each patch
    test_date_indices = np.random.randint(0, n_dates, size=n_patches)
    
    # Count distribution
    test_date_counts = np.bincount(test_date_indices, minlength=n_dates)
    print(f"\n  Test date distribution:")
    for i, count in enumerate(test_date_counts):
        print(f"    Date {i} ({DATE_NAMES[i]}): {count} patches ({count/n_patches*100:.1f}%)")
    
    # Create X_train_full (all patches, 3 training dates each) and X_test (1 test date each)
    X_train_full = np.zeros((n_patches, n_bands * 3, patch_size, patch_size), dtype=X.dtype)
    X_test = np.zeros((n_patches, n_bands, patch_size, patch_size), dtype=X.dtype)
    
    for i in range(n_patches):
        test_idx = test_date_indices[i]
        train_indices = [d for d in range(n_dates) if d != test_idx]
        
        # Extract training dates and flatten: (6, 3, 224, 224) -> (18, 224, 224)
        X_train_patch = X[i, :, train_indices, :, :]  # (6, 3, 224, 224)
        X_train_full[i] = X_train_patch.reshape(-1, patch_size, patch_size)  # (18, 224, 224)
        
        # Extract test date: (6, 224, 224)
        X_test[i] = X[i, :, test_idx, :, :]
    
    y_test = y.copy()  # All patches have ground truth
    
    print(f"\n  X_train_full shape: {X_train_full.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Split training patches into train/val
    n_val = int(n_patches * validation_split)
    n_train = n_patches - n_val
    
    # Shuffle indices for train/val split
    indices = np.random.permutation(n_patches)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train = X_train_full[train_indices]
    y_train = y[train_indices]
    
    X_val = X_train_full[val_indices]
    y_val = y[val_indices]
    
    print(f"\n  Dataset split:")
    print(f"    Training: {len(X_train)} patches")
    print(f"    Validation: {len(X_val)} patches")
    print(f"    Test: {len(X_test)} patches (with randomized test dates)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, test_date_indices


def normalize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data per channel using training set statistics.
    """
    print("\nNormalizing data...")
    
    n_train_channels = X_train.shape[1]  # 18
    n_test_channels = X_test.shape[1]    # 6
    
    # Training set normalization (18 channels)
    train_means = np.zeros(n_train_channels)
    train_stds = np.zeros(n_train_channels)
    
    for c in range(n_train_channels):
        channel_data = X_train[:, c, :, :].flatten()
        train_means[c] = np.mean(channel_data)
        train_stds[c] = np.std(channel_data)
        if train_stds[c] < 1e-6:
            train_stds[c] = 1.0
    
    print(f"  Training channels: {n_train_channels}")
    print(f"  Mean range: {train_means.min():.2f} to {train_means.max():.2f}")
    print(f"  Std range: {train_stds.min():.2f} to {train_stds.max():.2f}")
    
    # Normalize training and validation
    X_train_norm = np.zeros_like(X_train, dtype=np.float32)
    X_val_norm = np.zeros_like(X_val, dtype=np.float32)
    
    for c in range(n_train_channels):
        X_train_norm[:, c, :, :] = (X_train[:, c, :, :] - train_means[c]) / train_stds[c]
        X_val_norm[:, c, :, :] = (X_val[:, c, :, :] - train_means[c]) / train_stds[c]
    
    # For test set (6 channels), use band-wise mean of training stats
    # Training has 18 channels = 6 bands × 3 dates
    # Test has 6 channels = 6 bands × 1 date
    test_means = np.zeros(n_test_channels)
    test_stds = np.zeros(n_test_channels)
    
    for band in range(n_test_channels):  # 6 bands
        # Average stats across 3 training dates for this band
        band_means = [train_means[band + d * 6] for d in range(3)]
        band_stds = [train_stds[band + d * 6] for d in range(3)]
        test_means[band] = np.mean(band_means)
        test_stds[band] = np.mean(band_stds)
    
    X_test_norm = np.zeros_like(X_test, dtype=np.float32)
    for c in range(n_test_channels):
        X_test_norm[:, c, :, :] = (X_test[:, c, :, :] - test_means[c]) / test_stds[c]
    
    print(f"  Test channels: {n_test_channels}")
    
    return X_train_norm, X_val_norm, X_test_norm, train_means, train_stds


def verify_data_integrity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Check data for issues."""
    print("\nVerifying data integrity...")
    
    # Check for NaN values
    train_nan = np.isnan(X_train).sum()
    test_nan = np.isnan(X_test).sum()
    
    if train_nan > 0:
        print(f"  WARNING: {train_nan} NaN values in training data")
    if test_nan > 0:
        print(f"  WARNING: {test_nan} NaN values in test data")
    
    print(f"  X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  X_test range: [{X_test.min():.2f}, {X_test.max():.2f}]")
    
    train_sugarcane = y_train.mean() * 100
    test_sugarcane = y_test.mean() * 100
    
    print(f"  Training sugarcane pixels: {train_sugarcane:.1f}%")
    print(f"  Test sugarcane pixels: {test_sugarcane:.1f}%")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def prepare_training_dataset():
    """Main function to prepare training dataset with randomized test dates."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 6 (v2): PREPARE DATASET WITH RANDOMIZED TEST DATES")
    print_separator()
    print()
    
    # Load all patches
    X, y, metadata = load_all_patches(PATCHES_DIR)
    
    # Create randomized split
    X_train, y_train, X_val, y_val, X_test, y_test, test_date_indices = create_randomized_split(
        X, y,
        validation_split=VALIDATION_SPLIT
    )
    
    # Normalize data
    X_train_norm, X_val_norm, X_test_norm, means, stds = normalize_data(
        X_train, X_val, X_test
    )
    
    # Verify data integrity
    verify_data_integrity(X_train_norm, y_train, X_test_norm, y_test)
    
    # Save dataset
    print_separator("-")
    print("Saving dataset...")
    print_separator("-")
    
    np.savez_compressed(
        OUTPUT_FILE,
        # Training data (18 channels = 6 bands × 3 dates)
        X_train=X_train_norm.astype(np.float32),
        y_train=y_train.astype(np.uint8),
        
        # Validation data (18 channels)
        X_val=X_val_norm.astype(np.float32),
        y_val=y_val.astype(np.uint8),
        
        # Test data (6 channels = 6 bands × 1 random date per patch)
        X_test=X_test_norm.astype(np.float32),
        y_test=y_test.astype(np.uint8),
        
        # Which date was used for test per patch
        test_date_indices=test_date_indices.astype(np.uint8),
        
        # Normalization parameters
        norm_means=means,
        norm_stds=stds,
        
        # Metadata
        n_patches=len(metadata),
        dates=DATE_NAMES
    )
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File size: {file_size_mb:.1f} MB")
    
    # Summary
    total_time = time.time() - start_time
    
    print()
    print_separator()
    print("DATASET SUMMARY")
    print_separator()
    print()
    print(f"Training set:")
    print(f"  X_train: {X_train_norm.shape} (N × 18_channels × 224 × 224)")
    print(f"  y_train: {y_train.shape}")
    print()
    print(f"Validation set:")
    print(f"  X_val: {X_val_norm.shape}")
    print(f"  y_val: {y_val.shape}")
    print()
    print(f"Test set (RANDOMIZED test dates per patch):")
    print(f"  X_test: {X_test_norm.shape} (N × 6_channels × 224 × 224)")
    print(f"  y_test: {y_test.shape}")
    print()
    print(f"Total processing time: {total_time:.1f} seconds")
    print()
    print("Dataset ready for Phase 7 (training with randomized approach).")


if __name__ == "__main__":
    prepare_training_dataset()
