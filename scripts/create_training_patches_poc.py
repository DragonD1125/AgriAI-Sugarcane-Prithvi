#!/usr/bin/env python3
"""
Phase 5: Create Training Patches & Ground Truth Masks (PoC)

This script creates 224×224 pixel patches from:
1. Sentinel-2 imagery (4 temporal dates × 6 bands = 24 channels)
2. Di Tommaso ground truth TIFs (Band 2 = sugarcane labels)

Patches are sampled from coordinates identified in Phase 2-3 that fall
within the PoC MGRS tile 44RDM (Uttar Pradesh, India).

Output: NPZ files with X (imagery) and y (labels) for training

Author: Agri AI Project
Date: 2024
"""

import json
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from rasterio.crs import CRS


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
SENTINEL_DIR = BASE_DIR / "data" / "sentinel2_poc"
COORDS_CSV = BASE_DIR / "data" / "sugarcane_field_locations.csv"
MGRS_JSON = BASE_DIR / "data" / "mgrs_tiles_needed.json"
TIFF_DIR = BASE_DIR / "data" / "india_GEDIS2_v1"
OUTPUT_DIR = BASE_DIR / "data" / "patches_poc"

# Patch configuration
PATCH_SIZE = 224  # pixels
HALF_PATCH = PATCH_SIZE // 2  # 112 pixels

# Sentinel-2 file mapping (temporal order)
S2_FILES = [
    'S2_PoC_20200315_planting.tif',      # Date 1: Training
    'S2_PoC_20200815_peak_growth.tif',   # Date 2: Training
    'S2_PoC_20201215_pre_harvest.tif',   # Date 3: Training
    'S2_PoC_20210130_post_harvest.tif',  # Date 4: TESTING (temporal independence)
]

# Band indices in stacked Sentinel-2 files (0-indexed)
# Bands: B02, B03, B04, B8A, B11, B12, SCL
BAND_INDICES = [0, 1, 2, 3, 4, 5]  # First 6 bands (excluding SCL)
SCL_INDEX = 6  # Scene Classification Layer

# Di Tommaso band for ground truth
DITOMMASO_SUGARCANE_BAND = 2  # 1-indexed in rasterio (Band 2 = sugarcane labels)

# Maximum patches per tile to generate
MAX_PATCHES = 200

# Minimum pixel spacing between patches (to avoid overlap)
MIN_SPACING = 224  # 1 patch width


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


def load_poc_tile_info() -> Dict:
    """Load PoC tile bounds from Phase 3 output."""
    with open(MGRS_JSON, 'r') as f:
        data = json.load(f)
    return data['poc_tile']


def load_coordinates_for_tile(poc_tile: Dict) -> pd.DataFrame:
    """Load sugarcane coordinates within the PoC tile bounds."""
    bounds = poc_tile['bounds']
    
    print(f"Loading coordinates within bounds:")
    print(f"  Lat: {bounds['lat_min']:.4f} to {bounds['lat_max']:.4f}")
    print(f"  Lon: {bounds['lon_min']:.4f} to {bounds['lon_max']:.4f}")
    
    # Load in chunks to handle large file
    chunks = []
    for chunk in pd.read_csv(COORDS_CSV, chunksize=100000):
        mask = (
            (chunk['latitude'] >= bounds['lat_min']) &
            (chunk['latitude'] <= bounds['lat_max']) &
            (chunk['longitude'] >= bounds['lon_min']) &
            (chunk['longitude'] <= bounds['lon_max'])
        )
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)
    
    if not chunks:
        raise ValueError("No coordinates found within PoC tile bounds")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Found {len(df):,} coordinates in PoC tile")
    
    return df


def sample_coordinates(df: pd.DataFrame, max_patches: int, min_spacing: int) -> pd.DataFrame:
    """
    Sample coordinates with minimum spacing to avoid overlapping patches.
    Uses a simple grid-based sampling approach.
    """
    print(f"\nSampling coordinates with min spacing of {min_spacing} pixels...")
    
    # Convert min_spacing to approximate degrees (10m per pixel)
    spacing_deg = (min_spacing * 10) / 111000  # ~111km per degree
    
    # Create a grid-based sample
    df = df.copy()
    df['lat_bin'] = (df['latitude'] / spacing_deg).astype(int)
    df['lon_bin'] = (df['longitude'] / spacing_deg).astype(int)
    
    # Keep only one coordinate per grid cell
    sampled = df.groupby(['lat_bin', 'lon_bin']).first().reset_index()
    
    # Further sample if still too many
    if len(sampled) > max_patches:
        sampled = sampled.sample(n=max_patches, random_state=42)
    
    print(f"  Sampled {len(sampled)} patches from {len(df)} coordinates")
    
    return sampled[['latitude', 'longitude', 'source_file']].reset_index(drop=True)


# ============================================================================
# PATCH EXTRACTION FUNCTIONS
# ============================================================================

def get_pixel_coords(lat: float, lon: float, transform, crs) -> Tuple[int, int]:
    """
    Convert lat/lon to pixel coordinates.
    Returns (row, col) in the raster.
    """
    if crs.is_geographic:
        # Geographic CRS (degrees) - direct conversion
        col = int((lon - transform.c) / transform.a)
        row = int((lat - transform.f) / transform.e)
    else:
        # Projected CRS (meters) - need to transform coordinates
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
    
    return row, col


def extract_patch_from_sentinel(
    lat: float,
    lon: float,
    s2_files: List[Path],
    patch_size: int = PATCH_SIZE
) -> Optional[np.ndarray]:
    """
    Extract a patch from all temporal Sentinel-2 files.
    
    Returns:
        numpy array of shape (6, 4, 224, 224) - bands × dates × height × width
        or None if extraction fails
    """
    half = patch_size // 2
    patches = []
    
    for s2_file in s2_files:
        with rasterio.open(s2_file) as src:
            # Get pixel coordinates
            row, col = get_pixel_coords(lat, lon, src.transform, src.crs)
            
            # Define window
            window = Window(
                col_off=col - half,
                row_off=row - half,
                width=patch_size,
                height=patch_size
            )
            
            # Check bounds
            if (window.col_off < 0 or window.row_off < 0 or
                window.col_off + window.width > src.width or
                window.row_off + window.height > src.height):
                return None
            
            # Read 6 spectral bands (excluding SCL)
            try:
                data = src.read(list(range(1, 7)), window=window)  # Bands 1-6
                if data.shape != (6, patch_size, patch_size):
                    return None
                patches.append(data)
            except Exception:
                return None
    
    if len(patches) != len(s2_files):
        return None
    
    # Stack: (6, 4, 224, 224)
    return np.stack(patches, axis=1)


def extract_ground_truth(
    lat: float,
    lon: float,
    source_file: str,
    tiff_dir: Path,
    patch_size: int = PATCH_SIZE
) -> Optional[np.ndarray]:
    """
    Extract ground truth mask from Di Tommaso TIF.
    
    Returns:
        numpy array of shape (224, 224) with binary labels (0/1)
        or None if extraction fails
    """
    half = patch_size // 2
    
    # Find the source TIF file
    tif_path = tiff_dir / source_file
    if not tif_path.exists():
        # Try without .tif extension
        tif_path = tiff_dir / source_file.replace('.tif', '')
        if not tif_path.exists():
            return None
    
    try:
        with rasterio.open(tif_path) as src:
            # Get pixel coordinates (Di Tommaso TIFs are in EPSG:4326)
            row, col = get_pixel_coords(lat, lon, src.transform, src.crs)
            
            # Define window
            window = Window(
                col_off=col - half,
                row_off=row - half,
                width=patch_size,
                height=patch_size
            )
            
            # Check bounds
            if (window.col_off < 0 or window.row_off < 0 or
                window.col_off + window.width > src.width or
                window.row_off + window.height > src.height):
                return None
            
            # Read sugarcane band (Band 2)
            data = src.read(DITOMMASO_SUGARCANE_BAND, window=window)
            
            if data.shape != (patch_size, patch_size):
                return None
            
            # Ensure binary (0/1)
            data = (data > 0).astype(np.uint8)
            
            return data
            
    except Exception:
        return None


def extract_scl_mask(
    lat: float,
    lon: float,
    s2_files: List[Path],
    patch_size: int = PATCH_SIZE
) -> Optional[np.ndarray]:
    """
    Extract SCL (Scene Classification Layer) mask for cloud filtering.
    
    Returns:
        numpy array of shape (4, 224, 224) with SCL values per date
        or None if extraction fails
    """
    half = patch_size // 2
    masks = []
    
    for s2_file in s2_files:
        with rasterio.open(s2_file) as src:
            row, col = get_pixel_coords(lat, lon, src.transform, src.crs)
            
            window = Window(
                col_off=col - half,
                row_off=row - half,
                width=patch_size,
                height=patch_size
            )
            
            if (window.col_off < 0 or window.row_off < 0 or
                window.col_off + window.width > src.width or
                window.row_off + window.height > src.height):
                return None
            
            try:
                # Read SCL band (band 7 in our stacked files)
                scl = src.read(7, window=window)
                masks.append(scl)
            except Exception:
                return None
    
    if len(masks) != len(s2_files):
        return None
    
    return np.stack(masks, axis=0)


def compute_valid_pixel_mask(scl: np.ndarray) -> np.ndarray:
    """
    Compute valid pixel mask from SCL.
    
    SCL values:
        0 = No data, 1 = Saturated/Defective, 2 = Dark area, 3 = Cloud shadow
        4 = Vegetation, 5 = Bare soil, 6 = Water, 7 = Unclassified
        8 = Cloud medium, 9 = Cloud high, 10 = Thin cirrus, 11 = Snow/Ice
    
    Valid pixels: 4 (vegetation), 5 (bare soil), 6 (water), 7 (unclassified)
    """
    valid_values = [4, 5, 6, 7]
    mask = np.isin(scl, valid_values)
    return mask


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def create_patches():
    """Main function to create training patches."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 5: CREATE TRAINING PATCHES (PoC)")
    print_separator()
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load PoC tile info
    print("Loading PoC tile information...")
    poc_tile = load_poc_tile_info()
    print(f"  Tile ID: {poc_tile['tile_id']}")
    print()
    
    # Load and sample coordinates
    coords = load_coordinates_for_tile(poc_tile)
    sampled_coords = sample_coordinates(coords, MAX_PATCHES, MIN_SPACING)
    
    # Get Sentinel-2 files
    s2_files = [SENTINEL_DIR / f for f in S2_FILES]
    for f in s2_files:
        if not f.exists():
            raise FileNotFoundError(f"Sentinel-2 file not found: {f}")
    print(f"\nSentinel-2 files: {len(s2_files)}")
    for f in s2_files:
        print(f"  {f.name}")
    
    # Extract patches
    print_separator("-")
    print("Extracting patches...")
    print_separator("-")
    
    successful = 0
    failed = 0
    
    for idx, row in sampled_coords.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        source_file = row['source_file']
        
        # Extract Sentinel-2 patch
        X = extract_patch_from_sentinel(lat, lon, s2_files)
        if X is None:
            failed += 1
            continue
        
        # Extract ground truth
        y = extract_ground_truth(lat, lon, source_file, TIFF_DIR)
        if y is None:
            failed += 1
            continue
        
        # Extract SCL for quality check
        scl = extract_scl_mask(lat, lon, s2_files)
        
        # Compute valid pixel percentage
        if scl is not None:
            valid_mask = compute_valid_pixel_mask(scl)
            valid_pct = valid_mask.mean() * 100
        else:
            valid_pct = 100.0  # Assume valid if SCL not available
        
        # Skip patches with too many clouds (< 50% valid)
        if valid_pct < 50:
            failed += 1
            continue
        
        # Compute sugarcane percentage in ground truth
        sugarcane_pct = y.mean() * 100
        
        # Save patch
        patch_file = OUTPUT_DIR / f"patch_{idx:04d}.npz"
        np.savez_compressed(
            patch_file,
            X=X.astype(np.float32),  # (6, 4, 224, 224)
            y=y.astype(np.uint8),     # (224, 224)
            latitude=lat,
            longitude=lon,
            source_file=source_file,
            valid_pct=valid_pct,
            sugarcane_pct=sugarcane_pct,
            dates=['2020-03-15', '2020-09-28', '2020-12-15', '2021-02-25']
        )
        
        successful += 1
        
        if (successful + failed) % 20 == 0:
            print(f"  Progress: {successful + failed}/{len(sampled_coords)}, "
                  f"successful: {successful}, failed: {failed}")
    
    # Summary
    total_time = time.time() - start_time
    
    print_separator()
    print("PATCH EXTRACTION SUMMARY")
    print_separator()
    print(f"Total patches attempted: {len(sampled_coords)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / len(sampled_coords) * 100:.1f}%")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    if successful > 0:
        # Show first patch info
        first_patch = list(OUTPUT_DIR.glob("patch_*.npz"))[0]
        with np.load(first_patch) as data:
            print(f"Sample patch shape:")
            print(f"  X (imagery): {data['X'].shape}")
            print(f"  y (labels): {data['y'].shape}")
            print(f"  Valid pixels: {data['valid_pct']:.1f}%")
            print(f"  Sugarcane pixels: {data['sugarcane_pct']:.1f}%")
    
    print()
    print("Phase 5 complete. Ready for Phase 6 (dataset preparation).")


if __name__ == "__main__":
    create_patches()
