#!/usr/bin/env python3
"""
Wheat PoC - Phase 1: Extract Wheat Coordinates from CROPGRIDS

Extracts lat/lon coordinates of wheat fields in India from the
CROPGRIDSv1.08_wheat.nc NetCDF file.

Author: Agri AI Project
Date: 2026
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
NC_FILE = BASE_DIR / "data" / "cropgrids" / "CROPGRIDSv1.08_NC_maps" / "CROPGRIDSv1.08_NC_maps" / "CROPGRIDSv1.08_wheat.nc"
OUTPUT_FILE = BASE_DIR / "data" / "wheat_field_locations.csv"

# India bounding box
INDIA_LAT_MIN = 8.0
INDIA_LAT_MAX = 35.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 97.0

# Minimum cropland area threshold (hectares)
MIN_AREA_THRESHOLD = 0.1

# Maximum coordinates to extract (for memory efficiency)
MAX_COORDS = 500_000


# ============================================================================
# MAIN EXTRACTION
# ============================================================================

def extract_wheat_coordinates():
    """Extract wheat field coordinates from CROPGRIDS NetCDF."""
    start_time = time.time()
    
    print("=" * 70)
    print("WHEAT PoC - PHASE 1: EXTRACT COORDINATES")
    print("=" * 70)
    print()
    
    print(f"Loading: {NC_FILE}")
    ds = xr.open_dataset(NC_FILE)
    
    print("\nDataset variables:")
    for var in ds.data_vars:
        print(f"  - {var}")
    
    # Get cropland area data
    # CROPGRIDS uses 'croparea' or similar variable name
    area_var = None
    for var in ['croparea', 'cropland_area', 'area', 'crop_area']:
        if var in ds.data_vars:
            area_var = var
            break
    
    if area_var is None:
        # Use first available variable
        area_var = list(ds.data_vars)[0]
    
    print(f"\nUsing variable: {area_var}")
    
    # Get latitude and longitude coordinates
    lat = ds['lat'].values
    lon = ds['lon'].values
    data = ds[area_var].values
    
    print(f"Global data shape: {data.shape}")
    print(f"Lat range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"Lon range: {lon.min():.2f} to {lon.max():.2f}")
    
    # Find India subset indices
    lat_mask = (lat >= INDIA_LAT_MIN) & (lat <= INDIA_LAT_MAX)
    lon_mask = (lon >= INDIA_LON_MIN) & (lon <= INDIA_LON_MAX)
    
    lat_india = lat[lat_mask]
    lon_india = lon[lon_mask]
    
    # Get India subset of data
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    india_data = data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    
    print(f"\nIndia subset:")
    print(f"  Shape: {india_data.shape}")
    print(f"  Lat range: {lat_india.min():.2f} to {lat_india.max():.2f}")
    print(f"  Lon range: {lon_india.min():.2f} to {lon_india.max():.2f}")
    
    # Find pixels with wheat
    valid_mask = (~np.isnan(india_data)) & (india_data >= MIN_AREA_THRESHOLD)
    valid_count = np.count_nonzero(valid_mask)
    
    print(f"  Valid wheat pixels: {valid_count:,}")
    
    # Extract coordinates
    print("\nExtracting coordinates...")
    
    # Get indices where wheat exists
    row_idx, col_idx = np.where(valid_mask)
    
    # Convert to lat/lon
    lats = lat_india[row_idx]
    lons = lon_india[col_idx]
    areas = india_data[row_idx, col_idx]
    
    # Sample if too many
    if len(lats) > MAX_COORDS:
        print(f"  Sampling {MAX_COORDS:,} from {len(lats):,} coordinates...")
        np.random.seed(42)
        indices = np.random.choice(len(lats), MAX_COORDS, replace=False)
        lats = lats[indices]
        lons = lons[indices]
        areas = areas[indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'cropland_area': areas,
        'crop_type': 'wheat'
    })
    
    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Coordinates extracted: {len(df):,}")
    print(f"Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.1f} MB")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    
    ds.close()
    
    return df


if __name__ == "__main__":
    extract_wheat_coordinates()
