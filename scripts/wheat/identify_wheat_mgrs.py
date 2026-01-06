#!/usr/bin/env python3
"""
Wheat PoC - Phase 2: Identify MGRS Tiles

Converts wheat coordinates to MGRS tiles and selects a PoC tile
in the wheat belt region (Punjab/Haryana/UP).

Author: Agri AI Project
Date: 2026
"""

import json
import time
from pathlib import Path
from collections import Counter
import pandas as pd
import utm


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
COORDS_FILE = BASE_DIR / "data" / "wheat_field_locations.csv"
OUTPUT_FILE = BASE_DIR / "data" / "wheat_mgrs_tiles.json"

# Minimum coordinates per tile to be considered significant
MIN_COORDS_PER_TILE = 100


# ============================================================================
# MGRS CONVERSION
# ============================================================================

def latlon_to_mgrs(lat: float, lon: float) -> str:
    """Convert lat/lon to MGRS tile ID (without fine grid)."""
    try:
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        
        # Calculate 100km square ID
        col_letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
        row_letters = 'ABCDEFGHJKLMNPQRSTUV'
        
        set_number = (zone_number - 1) % 6
        
        col_idx = int(easting / 100000) - 1
        if set_number % 2 == 0:
            col_idx = (col_idx + 0) % 8
        else:
            col_idx = (col_idx + 8) % 8
        
        row_idx = int(northing / 100000) % 20
        
        col_letter = col_letters[col_idx % len(col_letters)]
        row_letter = row_letters[row_idx % len(row_letters)]
        
        return f"{zone_number}{zone_letter}{col_letter}{row_letter}"
    except Exception:
        return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def identify_mgrs_tiles():
    """Identify MGRS tiles containing wheat fields."""
    start_time = time.time()
    
    print("=" * 70)
    print("WHEAT PoC - PHASE 2: IDENTIFY MGRS TILES")
    print("=" * 70)
    print()
    
    # Load coordinates
    print(f"Loading: {COORDS_FILE}")
    df = pd.read_csv(COORDS_FILE)
    print(f"Loaded {len(df):,} coordinates")
    print()
    
    # Convert to MGRS tiles
    print("Converting to MGRS tiles...")
    mgrs_tiles = []
    
    for i, (lat, lon) in enumerate(zip(df['latitude'], df['longitude'])):
        tile = latlon_to_mgrs(lat, lon)
        mgrs_tiles.append(tile)
        
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,}/{len(df):,}")
    
    df['mgrs_tile'] = mgrs_tiles
    
    # Count per tile
    tile_counts = Counter(mgrs_tiles)
    
    # Filter significant tiles
    significant_tiles = {
        tile: count 
        for tile, count in tile_counts.items() 
        if tile and count >= MIN_COORDS_PER_TILE
    }
    
    print(f"\nTotal MGRS tiles: {len(tile_counts)}")
    print(f"Significant tiles (>={MIN_COORDS_PER_TILE} coords): {len(significant_tiles)}")
    
    # Find top tiles
    sorted_tiles = sorted(significant_tiles.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 tiles by wheat density:")
    for tile, count in sorted_tiles[:10]:
        print(f"  {tile}: {count:,} coordinates")
    
    # Select PoC tile (highest density)
    poc_tile_id = sorted_tiles[0][0]
    poc_count = sorted_tiles[0][1]
    
    # Get bounds for PoC tile
    poc_coords = df[df['mgrs_tile'] == poc_tile_id]
    poc_bounds = {
        'lat_min': float(poc_coords['latitude'].min()),
        'lat_max': float(poc_coords['latitude'].max()),
        'lon_min': float(poc_coords['longitude'].min()),
        'lon_max': float(poc_coords['longitude'].max())
    }
    
    print(f"\nSelected PoC tile: {poc_tile_id}")
    print(f"  Coordinates: {poc_count:,}")
    print(f"  Bounds: Lat [{poc_bounds['lat_min']:.4f}, {poc_bounds['lat_max']:.4f}]")
    print(f"          Lon [{poc_bounds['lon_min']:.4f}, {poc_bounds['lon_max']:.4f}]")
    
    # Save results
    output = {
        'total_tiles': len(tile_counts),
        'significant_tiles_count': len(significant_tiles),
        'significant_tiles': significant_tiles,
        'poc_tile': {
            'tile_id': poc_tile_id,
            'coordinate_count': poc_count,
            'bounds': poc_bounds
        },
        'crop_type': 'wheat'
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("MGRS IDENTIFICATION COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Time: {elapsed:.1f} seconds")
    
    return output


if __name__ == "__main__":
    identify_mgrs_tiles()
