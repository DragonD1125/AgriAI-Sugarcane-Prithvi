#!/usr/bin/env python3
"""
Phase 3: Identify MGRS Tiles for Sugarcane Coordinates

This script converts lat/lon coordinates to MGRS grid references and identifies
which MGRS tiles contain sugarcane fields for Sentinel-2 data download.

MGRS (Military Grid Reference System):
- Each tile is 100km × 100km at equator
- Format: Grid Zone Designator (e.g., 43Q) + 100km Square ID (e.g., GD)
- Example: 43QGD = Grid Zone 43Q, Square GD

Output: JSON file mapping MGRS tiles to coordinate counts and bounds.

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
import pandas as pd

# Use utm library for coordinate conversion
try:
    import utm
    UTM_AVAILABLE = True
except ImportError:
    UTM_AVAILABLE = False
    print("WARNING: utm library not installed. Install with: pip install utm")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
INPUT_CSV = BASE_DIR / "data" / "sugarcane_field_locations.csv"
OUTPUT_JSON = BASE_DIR / "data" / "mgrs_tiles_needed.json"

# Chunk size for processing large CSV
CHUNK_SIZE = 100000  # Process 100K rows at a time

# Minimum coordinates per tile to be considered significant
MIN_COORDS_PER_TILE = 1000


# ============================================================================
# MGRS CONVERSION FUNCTIONS
# ============================================================================

def get_utm_zone(lon: float) -> int:
    """Calculate UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1


def get_utm_band(lat: float) -> str:
    """Calculate UTM latitude band letter."""
    bands = "CDEFGHJKLMNPQRSTUVWX"
    if lat < -80:
        return 'A'
    elif lat > 84:
        return 'Z'
    else:
        idx = int((lat + 80) / 8)
        idx = min(idx, len(bands) - 1)
        return bands[idx]


def get_100km_square_id(easting: float, northing: float, zone: int) -> str:
    """
    Approximate 100km square ID within UTM zone.
    This is a simplified version - actual MGRS uses a more complex grid.
    """
    # Simplified: divide into 100km squares
    col_letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # No I, O
    row_letters = "ABCDEFGHJKLMNPQRSTUV"  # Repeating pattern
    
    # Column (easting): 100km = 100,000m
    col_idx = int(easting / 100000) % len(col_letters)
    
    # Row (northing): 100km = 100,000m, but wraps every 2,000km
    row_idx = int(northing / 100000) % len(row_letters)
    
    return col_letters[col_idx] + row_letters[row_idx]


def latlon_to_mgrs_utm(lat: float, lon: float) -> str:
    """
    Convert lat/lon to MGRS tile ID using utm library.
    Returns format like '43QGD' (zone + band + 100km square)
    """
    if UTM_AVAILABLE:
        try:
            # Get UTM coordinates
            easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
            band = zone_letter
            zone = zone_number
        except Exception:
            # Fallback to manual calculation
            zone = get_utm_zone(lon)
            band = get_utm_band(lat)
            from math import cos, radians
            central_meridian = (zone - 1) * 6 - 180 + 3
            easting = 500000 + (lon - central_meridian) * 111319.5 * cos(radians(lat))
            northing = lat * 110574.3
            if lat < 0:
                northing += 10000000
    else:
        zone = get_utm_zone(lon)
        band = get_utm_band(lat)
        from math import cos, radians
        central_meridian = (zone - 1) * 6 - 180 + 3
        easting = 500000 + (lon - central_meridian) * 111319.5 * cos(radians(lat))
        northing = lat * 110574.3
        if lat < 0:
            northing += 10000000
    
    square_id = get_100km_square_id(easting, northing, zone)
    
    return f"{zone}{band}{square_id}"


def convert_coordinates_to_mgrs(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> pd.Series:
    """
    Convert DataFrame of coordinates to MGRS tile IDs.
    Uses utm library for accurate conversions.
    """
    print(f"  Converting {len(df):,} coordinates to MGRS...")
    
    mgrs_tiles = df.apply(
        lambda row: latlon_to_mgrs_utm(row[lat_col], row[lon_col]),
        axis=1
    )
    
    return mgrs_tiles


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


def process_coordinates_chunked(csv_path: Path) -> Dict:
    """
    Process coordinates in chunks to identify MGRS tiles.
    Returns dictionary with tile statistics.
    """
    print(f"Loading coordinates from: {csv_path}")
    print(f"Processing in chunks of {CHUNK_SIZE:,} rows...")
    print()
    
    # Dictionary to accumulate tile statistics
    tile_stats = {}
    total_coords = 0
    
    # Process in chunks
    chunk_num = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        chunk_num += 1
        chunk_start = time.time()
        
        # Convert to MGRS
        mgrs_tiles = convert_coordinates_to_mgrs(chunk)
        
        # Count tiles in this chunk
        for tile_id in mgrs_tiles:
            if tile_id not in tile_stats:
                tile_stats[tile_id] = {
                    'count': 0,
                    'lat_min': float('inf'),
                    'lat_max': float('-inf'),
                    'lon_min': float('inf'),
                    'lon_max': float('-inf')
                }
            tile_stats[tile_id]['count'] += 1
        
        # Update bounds for each tile
        for idx, tile_id in enumerate(mgrs_tiles):
            lat = chunk.iloc[idx]['latitude']
            lon = chunk.iloc[idx]['longitude']
            tile_stats[tile_id]['lat_min'] = min(tile_stats[tile_id]['lat_min'], lat)
            tile_stats[tile_id]['lat_max'] = max(tile_stats[tile_id]['lat_max'], lat)
            tile_stats[tile_id]['lon_min'] = min(tile_stats[tile_id]['lon_min'], lon)
            tile_stats[tile_id]['lon_max'] = max(tile_stats[tile_id]['lon_max'], lon)
        
        total_coords += len(chunk)
        chunk_time = time.time() - chunk_start
        
        print(f"  Chunk {chunk_num}: processed {len(chunk):,} coords in {chunk_time:.1f}s "
              f"(total: {total_coords:,}, tiles: {len(tile_stats)})")
    
    return tile_stats, total_coords


def analyze_tiles(tile_stats: Dict) -> Dict:
    """
    Analyze tile statistics and prepare output.
    """
    print_separator()
    print("MGRS TILE ANALYSIS")
    print_separator()
    print()
    
    # Filter significant tiles
    significant_tiles = {
        k: v for k, v in tile_stats.items() 
        if v['count'] >= MIN_COORDS_PER_TILE
    }
    
    print(f"Total unique MGRS tiles: {len(tile_stats)}")
    print(f"Significant tiles (≥{MIN_COORDS_PER_TILE:,} coords): {len(significant_tiles)}")
    print()
    
    # Sort by count descending
    sorted_tiles = sorted(
        significant_tiles.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Print top tiles
    print("Top 20 MGRS tiles by coordinate count:")
    print("-" * 70)
    print(f"{'Tile':<10} {'Count':>12} {'Lat Range':<20} {'Lon Range':<20}")
    print("-" * 70)
    
    for tile_id, stats in sorted_tiles[:20]:
        lat_range = f"{stats['lat_min']:.2f}° - {stats['lat_max']:.2f}°"
        lon_range = f"{stats['lon_min']:.2f}° - {stats['lon_max']:.2f}°"
        print(f"{tile_id:<10} {stats['count']:>12,} {lat_range:<20} {lon_range:<20}")
    
    print()
    
    # Select PoC tile (highest count)
    if sorted_tiles:
        poc_tile = sorted_tiles[0][0]
        poc_count = sorted_tiles[0][1]['count']
        print(f"Recommended PoC tile: {poc_tile} ({poc_count:,} coordinates)")
    else:
        poc_tile = None
        print("WARNING: No significant tiles found!")
    
    return sorted_tiles, poc_tile


def save_results(
    tile_stats: Dict,
    sorted_tiles: List,
    poc_tile: str,
    total_coords: int,
    output_path: Path
) -> None:
    """
    Save results to JSON file.
    """
    print_separator()
    print("SAVING RESULTS")
    print_separator()
    print()
    
    # Prepare output structure
    output = {
        'metadata': {
            'total_coordinates': total_coords,
            'total_tiles': len(tile_stats),
            'significant_tiles': len(sorted_tiles),
            'min_coords_threshold': MIN_COORDS_PER_TILE,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'utm_library_used': UTM_AVAILABLE
        },
        'poc_tile': {
            'tile_id': poc_tile,
            'count': tile_stats[poc_tile]['count'] if poc_tile else 0,
            'bounds': {
                'lat_min': tile_stats[poc_tile]['lat_min'] if poc_tile else None,
                'lat_max': tile_stats[poc_tile]['lat_max'] if poc_tile else None,
                'lon_min': tile_stats[poc_tile]['lon_min'] if poc_tile else None,
                'lon_max': tile_stats[poc_tile]['lon_max'] if poc_tile else None
            }
        },
        'tiles': {}
    }
    
    # Add all significant tiles
    for tile_id, stats in sorted_tiles:
        output['tiles'][tile_id] = {
            'count': stats['count'],
            'bounds': {
                'lat_min': round(stats['lat_min'], 6),
                'lat_max': round(stats['lat_max'], 6),
                'lon_min': round(stats['lon_min'], 6),
                'lon_max': round(stats['lon_max'], 6)
            }
        }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main entry point."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 3: IDENTIFY MGRS TILES")
    print_separator()
    print()
    
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"UTM library available: {UTM_AVAILABLE}")
    print()
    
    # Validate input
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        sys.exit(1)
    
    # Process coordinates
    tile_stats, total_coords = process_coordinates_chunked(INPUT_CSV)
    
    # Analyze tiles
    sorted_tiles, poc_tile = analyze_tiles(tile_stats)
    
    # Save results
    save_results(tile_stats, sorted_tiles, poc_tile, total_coords, OUTPUT_JSON)
    
    # Summary
    total_time = time.time() - start_time
    print()
    print_separator()
    print("PHASE 3 COMPLETE")
    print_separator()
    print()
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Coordinates processed: {total_coords:,}")
    print(f"MGRS tiles identified: {len(sorted_tiles)}")
    print(f"PoC tile selected: {poc_tile}")
    print()
    print("Next step: Run download_sentinel2_poc.py to download Sentinel-2 data")


if __name__ == "__main__":
    main()
