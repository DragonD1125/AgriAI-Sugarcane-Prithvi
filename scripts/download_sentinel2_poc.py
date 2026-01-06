#!/usr/bin/env python3
"""
Phase 4: Download Sentinel-2 Data for PoC MGRS Tile

This script downloads Sentinel-2 imagery for the PoC tile identified in Phase 3.
Uses Microsoft Planetary Computer STAC API (no authentication required).

Sentinel-2 Bands for Prithvi:
- B02 (Blue): 490 nm
- B03 (Green): 560 nm  
- B04 (Red): 665 nm
- B8A (Narrow NIR): 865 nm
- B11 (SWIR1): 1610 nm
- B12 (SWIR2): 2190 nm

Author: Agri AI Project
Date: 2024
"""

import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.crs import CRS
from pystac_client import Client
import planetary_computer as pc


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
MGRS_JSON = BASE_DIR / "data" / "mgrs_tiles_needed.json"
OUTPUT_DIR = BASE_DIR / "data" / "sentinel2_poc"

# Sentinel-2 bands for Prithvi (in order)
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
BAND_NAMES = ['Blue', 'Green', 'Red', 'NIR_Narrow', 'SWIR1', 'SWIR2']

# Also get SCL for cloud masking
BANDS_WITH_SCL = BANDS + ['SCL']

# Temporal dates for crop cycle (2020-2021)
# Priority: find clearest images with minimal cloud cover around these timeframes
TARGET_DATES = [
    ('2020-03-15', 'planting'),      # Planting/Early growth - March 2020
    ('2020-08-15', 'peak_growth'),   # Peak vegetative growth - August 2020
    ('2020-12-15', 'pre_harvest'),   # Pre-harvest - December 2020
    ('2021-01-30', 'post_harvest'),  # Post-harvest (for TESTING) - Jan/Feb 2021
]

# Date window (±days around target date)
DATE_WINDOW_DAYS = 30

# Maximum cloud cover percentage
MAX_CLOUD_COVER = 30

# Planetary Computer STAC endpoint
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_separator(char: str = "=", length: int = 70) -> None:
    print(char * length)


def load_poc_tile_info() -> Dict:
    """Load PoC tile information from Phase 3 output."""
    if not MGRS_JSON.exists():
        raise FileNotFoundError(f"MGRS tiles JSON not found: {MGRS_JSON}")
    
    with open(MGRS_JSON, 'r') as f:
        data = json.load(f)
    
    poc_tile = data.get('poc_tile', {})
    if not poc_tile.get('tile_id'):
        raise ValueError("No PoC tile found in MGRS JSON")
    
    return poc_tile


def get_date_range(target_date: str, window_days: int = DATE_WINDOW_DAYS) -> Tuple[str, str]:
    """Get date range around target date."""
    target = datetime.strptime(target_date, '%Y-%m-%d')
    start = target - timedelta(days=window_days)
    end = target + timedelta(days=window_days)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def search_sentinel2(
    bbox: List[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int = MAX_CLOUD_COVER
) -> List:
    """
    Search for Sentinel-2 imagery using Microsoft Planetary Computer STAC API.
    """
    catalog = Client.open(PC_STAC_URL)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover}
        }
    )
    
    return list(search.items())


def find_covering_tile(items: List, bbox: List[float]) -> Optional[object]:
    """
    Find a STAC item that actually covers our target bbox center.
    """
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    
    for item in items:
        item_bbox = item.bbox
        # Check if item bbox contains our center point
        if (item_bbox[0] <= center_lon <= item_bbox[2] and
            item_bbox[1] <= center_lat <= item_bbox[3]):
            return item
    
    return None


def download_and_clip_band(
    signed_item,
    band_name: str,
    bbox_4326: List[float],
    output_path: Path
) -> bool:
    """
    Download a band and clip to bounding box.
    Transforms bbox from WGS84 to raster CRS.
    """
    # Get asset
    asset = signed_item.assets.get(band_name)
    if not asset:
        print(f"      Asset {band_name} not found")
        return False
    
    url = asset.href
    
    try:
        with rasterio.open(url) as src:
            # Transform bbox from WGS84 (EPSG:4326) to raster CRS
            src_crs = src.crs
            bbox_transformed = transform_bounds(
                CRS.from_epsg(4326),  # Source: WGS84
                src_crs,               # Destination: raster CRS (UTM)
                *bbox_4326
            )
            
            # Check if transformed bbox intersects with raster bounds
            rb = src.bounds
            if (bbox_transformed[0] > rb.right or bbox_transformed[2] < rb.left or
                bbox_transformed[1] > rb.top or bbox_transformed[3] < rb.bottom):
                print(f"      Transformed bbox does not intersect, skipping...")
                return False
            
            # Clip to raster bounds
            clip_bbox = [
                max(bbox_transformed[0], rb.left),
                max(bbox_transformed[1], rb.bottom),
                min(bbox_transformed[2], rb.right),
                min(bbox_transformed[3], rb.top)
            ]
            
            # Calculate window from clipped bbox
            window = from_bounds(*clip_bbox, src.transform)
            
            # Round window to integer pixels
            window = Window(
                int(window.col_off),
                int(window.row_off),
                int(window.width) + 1,  # Add 1 to ensure we capture the edge
                int(window.height) + 1
            )
            
            # Ensure window is within raster bounds
            window = Window(
                max(0, window.col_off),
                max(0, window.row_off),
                min(window.width, src.width - window.col_off),
                min(window.height, src.height - window.row_off)
            )
            
            if window.width <= 0 or window.height <= 0:
                print(f"      Empty window after clipping, skipping...")
                return False
            
            # Read windowed data
            data = src.read(1, window=window)
            
            if data.size == 0:
                print(f"      No data in window, skipping...")
                return False
            
            # Calculate new transform for the window
            transform = src.window_transform(window)
            
            # Create output profile
            profile = src.profile.copy()
            profile.update(
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                transform=transform,
                compress='lzw',
                tiled=True
            )
            
            # Write to file
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            return True
            
    except Exception as e:
        print(f"      Error: {e}")
        traceback.print_exc()
        return False


def download_sentinel2_for_date(
    poc_tile: Dict,
    target_date: str,
    phase_name: str,
    output_dir: Path
) -> Optional[Path]:
    """
    Download Sentinel-2 data for a specific date.
    """
    bounds = poc_tile['bounds']
    bbox = [
        bounds['lon_min'],
        bounds['lat_min'],
        bounds['lon_max'],
        bounds['lat_max']
    ]
    
    # Get date range
    start_date, end_date = get_date_range(target_date)
    
    print(f"  Searching for Sentinel-2 imagery...")
    print(f"    Bbox (WGS84): {[round(b, 4) for b in bbox]}")
    print(f"    Date range: {start_date} to {end_date}")
    
    # Search for items
    items = search_sentinel2(bbox, start_date, end_date)
    
    if not items:
        print(f"  WARNING: No imagery found for date range")
        return None
    
    print(f"  Found {len(items)} scenes")
    
    # Sort by cloud cover
    items_sorted = sorted(items, key=lambda x: x.properties.get('eo:cloud_cover', 100))
    
    # Find tile that covers our area
    best_item = find_covering_tile(items_sorted, bbox)
    
    if not best_item:
        print(f"  WARNING: No tile found that covers our target area")
        print(f"  Available tiles:")
        for item in items_sorted[:5]:
            print(f"    - {item.id}: bbox={item.bbox}")
        return None
    
    print(f"\n  Using scene: {best_item.id}")
    print(f"    Date: {best_item.datetime}")
    print(f"    Cloud cover: {best_item.properties.get('eo:cloud_cover', 'N/A')}%")
    
    # Sign the item
    signed_item = pc.sign(best_item)
    
    # Download each band
    output_bands = []
    
    for band in BANDS_WITH_SCL:
        print(f"    Downloading band {band}...")
        band_path = output_dir / f"temp_{band}_{phase_name}.tif"
        
        try:
            success = download_and_clip_band(signed_item, band, bbox, band_path)
            if success:
                output_bands.append(band_path)
                size_kb = band_path.stat().st_size / 1024
                print(f"      ✓ Success ({size_kb:.0f} KB)")
            else:
                print(f"      ✗ Failed")
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    # Check if we got enough bands
    if len(output_bands) >= len(BANDS):  # At least 6 core bands
        print(f"\n  Successfully downloaded {len(output_bands)}/{len(BANDS_WITH_SCL)} bands")
        
        # Stack bands into single file
        output_file = output_dir / f"S2_PoC_{target_date.replace('-', '')}_{phase_name}.tif"
        
        print(f"  Stacking bands...")
        stack_bands(output_bands, output_file)
        
        # Clean up temp files
        for temp_file in output_bands:
            if temp_file.exists():
                temp_file.unlink()
        
        print(f"  ✓ Saved to: {output_file}")
        return output_file
    else:
        print(f"\n  ✗ Only {len(output_bands)}/{len(BANDS)} bands downloaded")
        # Clean up partial downloads
        for temp_file in output_bands:
            if temp_file.exists():
                temp_file.unlink()
        return None


def stack_bands(band_files: List[Path], output_path: Path) -> None:
    """
    Stack individual band files into a multi-band GeoTIFF.
    Resamples bands to match the first band's resolution.
    """
    if not band_files:
        raise ValueError("No band files to stack")
    
    # Read first band to get reference profile
    with rasterio.open(band_files[0]) as ref_src:
        ref_profile = ref_src.profile.copy()
        ref_data = ref_src.read(1)
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_height = ref_src.height
        ref_width = ref_src.width
    
    # Update profile for multi-band output
    ref_profile.update(
        count=len(band_files),
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    # Create output and write all bands
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        for i, band_file in enumerate(band_files, 1):
            with rasterio.open(band_file) as src:
                data = src.read(1)
                
                # Resample if dimensions don't match (20m bands like B11, B12, SCL)
                if data.shape != (ref_height, ref_width):
                    # Resample to match reference
                    resampled = np.zeros((ref_height, ref_width), dtype=data.dtype)
                    reproject(
                        source=data,
                        destination=resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    dst.write(resampled, i)
                else:
                    dst.write(data, i)
        
        # Set band descriptions
        band_names = BANDS + ['SCL'] if len(band_files) == len(BANDS_WITH_SCL) else BANDS[:len(band_files)]
        dst.descriptions = tuple(band_names)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main download function."""
    start_time = time.time()
    
    print_separator()
    print("PHASE 4: DOWNLOAD SENTINEL-2 DATA (PoC)")
    print_separator()
    print()
    
    # Load PoC tile info
    print("Loading PoC tile information...")
    poc_tile = load_poc_tile_info()
    
    print(f"  Tile ID: {poc_tile['tile_id']}")
    print(f"  Coordinates: {poc_tile['count']:,}")
    bounds = poc_tile['bounds']
    print(f"  Bounds:")
    print(f"    Lat: {bounds['lat_min']:.4f}° to {bounds['lat_max']:.4f}°")
    print(f"    Lon: {bounds['lon_min']:.4f}° to {bounds['lon_max']:.4f}°")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Download for each date
    downloaded_files = []
    for target_date, phase_name in TARGET_DATES:
        print_separator("-")
        print(f"Phase: {phase_name} ({target_date})")
        print_separator("-")
        
        try:
            output_file = download_sentinel2_for_date(
                poc_tile, target_date, phase_name, OUTPUT_DIR
            )
            if output_file:
                downloaded_files.append(output_file)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
        
        print()
    
    # Summary
    print_separator()
    print("DOWNLOAD SUMMARY")
    print_separator()
    print(f"Files downloaded: {len(downloaded_files)}/{len(TARGET_DATES)}")
    for f in downloaded_files:
        size = f.stat().st_size / (1024 * 1024)
        print(f"  ✓ {f.name}: {size:.1f} MB")
    
    # Timing
    total_time = time.time() - start_time
    print()
    print_separator()
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print_separator()
    
    if len(downloaded_files) >= len(TARGET_DATES) - 1:  # At least 3 of 4
        print("\nPhase 4 complete. Ready for Phase 5 (patch creation).")
    else:
        print("\nWARNING: Not all dates downloaded. Check errors above.")


if __name__ == "__main__":
    main()
