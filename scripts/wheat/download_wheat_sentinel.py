#!/usr/bin/env python3
"""
Wheat PoC - Phase 3: Download Sentinel-2 Data

Downloads 4 temporal Sentinel-2 images for wheat crop cycle:
- Dec 2020: Tillering stage (45 DAS)
- Jan 2021: Heading stage (75 DAS)
- Mar 2021: Maturity stage (115 DAS)
- Apr 2021: Post-harvest (testing date)

Uses Microsoft Planetary Computer STAC API (no authentication required).

Author: Agri AI Project
Date: 2026
"""

import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
import planetary_computer
from pystac_client import Client


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
MGRS_FILE = BASE_DIR / "data" / "wheat_mgrs_tiles.json"
OUTPUT_DIR = BASE_DIR / "data" / "sentinel2_wheat_poc"

# Wheat crop cycle dates (Rabi season 2020-2021)
# Sowing: Oct-Nov 2020, Harvest: Mar-Apr 2021
TARGET_DATES = [
    ('2020-12-15', 'tillering'),    # ~45 DAS - vegetative growth
    ('2021-01-15', 'heading'),      # ~75 DAS - peak biomass
    ('2021-03-01', 'maturity'),     # ~115 DAS - yellowing
    ('2021-04-15', 'post_harvest'), # Post-harvest - bare soil/stubble
]

# Sentinel-2 bands to download
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
BANDS_WITH_SCL = BANDS + ['SCL']

# Cloud cover threshold
MAX_CLOUD = 30

# STAC endpoint
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


# ============================================================================
# SENTINEL-2 DOWNLOAD FUNCTIONS
# ============================================================================

def load_poc_tile() -> Dict:
    """Load PoC tile info."""
    with open(MGRS_FILE, 'r') as f:
        data = json.load(f)
    return data['poc_tile']


def get_date_range(target_date: str, days_buffer: int = 30) -> tuple:
    """Get search date range around target."""
    target = datetime.strptime(target_date, '%Y-%m-%d')
    start = target - timedelta(days=days_buffer)
    end = target + timedelta(days=days_buffer)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')


def search_sentinel2(bbox: List[float], start_date: str, end_date: str, max_cloud: int = 30):
    """Search for Sentinel-2 imagery."""
    catalog = Client.open(STAC_URL)
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )
    
    return list(search.items())


def find_covering_tile(items, bbox: List[float]):
    """Find item that covers the center of our bbox."""
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    
    for item in items:
        item_bbox = item.bbox
        if (item_bbox[0] <= center_lon <= item_bbox[2] and
            item_bbox[1] <= center_lat <= item_bbox[3]):
            return item
    return None


def download_and_clip_band(signed_item, band_name: str, bbox_4326: List[float], output_path: Path) -> bool:
    """Download a band and clip to bbox."""
    asset = signed_item.assets.get(band_name)
    if not asset:
        # Try alternative names
        alt_names = {'B8A': 'nir08', 'B11': 'swir16', 'B12': 'swir22', 'SCL': 'scl'}
        asset = signed_item.assets.get(alt_names.get(band_name, band_name.lower()))
    
    if not asset:
        print(f"      Asset {band_name} not found")
        return False
    
    url = asset.href
    
    try:
        with rasterio.open(url) as src:
            # Transform bbox to raster CRS
            src_crs = src.crs
            bbox_transformed = transform_bounds(
                CRS.from_epsg(4326), src_crs, *bbox_4326
            )
            
            # Check intersection
            rb = src.bounds
            if (bbox_transformed[0] > rb.right or bbox_transformed[2] < rb.left or
                bbox_transformed[1] > rb.top or bbox_transformed[3] < rb.bottom):
                return False
            
            # Clip to raster bounds
            clip_bbox = [
                max(bbox_transformed[0], rb.left),
                max(bbox_transformed[1], rb.bottom),
                min(bbox_transformed[2], rb.right),
                min(bbox_transformed[3], rb.top)
            ]
            
            window = from_bounds(*clip_bbox, src.transform)
            window = Window(
                int(window.col_off), int(window.row_off),
                int(window.width) + 1, int(window.height) + 1
            )
            window = Window(
                max(0, window.col_off), max(0, window.row_off),
                min(window.width, src.width - window.col_off),
                min(window.height, src.height - window.row_off)
            )
            
            if window.width <= 0 or window.height <= 0:
                return False
            
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            
            profile = src.profile.copy()
            profile.update(
                driver='GTiff', height=data.shape[0], width=data.shape[1],
                count=1, dtype=data.dtype, transform=transform,
                compress='lzw', tiled=True
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            return True
            
    except Exception as e:
        print(f"      Error: {e}")
        return False


def stack_bands(band_files: List[Path], output_file: Path) -> bool:
    """Stack multiple band files into single GeoTIFF."""
    if not band_files:
        return False
    
    # Get reference from first band
    with rasterio.open(band_files[0]) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs
    
    # Read and resample all bands
    bands_data = []
    for bf in band_files:
        with rasterio.open(bf) as src:
            if (src.height, src.width) != ref_shape:
                data = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear)
            else:
                data = src.read(1)
            bands_data.append(data)
    
    # Write stacked file
    profile = {
        'driver': 'GTiff',
        'dtype': bands_data[0].dtype,
        'width': ref_shape[1],
        'height': ref_shape[0],
        'count': len(bands_data),
        'crs': ref_crs,
        'transform': ref_transform,
        'compress': 'lzw',
        'tiled': True
    }
    
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i, data in enumerate(bands_data):
            dst.write(data, i + 1)
    
    return True


def download_for_date(poc_tile: Dict, target_date: str, phase_name: str) -> Optional[Path]:
    """Download Sentinel-2 for a specific date."""
    bounds = poc_tile['bounds']
    bbox = [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']]
    
    start_date, end_date = get_date_range(target_date)
    
    print(f"\n  Searching for {phase_name} ({target_date})...")
    print(f"    Date range: {start_date} to {end_date}")
    
    items = search_sentinel2(bbox, start_date, end_date, MAX_CLOUD)
    
    if not items:
        # Try with higher cloud threshold
        items = search_sentinel2(bbox, start_date, end_date, 50)
    
    if not items:
        print(f"    No imagery found")
        return None
    
    print(f"    Found {len(items)} scenes")
    
    # Find covering tile
    covering_item = find_covering_tile(items, bbox)
    if not covering_item:
        # Sort by cloud and take first
        items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100))
        covering_item = items[0]
    
    print(f"    Using: {covering_item.id}")
    print(f"    Date: {covering_item.datetime}")
    print(f"    Cloud: {covering_item.properties.get('eo:cloud_cover', 'N/A')}%")
    
    # Sign item
    signed_item = planetary_computer.sign(covering_item)
    
    # Download bands
    temp_bands = []
    for band in BANDS_WITH_SCL:
        band_path = OUTPUT_DIR / f"temp_{band}_{phase_name}.tif"
        print(f"    Downloading {band}...", end=' ')
        
        if download_and_clip_band(signed_item, band, bbox, band_path):
            print("OK")
            temp_bands.append(band_path)
        else:
            print("FAILED")
    
    if len(temp_bands) < len(BANDS):
        print(f"    Only {len(temp_bands)}/{len(BANDS_WITH_SCL)} bands downloaded")
        return None
    
    # Stack
    output_file = OUTPUT_DIR / f"S2_Wheat_{target_date.replace('-', '')}_{phase_name}.tif"
    print(f"    Stacking bands...")
    stack_bands(temp_bands, output_file)
    
    # Cleanup
    for tf in temp_bands:
        if tf.exists():
            tf.unlink()
    
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"    Saved: {output_file.name} ({size_mb:.1f} MB)")
    
    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main download function."""
    start_time = time.time()
    
    print("=" * 70)
    print("WHEAT PoC - PHASE 3: DOWNLOAD SENTINEL-2 DATA")
    print("=" * 70)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load PoC tile
    poc_tile = load_poc_tile()
    print(f"PoC Tile: {poc_tile['tile_id']}")
    print(f"Coordinates: {poc_tile['coordinate_count']}")
    print(f"Bounds: {poc_tile['bounds']}")
    
    # Download each date
    downloaded = []
    for target_date, phase_name in TARGET_DATES:
        result = download_for_date(poc_tile, target_date, phase_name)
        if result:
            downloaded.append(result)
    
    # Summary
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"Successfully downloaded: {len(downloaded)}/{len(TARGET_DATES)} images")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    for f in downloaded:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
