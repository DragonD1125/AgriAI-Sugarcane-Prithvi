#!/usr/bin/env python3
"""
Test Sentinel-2 download from Planetary Computer
"""
import json
import traceback
from pathlib import Path

print("Importing libraries...")

import rasterio
from rasterio.windows import from_bounds
from pystac_client import Client
import planetary_computer as pc

print("Libraries imported successfully")

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
OUTPUT_DIR = BASE_DIR / "data" / "sentinel2_poc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load PoC tile
with open(BASE_DIR / 'data/mgrs_tiles_needed.json') as f:
    data = json.load(f)
poc = data['poc_tile']
bounds = poc['bounds']
bbox = [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']]
print(f"Tile: {poc['tile_id']}")
print(f"Bbox: {bbox}")

# Search STAC
print("\nSearching for Sentinel-2 imagery...")
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2022-03-01/2022-03-31',
    query={'eo:cloud_cover': {'lt': 30}}
)
items = list(search.items())
print(f"Found {len(items)} items")

if not items:
    print("No items found!")
    exit(1)

# Get best item (lowest cloud cover)
items_sorted = sorted(items, key=lambda x: x.properties.get('eo:cloud_cover', 100))
best_item = items_sorted[0]
print(f"\nBest item: {best_item.id}")
print(f"Date: {best_item.datetime}")
print(f"Cloud cover: {best_item.properties.get('eo:cloud_cover')}%")
print(f"Available assets: {list(best_item.assets.keys())}")

# Sign the item
print("\nSigning item for access...")
signed_item = pc.sign(best_item)

# Try to download B02 (Blue band)
band = 'B02'
print(f"\nDownloading band {band}...")

asset = signed_item.assets.get(band)
if not asset:
    print(f"Asset {band} not found, available: {list(signed_item.assets.keys())}")
    exit(1)

url = asset.href
print(f"URL: {url[:100]}...")

try:
    print("Opening remote file...")
    with rasterio.open(url) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Transform: {src.transform}")
        
        # Calculate window from bbox
        print(f"\nCalculating window for bbox...")
        try:
            window = from_bounds(*bbox, src.transform)
            print(f"  Window: {window}")
            
            # Read windowed data
            print("Reading windowed data...")
            data = src.read(1, window=window)
            print(f"  Data shape: {data.shape}")
            print(f"  Data dtype: {data.dtype}")
            print(f"  Data range: {data.min()} to {data.max()}")
            
            # Save to file
            output_path = OUTPUT_DIR / f"test_B02.tif"
            print(f"\nSaving to {output_path}...")
            
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                transform=transform,
                compress='lzw'
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            print(f"SUCCESS! Saved to {output_path}")
            print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"Window/clip error: {e}")
            traceback.print_exc()
            
except Exception as e:
    print(f"Error opening/reading file: {e}")
    traceback.print_exc()
