#!/usr/bin/env python3
"""Debug script to find correct Sentinel-2 tiles for our bbox"""
import json
from pathlib import Path
from pystac_client import Client
import planetary_computer as pc
import rasterio

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")

# Load PoC tile
with open(BASE_DIR / 'data/mgrs_tiles_needed.json') as f:
    data = json.load(f)
poc = data['poc_tile']
bounds = poc['bounds']
bbox = [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']]

print(f"Target bbox: {bbox}")
print(f"  Lon: {bbox[0]:.4f} to {bbox[2]:.4f}")
print(f"  Lat: {bbox[1]:.4f} to {bbox[3]:.4f}")

# Search STAC with full month
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2022-03-01/2022-03-31',
    query={'eo:cloud_cover': {'lt': 30}}
)
items = list(search.items())
print(f"\nFound {len(items)} items")

# Check which tiles contain our bbox
print("\nChecking which tiles contain our bbox:")
for i, item in enumerate(items[:10]):
    # Get item bbox from geometry
    item_bbox = item.bbox
    
    # Check intersection
    intersects = not (
        bbox[0] > item_bbox[2] or  # our left > their right
        bbox[2] < item_bbox[0] or  # our right < their left
        bbox[1] > item_bbox[3] or  # our bottom > their top
        bbox[3] < item_bbox[1]     # our top < their bottom
    )
    
    # Check containment (does item contain our center point?)
    center_lon = (bbox[0] + bbox[2]) / 2
    center_lat = (bbox[1] + bbox[3]) / 2
    contains_center = (
        item_bbox[0] <= center_lon <= item_bbox[2] and
        item_bbox[1] <= center_lat <= item_bbox[3]
    )
    
    cloud = item.properties.get('eo:cloud_cover', 'N/A')
    print(f"\n{i+1}. {item.id}")
    print(f"   Item bbox: {[round(b, 4) for b in item_bbox]}")
    print(f"   Cloud: {cloud}%")
    print(f"   Intersects: {intersects}")
    print(f"   Contains center ({center_lon:.4f}, {center_lat:.4f}): {contains_center}")
    
    if contains_center:
        print("   *** THIS TILE CONTAINS OUR AREA ***")
        
        # Try to read actual bounds from B02
        signed = pc.sign(item)
        b02_url = signed.assets.get('B02').href
        with rasterio.open(b02_url) as src:
            print(f"   Actual raster bounds: {src.bounds}")
            
            # Check if our bbox intersects with raster bounds
            rb = src.bounds
            raster_intersects = not (
                bbox[0] > rb.right or
                bbox[2] < rb.left or
                bbox[1] > rb.top or
                bbox[3] < rb.bottom
            )
            print(f"   Raster intersects bbox: {raster_intersects}")
        break
