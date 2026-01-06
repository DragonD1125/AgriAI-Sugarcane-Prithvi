#!/usr/bin/env python3
"""Download just the August 2020 peak_growth scene - extended search"""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.crs import CRS
from pystac_client import Client
import planetary_computer as pc

BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
OUTPUT_DIR = BASE_DIR / "data" / "sentinel2_poc"

BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
BANDS_WITH_SCL = BANDS + ['SCL']

# Load PoC tile
with open(BASE_DIR / 'data/mgrs_tiles_needed.json') as f:
    data = json.load(f)
poc = data['poc_tile']
bounds = poc['bounds']
bbox = [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']]

# Search July-September 2020 (monsoon season, may have more clouds)
print("Searching July-September 2020...")
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2020-07-01/2020-09-30',
    query={'eo:cloud_cover': {'lt': 50}}  # Higher threshold for monsoon
)
items = list(search.items())
print(f"Found {len(items)} items")

# Find covering tile
center_lon = (bbox[0] + bbox[2]) / 2
center_lat = (bbox[1] + bbox[3]) / 2

covering_items = []
for item in items:
    ib = item.bbox
    if ib[0] <= center_lon <= ib[2] and ib[1] <= center_lat <= ib[3]:
        covering_items.append(item)

print(f"Items covering our area: {len(covering_items)}")

if covering_items:
    # Sort by cloud cover
    covering_items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100))
    
    for item in covering_items[:5]:
        print(f"  {item.id}: {item.datetime}, cloud={item.properties.get('eo:cloud_cover'):.1f}%")
    
    best = covering_items[0]
    print(f"\nUsing: {best.id}")
    print(f"Date: {best.datetime}")
    print(f"Cloud: {best.properties.get('eo:cloud_cover')}%")
    
    signed = pc.sign(best)
    output_bands = []
    
    for band in BANDS_WITH_SCL:
        print(f"  Downloading {band}...")
        asset = signed.assets.get(band)
        if not asset:
            print(f"    Not found")
            continue
        
        band_path = OUTPUT_DIR / f"temp_{band}_peak_growth.tif"
        
        try:
            with rasterio.open(asset.href) as src:
                bbox_transformed = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
                rb = src.bounds
                clip_bbox = [
                    max(bbox_transformed[0], rb.left),
                    max(bbox_transformed[1], rb.bottom),
                    min(bbox_transformed[2], rb.right),
                    min(bbox_transformed[3], rb.top)
                ]
                window = from_bounds(*clip_bbox, src.transform)
                window = Window(int(window.col_off), int(window.row_off),
                               int(window.width) + 1, int(window.height) + 1)
                window = Window(max(0, window.col_off), max(0, window.row_off),
                               min(window.width, src.width - window.col_off),
                               min(window.height, src.height - window.row_off))
                
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                
                profile = src.profile.copy()
                profile.update(driver='GTiff', height=data.shape[0], width=data.shape[1],
                              count=1, dtype=data.dtype, transform=transform, compress='lzw')
                
                with rasterio.open(band_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                
                output_bands.append(band_path)
                print(f"    OK ({band_path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"    Error: {e}")
    
    if len(output_bands) >= 6:
        # Stack bands
        output_file = OUTPUT_DIR / "S2_PoC_20200815_peak_growth.tif"
        print(f"\nStacking to {output_file}...")
        
        with rasterio.open(output_bands[0]) as ref:
            ref_profile = ref.profile.copy()
            ref_profile.update(count=len(output_bands), compress='lzw', tiled=True)
            ref_h, ref_w = ref.height, ref.width
            ref_transform = ref.transform
            ref_crs = ref.crs
        
        with rasterio.open(output_file, 'w', **ref_profile) as dst:
            for i, bf in enumerate(output_bands, 1):
                with rasterio.open(bf) as src:
                    d = src.read(1)
                    if d.shape != (ref_h, ref_w):
                        resampled = np.zeros((ref_h, ref_w), dtype=d.dtype)
                        reproject(d, resampled, src_transform=src.transform, src_crs=src.crs,
                                 dst_transform=ref_transform, dst_crs=ref_crs,
                                 resampling=Resampling.bilinear)
                        dst.write(resampled, i)
                    else:
                        dst.write(d, i)
            dst.descriptions = tuple(BANDS_WITH_SCL[:len(output_bands)])
        
        # Clean up
        for bf in output_bands:
            bf.unlink()
        
        print(f"Done: {output_file} ({output_file.stat().st_size // (1024*1024)} MB)")
    else:
        print(f"Only {len(output_bands)} bands downloaded, cleanup...")
        for bf in output_bands:
            if bf.exists():
                bf.unlink()
else:
    print("No covering items found!")
    print("\nAll items found:")
    for item in items[:10]:
        print(f"  {item.id}: bbox={item.bbox}")
