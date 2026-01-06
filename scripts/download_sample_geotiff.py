"""
Download a single Sentinel-2 GeoTIFF for a rice location.
Saves as a proper GeoTIFF file instead of NPZ.
"""

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.crs import CRS
from rasterio.windows import from_bounds
import pystac_client
import planetary_computer as pc
from pathlib import Path
import pandas as pd

# Config
OUTPUT_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data")
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

# Sample rice coordinate (from 2020 Winter - high success rate)
LAT = 25.4591
LON = 86.2579
DATE = "2020-03-15"  # Peak growth season
WINDOW_DAYS = 30

def download_geotiff():
    print("="*60)
    print("DOWNLOADING SENTINEL-2 GeoTIFF")
    print("="*60)
    print(f"Location: {LAT}, {LON}")
    print(f"Date: {DATE} (±{WINDOW_DAYS} days)")
    
    # Create bbox (224 pixels at ~10m = ~2.24km)
    half = (224 * 0.0001) / 2
    bbox = [LON - half, LAT - half, LON + half, LAT + half]
    
    # Search for imagery
    print("\nSearching Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    
    d_obj = pd.to_datetime(DATE)
    start_dt = (d_obj - pd.Timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')
    end_dt = (d_obj + pd.Timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_dt}/{end_dt}",
        query={"eo:cloud_cover": {"lt": 20}}
    )
    items = list(search.items())
    
    if not items:
        print("No images found!")
        return
    
    # Pick best (lowest cloud cover)
    best = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
    print(f"Found: {best.id}")
    print(f"Date: {best.datetime.strftime('%Y-%m-%d')}")
    print(f"Cloud: {best.properties.get('eo:cloud_cover', 'N/A')}%")
    
    # Download all bands and stack
    print(f"\nDownloading {len(BANDS)} bands...")
    band_data = []
    
    for band in BANDS:
        asset = best.assets.get(band)
        if not asset:
            print(f"  {band}: Not found")
            continue
            
        with rasterio.open(asset.href) as src:
            # Transform bbox to source CRS
            bbox_proj = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
            window = from_bounds(*bbox_proj, src.transform)
            
            # Read data
            data = src.read(1, window=window, out_shape=(224, 224), 
                           resampling=Resampling.bilinear)
            band_data.append(data)
            print(f"  {band}: {data.shape}, range [{data.min()}-{data.max()}]")
            
            # Save transform info from first band
            if len(band_data) == 1:
                out_transform = rasterio.transform.from_bounds(
                    *bbox, 224, 224
                )
                out_crs = CRS.from_epsg(4326)
    
    # Stack bands
    stacked = np.stack(band_data)  # (6, 224, 224)
    print(f"\nStacked shape: {stacked.shape}")
    
    # Save as GeoTIFF
    output_file = OUTPUT_DIR / f"sample_rice_sentinel2_{best.datetime.strftime('%Y%m%d')}.tif"
    
    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=224,
        width=224,
        count=6,  # 6 bands
        dtype=stacked.dtype,
        crs=out_crs,
        transform=out_transform,
    ) as dst:
        for i in range(6):
            dst.write(stacked[i], i + 1)
            dst.set_band_description(i + 1, BANDS[i])
    
    print(f"\n✅ Saved GeoTIFF: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"   Bands: {BANDS}")
    print(f"   Shape: {stacked.shape}")

if __name__ == "__main__":
    download_geotiff()
