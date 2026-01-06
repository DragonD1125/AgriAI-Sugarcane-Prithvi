"""
Create Rice Dataset from ALL Years (2018, 2020, 2022)
======================================================
Downloads Sentinel-2 imagery for rice coordinates from all years.
Uses year-matched imagery for temporal accuracy.
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.crs import CRS
import pystac_client
import planetary_computer as pc
from pathlib import Path
import time
import sys
import random
from typing import List, Dict, Optional, Tuple

# Configuration
BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi")
COORDS_FILE = BASE_DIR / "data" / "rice_coordinates_all_years.csv"
OUTPUT_DIR = BASE_DIR / "data" / "rice_patches_v2"
RICE_TIFS = {
    (2018, "Autumn"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Autumn_rice-WGS84.tif",
    (2018, "Summer"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Summer_rice-WGS84.tif",
    (2018, "Winter"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Winter_rice-WGS84.tif",
    (2020, "Autumn"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2020-Autumn_rice-WGS84.tif",
    (2020, "Summer"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2020-Summer_rice-WGS84.tif",
    (2020, "Winter"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2020-Winter_rice-WGS84.tif",
    (2022, "Autumn"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2022-Autumn_rice-WGS84.tif",
    (2022, "Summer"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2022-Summer_rice-WGS84.tif",
    (2022, "Winter"): BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2022-Winter_rice-WGS84.tif",
}
PATCH_SIZE = 224
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

def get_dates_for_year_season(year, season):
    """Get 3 phenological dates for a given year and season."""
    if season == "Autumn":  # Kharif
        return [f"{year}-06-15", f"{year}-09-01", f"{year}-11-15"]
    elif season == "Winter":  # Rabi Early
        return [f"{year}-01-15", f"{year}-03-15", f"{year}-05-15"]
    elif season == "Summer":  # Rabi Late/Zaid
        return [f"{year}-04-15", f"{year}-06-15", f"{year}-08-15"]
    return [f"{year}-06-15", f"{year}-09-01", f"{year}-11-15"]

def get_bbox_for_patch(lat, lon, size=224, res_deg=0.0001):
    half = (size * res_deg) / 2
    return [lon - half, lat - half, lon + half, lat + half]

def download_patch_band(signed_item, band, bbox, output_shape=(224, 224)):
    asset = signed_item.assets.get(band)
    if not asset: return None
    
    try:
        with rasterio.open(asset.href) as src:
            bbox_proj = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
            window = from_bounds(*bbox_proj, src.transform)
            data = src.read(1, window=window, out_shape=output_shape, resampling=Resampling.bilinear)
            return data
    except Exception as e:
        return None

def create_rice_dataset():
    if not COORDS_FILE.exists():
        print("Coordinates file not found! Run extract_rice_coordinates_all.py first.")
        return

    df = pd.read_csv(COORDS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample 500 patches total (balanced across years)
    TARGET_PATCHES = 500
    if len(df) > TARGET_PATCHES:
        df = df.sample(TARGET_PATCHES, random_state=42)
    
    print(f"Processing {len(df)} locations...")
    print(f"Year distribution: {df['year'].value_counts().to_dict()}")
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", 
        modifier=pc.sign_inplace
    )
    
    success_count = 0
    WINDOW_DAYS = 30
    
    for idx, row in df.iterrows():
        lat, lon, year, season = row['lat'], row['lon'], int(row['year']), row['season']
        print(f"[{idx+1}/{len(df)}] {year} {season} lat={lat:.4f}, lon={lon:.4f}")
        
        temporal_stack = []
        valid_dates = []
        target_dates = get_dates_for_year_season(year, season)
        bbox = get_bbox_for_patch(lat, lon)
        
        bad_location = False
        for date_str in target_dates:
            d_obj = pd.to_datetime(date_str)
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
                print(f"  No scene for {date_str}")
                bad_location = True
                break
            
            best_item = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
            
            bands_data = []
            for b in BANDS:
                d = download_patch_band(best_item, b, bbox)
                if d is None:
                    bad_location = True
                    break
                bands_data.append(d)
                
            if bad_location: break
            
            img = np.stack(bands_data)
            temporal_stack.append(img)
            valid_dates.append(best_item.datetime.strftime("%Y-%m-%d"))
            
        if bad_location or len(temporal_stack) != 3:
            print("  Skipping (incomplete)")
            continue
        
        X = np.stack(temporal_stack, axis=1)
        
        # Extract ground truth
        tif_key = (year, season)
        tif_path = RICE_TIFS.get(tif_key)
        if not tif_path or not tif_path.exists():
            print(f"  Missing TIF for {year} {season}")
            continue
            
        try:
            with rasterio.open(tif_path) as src:
                row_off, col_off = src.index(lon, lat)
                window = Window(col_off - 112, row_off - 112, 224, 224)
                mask = src.read(1, window=window)
                if mask.shape != (224, 224):
                    mask_padded = np.zeros((224, 224), dtype=mask.dtype)
                    h, w = mask.shape
                    mask_padded[:h, :w] = mask
                    mask = mask_padded
                y = (mask > 0).astype(np.uint8) * 2  # Label 2 = Rice
        except Exception as e:
            print(f"  Mask error: {e}")
            continue
        
        saveloc = OUTPUT_DIR / f"rice_{year}_{season}_{idx}.npz"
        np.savez_compressed(saveloc, X=X, y=y, lat=lat, lon=lon, year=year, dates=valid_dates)
        success_count += 1
        print(f"  ✅ Saved")
        
    print(f"\n{'='*60}")
    print(f"Done. Created {success_count} rice patches.")
    print(f"{'='*60}")

if __name__ == "__main__":
    create_rice_dataset()
