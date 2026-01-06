
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
COORDS_FILE = BASE_DIR / "data" / "rice_coordinates.csv"
OUTPUT_DIR = BASE_DIR / "data" / "rice_patches"
RICE_TIFS = {
    "Autumn": BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Autumn_rice-WGS84.tif",
    "Summer": BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Summer_rice-WGS84.tif",
    "Winter": BASE_DIR / "data" / "rice_2018" / "classified-10m-India-2018-Winter_rice-WGS84.tif"
}
PATCH_SIZE = 224
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

# 2018 Rice Crop Cycles (India)
# Autumn (Kharif): June-Nov | Winter/Summer (Rabi): Dec-April
def get_dates_for_season(season):
    if season == "Autumn": # Kharif
        return ["2018-06-15", "2018-09-01", "2018-11-15"]
    elif season == "Winter": # Rabi (Early)
        return ["2018-01-15", "2018-03-15", "2018-05-15"]
    elif season == "Summer": # Rabi (Late/Zaid)
        return ["2018-04-15", "2018-06-15", "2018-08-15"]
    return ["2018-06-15", "2018-09-01", "2018-11-15"] # Default

def get_bbox_for_patch(lat, lon, size=224, res_deg=0.0001):
    # Approx 10m/deg at equator is 0.00009, but simplified
    half = (size * res_deg) / 2
    return [lon - half, lat - half, lon + half, lat + half]

def download_patch_band(signed_item, band, bbox, output_shape=(224, 224)):
    asset = signed_item.assets.get(band)
    if not asset: return None
    
    try:
        with rasterio.open(asset.href) as src:
            # Transform bbox to src crs
            bbox_proj = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
            window = from_bounds(*bbox_proj, src.transform)
            
            # Read and resize to exact patch size
            data = src.read(1, window=window, out_shape=output_shape, resampling=Resampling.bilinear)
            return data
    except Exception as e:
        print(f"Error reading band {band}: {e}")
        return None

def create_rice_dataset():
    if not COORDS_FILE.exists():
        print("Waiting for coordinates file to be generated...")
        return

    df = pd.read_csv(COORDS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process random subset for PoC
    if len(df) > 200:
        df = df.sample(200, random_state=42)
    
    print(f"Processing {len(df)} locations...")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    success_count = 0
    
    for idx, row in df.iterrows():
        lat, lon, season = row['lat'], row['lon'], row['season']
        print(f"[{idx+1}/{len(df)}] Processing {season} lat={lat:.4f}, lon={lon:.4f}")
        
        # 1. Download Sentinel-2 for 3 dates
        temporal_stack = []
        valid_dates = []
        
        # Adjust dates based on season
        target_dates = get_dates_for_season(season)
        WINDOW_DAYS = 30
        
        bbox = get_bbox_for_patch(lat, lon)
        
        # Find item for each date
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
                
            # Pick best item
            best_item = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))[0]
            
            # Download bands
            bands_data = []
            for b in BANDS:
                d = download_patch_band(best_item, b, bbox)
                if d is None:
                    bad_location = True
                    break
                bands_data.append(d)
                
            if bad_location: break
            
            # Stack bands -> (6, 224, 224)
            img = np.stack(bands_data)
            temporal_stack.append(img)
            valid_dates.append(best_item.datetime.strftime("%Y-%m-%d"))
            
        if bad_location or len(temporal_stack) != 3:
            print("  Skipping (incomplete temporal data)")
            continue
            
        # Stack time: (6, 3, 224, 224)
        X = np.stack(temporal_stack, axis=1)
        
        # 2. Extract Ground Truth
        # Read from local huge TIFs
        tif_path = RICE_TIFS.get(season)
        if not tif_path or not tif_path.exists():
            print(f"  Missing TIF for {season}")
            continue
            
        try:
            with rasterio.open(tif_path) as src:
                row_off, col_off = src.index(lon, lat)
                window = Window(col_off - 112, row_off - 112, 224, 224)
                mask = src.read(1, window=window)
                if mask.shape != (224, 224):
                    # Pad
                    mask_padded = np.zeros((224, 224), dtype=mask.dtype)
                    h, w = mask.shape
                    mask_padded[:h, :w] = mask
                    mask = mask_padded
                
                # Convert to Class 2 (Rice)
                y = (mask > 0).astype(np.uint8) * 2
        except Exception as e:
            print(f"  Mask extract error: {e}")
            continue
            
        # Save
        saveloc = OUTPUT_DIR / f"rice_{idx}.npz"
        np.savez_compressed(saveloc, X=X, y=y, lat=lat, lon=lon, dates=valid_dates)
        success_count += 1
        print(f"  Saved to {saveloc}")
        
    print(f"Done. Created {success_count} rice patches.")

if __name__ == "__main__":
    create_rice_dataset()
