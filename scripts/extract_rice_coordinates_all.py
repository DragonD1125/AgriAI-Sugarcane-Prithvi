"""
Extract Rice Coordinates from ALL Years (2018, 2020, 2022)
===========================================================
Extracts coordinates from all 9 rice GeoTIFF files to create a larger dataset.
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Configuration
DATA_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data\rice_2018")
OUTPUT_FILE = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data\rice_coordinates_all_years.csv")
SAMPLES_PER_FILE = 1000  # 1000 per file × 9 files = 9000 total

def get_coordinates(tif_path, num_samples, year, season):
    print(f"Processing {tif_path.name}...")
    coords = []
    
    with rasterio.open(tif_path) as src:
        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            rows, cols = np.where(data > 0)
            
            if len(rows) > 0:
                xs, ys = rasterio.transform.xy(src.transform, 
                                             window.row_off + rows, 
                                             window.col_off + cols)
                block_coords = list(zip(ys, xs))
                k = max(1, len(block_coords) // 100)
                sampled = random.sample(block_coords, min(k, len(block_coords)))
                coords.extend(sampled)
    
    print(f"  Found {len(coords)} potential rice pixels")
    
    if len(coords) > num_samples:
        final_coords = random.sample(coords, num_samples)
    else:
        final_coords = coords
        
    df = pd.DataFrame(final_coords, columns=['lat', 'lon'])
    df['year'] = year
    df['season'] = season
    return df

def main():
    dfs = []
    
    # Only 2020 and 2022 (2018 already done - 165 patches exist)
    files = [
        # 2020
        (DATA_DIR / "classified-10m-India-2020-Autumn_rice-WGS84.tif", 2020, "Autumn"),
        (DATA_DIR / "classified-10m-India-2020-Summer_rice-WGS84.tif", 2020, "Summer"),
        (DATA_DIR / "classified-10m-India-2020-Winter_rice-WGS84.tif", 2020, "Winter"),
        # 2022
        (DATA_DIR / "classified-10m-India-2022-Autumn_rice-WGS84.tif", 2022, "Autumn"),
        (DATA_DIR / "classified-10m-India-2022-Summer_rice-WGS84.tif", 2022, "Summer"),
        (DATA_DIR / "classified-10m-India-2022-Winter_rice-WGS84.tif", 2022, "Winter"),
    ]
    
    for fp, year, season in files:
        if fp.exists():
            df = get_coordinates(fp, SAMPLES_PER_FILE, year, season)
            dfs.append(df)
        else:
            print(f"Warning: {fp} not found")
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'='*60}")
        print(f"Saved {len(full_df)} coordinates to {OUTPUT_FILE}")
        print(f"{'='*60}")
        print("\nBreakdown by year and season:")
        print(full_df.groupby(['year', 'season']).size())
    else:
        print("No data found!")

if __name__ == "__main__":
    main()
