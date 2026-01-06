
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import random

# Configuration
DATA_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data\rice_2018")
OUTPUT_FILE = Path(r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data\rice_coordinates.csv")
SAMPLES_PER_FILE = 2000  # Total 6000 samples (enough for training)

def get_coordinates(tif_path, num_samples, season):
    print(f"Processing {tif_path.name}...")
    coords = []
    
    with rasterio.open(tif_path) as src:
        # Read in blocks to handle large files
        for ji, window in src.block_windows(1):
            # Read data
            data = src.read(1, window=window)
            
            # Find rice pixels (assuming value 1 is rice based on name "classified")
            # inspect_rice_data.py showed values [0], but we need to find 1s
            # We'll assume non-zero is rice given it's a mask
            rows, cols = np.where(data > 0)
            
            if len(rows) > 0:
                # Sample from this block
                # Probability proportional to block size/total valid? 
                # Simplification: Collect a reservoir from blocks
                
                # Transform to map coordinates
                # window.col_off + cols, window.row_off + rows
                xs, ys = rasterio.transform.xy(src.transform, 
                                             window.row_off + rows, 
                                             window.col_off + cols)
                
                block_coords = list(zip(ys, xs)) # lat, lon
                
                # Reservoir sampling or just append
                # Since we want to spread across the file, we'll append a few from each block
                # But simple random sample from all might be better if memory allows
                # Let's just collect all valid indices? No, too big (4GB file).
                
                # Strategy: Take random 1% of valid pixels from this block
                k = max(1, len(block_coords) // 100)
                sampled = random.sample(block_coords, k)
                coords.extend(sampled)
    
    print(f"  Found {len(coords)} potential rice pixels")
    
    # Final sample
    if len(coords) > num_samples:
        final_coords = random.sample(coords, num_samples)
    else:
        final_coords = coords
        
    df = pd.DataFrame(final_coords, columns=['lat', 'lon'])
    df['season'] = season
    return df

def main():
    dfs = []
    
    files = [
        (DATA_DIR / "classified-10m-India-2018-Autumn_rice-WGS84.tif", "Autumn"),
        (DATA_DIR / "classified-10m-India-2018-Summer_rice-WGS84.tif", "Summer"),
        (DATA_DIR / "classified-10m-India-2018-Winter_rice-WGS84.tif", "Winter")
    ]
    
    for fp, season in files:
        if fp.exists():
            df = get_coordinates(fp, SAMPLES_PER_FILE, season)
            dfs.append(df)
        else:
            print(f"Warning: {fp} not found")
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(full_df)} coordinates to {OUTPUT_FILE}")
        print(full_df.groupby('season').size())
    else:
        print("No data found!")

if __name__ == "__main__":
    main()
