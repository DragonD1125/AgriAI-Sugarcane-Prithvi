
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import pystac_client
import planetary_computer as pc
from pathlib import Path
import time
import random

# Config
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

def search_scene(bbox, season):
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # Season to date mapping (approximate for 2018)
    if season == "Autumn":
        date_range = "2018-09-01/2018-11-30"  # Harvest/Late
    elif season == "Summer":
        date_range = "2018-05-01/2018-08-31"  # Peak growth
    else: # Winter
        date_range = "2018-01-01/2018-03-31" 

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = list(search.items())
    return items[0] if items else None

def download_patch(item, bbox, output_path):
    # bbox is [minx, miny, maxx, maxy] in WGS84
    
    patch_data = [] # List of (6, 224, 224)
    
    # We want 3 timestamps? 
    # Prithvi expects 3 timestamps.
    # But for this Rice data (2018), we might not have the full temporal series logic perfectly aligned.
    # We'll just take 3 clear scenes from the season or around it.
    # Actually, simpler: Search for 3 distinct dates in 2018 for this location.
    
    return None # Placeholder - logic is complex for 3 timestamps

def create_rice_dataset():
    if not COORDS_FILE.exists():
        print(f"Coordinates file not found: {COORDS_FILE}")
        return

    df = pd.read_csv(COORDS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(df)} coordinates...")
    
    # For PoC, limit to 200 rice patches to match sugarcane size (~180)
    if len(df) > 200:
        df = df.sample(200, random_state=42)
    
    # ... (Actual download logic would go here, simplified for this file creation)
    # Since we can't easily download 3 timestamps aligned with phenology without more complex logic,
    # and the user wants a SCRIPT...
    
    print("NOTE: This script is a template. Actual Sentinel-2 download for 3 timestamps requires API access and time.")
    print("For this PoC, we will simulate Rice data by reusing Sugarcane data but validating the pipeline.")
    
    # Wait, "Do not fallback to anything other than Prithvi".
    # User expects REAL data.
    # I must implement the download logic properly.

    pass # Will rewrite in next step
