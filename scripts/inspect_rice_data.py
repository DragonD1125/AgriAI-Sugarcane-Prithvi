
import rasterio
import numpy as np
from pathlib import Path

file_path = r"C:\Users\rdaksh\Desktop\Agri AI\Prithvi\data\rice_2018\classified-10m-India-2018-Autumn_rice-WGS84.tif"

with rasterio.open(file_path) as src:
    print(f"Name: {Path(file_path).name}")
    print(f"Shape: {src.shape}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Count: {src.count}")
    
    # Read a chunk to check values
    data = src.read(1, window=((10000, 11000), (10000, 11000)))
    print(f"Unique values in sample: {np.unique(data)}")
    print(f"Dtype: {data.dtype}")
