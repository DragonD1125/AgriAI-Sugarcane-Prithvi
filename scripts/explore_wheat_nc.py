#!/usr/bin/env python3
"""
Explore the CROPGRIDS wheat.nc file structure.
"""

import xarray as xr
from pathlib import Path
import numpy as np

NC_FILE = Path(r"C:\Users\rdaksh\Desktop\Agri AI\data\cropgrids\CROPGRIDSv1.08_NC_maps\CROPGRIDSv1.08_NC_maps\CROPGRIDSv1.08_wheat.nc")

print(f"Loading: {NC_FILE}")
print(f"Exists: {NC_FILE.exists()}")
print()

# Open the dataset
ds = xr.open_dataset(NC_FILE)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(ds)
print()

print("=" * 70)
print("VARIABLES")
print("=" * 70)
for var in ds.data_vars:
    print(f"\n{var}:")
    print(f"  Shape: {ds[var].shape}")
    print(f"  Dims: {ds[var].dims}")
    print(f"  Attrs: {dict(ds[var].attrs)}")
    
    # Sample values
    data = ds[var].values
    print(f"  Min: {np.nanmin(data)}, Max: {np.nanmax(data)}")
    print(f"  Non-NaN count: {np.count_nonzero(~np.isnan(data))}")

print()
print("=" * 70)
print("COORDINATES")
print("=" * 70)
for coord in ds.coords:
    print(f"\n{coord}:")
    print(f"  Range: {ds[coord].values.min()} to {ds[coord].values.max()}")
    print(f"  Length: {len(ds[coord])}")

# Focus on India region
print()
print("=" * 70)
print("INDIA REGION CHECK (Lat: 8-35, Lon: 68-97)")
print("=" * 70)

lat_mask = (ds.lat >= 8) & (ds.lat <= 35)
lon_mask = (ds.lon >= 68) & (ds.lon <= 97)

india_ds = ds.sel(lat=ds.lat[lat_mask], lon=ds.lon[lon_mask])
print(f"India subset shape: {india_ds['cropland_area'].shape if 'cropland_area' in india_ds else 'N/A'}")

# Find wheat pixel count
for var in ds.data_vars:
    india_data = india_ds[var].values
    valid_count = np.count_nonzero((~np.isnan(india_data)) & (india_data > 0))
    print(f"  {var}: {valid_count:,} valid pixels in India")

ds.close()
