import json
from pathlib import Path
from pystac_client import Client

# Load PoC tile
with open('data/mgrs_tiles_needed.json') as f:
    data = json.load(f)
poc = data['poc_tile']
bounds = poc['bounds']
bbox = [bounds['lon_min'], bounds['lat_min'], bounds['lon_max'], bounds['lat_max']]
print(f"Tile: {poc['tile_id']}")
print(f"Bbox: {bbox}")

# Search STAC
catalog = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
search = catalog.search(
    collections=['sentinel-2-l2a'],
    bbox=bbox,
    datetime='2022-03-01/2022-03-31',
    query={'eo:cloud_cover': {'lt': 30}}
)
items = list(search.items())
print(f"Found {len(items)} items")
if items:
    print(f"First item: {items[0].id}")
    print(f"Assets: {list(items[0].assets.keys())[:10]}")
