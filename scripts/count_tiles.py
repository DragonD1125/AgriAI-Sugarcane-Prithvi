import pandas as pd
from pathlib import Path

tiles_dir = Path(r"C:\Users\rdaksh\Desktop\Agri AI\data\temp_tiles")
tiles = sorted(tiles_dir.glob("*.csv"))

print("Individual tile CSVs:")
print("=" * 60)

total = 0
for p in tiles:
    count = len(pd.read_csv(p))
    total += count
    print(f"  {p.name}: {count:,}")

print("=" * 60)
print(f"TOTAL COORDINATES AVAILABLE: {total:,}")
