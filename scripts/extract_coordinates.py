#!/usr/bin/env python3
"""
Extract Sugarcane Field Coordinates from Di Tommaso GeoTIFF Files
MEMORY-OPTIMIZED VERSION - Processes tiles individually with chunked reading

This script extracts high-confidence sugarcane field coordinates from multiple
GeoTIFF tiles provided by Di Tommaso et al. (2024) research dataset.

Key optimizations:
- Processes each tile individually and saves to separate CSV
- Uses windowed/chunked reading to avoid loading full arrays into memory
- Samples during extraction to limit coordinate count per tile
- Combines individual CSVs at the end for final output

Dataset: "Mapping sugarcane globally at 10 m resolution using GEDI and Sentinel-2"
Source: https://zenodo.org/records/10871164
Paper: https://doi.org/10.5194/essd-16-4931-2024

Author: Generated for Agri AI Project
Date: 2024
"""

from pathlib import Path
import sys
import time
import gc
from typing import List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# India geographic bounds for validation
INDIA_LAT_MIN = 8.0   # Southern tip (Kanyakumari)
INDIA_LAT_MAX = 35.0  # Northern India (Kashmir)
INDIA_LON_MIN = 68.0  # Western India (Gujarat)
INDIA_LON_MAX = 97.0  # Eastern India (Arunachal Pradesh)

# Band definitions
BAND_TALL_MONTHS = 1      # Number of months crop was tall (0-12)
BAND_SUGARCANE = 2        # Binary sugarcane classification (0/1)
BAND_ESA_CROP_MASK = 3    # ESA WorldCover cropland mask (0/1)

# Target output size
TARGET_MIN_COORDS = 150
TARGET_MAX_COORDS = 200

# Chunk size for windowed reading (rows at a time)
# Smaller = less memory, larger = faster
CHUNK_SIZE = 1000  # Process 1000 rows at a time

# Maximum coordinates to keep per tile (to prevent memory issues)
MAX_COORDS_PER_TILE = 500000  # 500K coords max per tile


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def print_separator(char: str = "=", length: int = 70) -> None:
    """Print a separator line."""
    print(char * length)


def validate_india_bounds(lat: float, lon: float) -> bool:
    """Check if coordinates fall within India bounds."""
    return (INDIA_LAT_MIN <= lat <= INDIA_LAT_MAX and 
            INDIA_LON_MIN <= lon <= INDIA_LON_MAX)


# ============================================================================
# MAIN EXTRACTION FUNCTIONS
# ============================================================================

def find_tif_files(directory: str) -> List[Path]:
    """
    Find all TIF files in the specified directory matching the Di Tommaso pattern.
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    pattern = "india_GEDIS2_v1*.tif"
    tif_files = sorted(list(dir_path.glob(pattern)))
    tif_files_upper = sorted(list(dir_path.glob("india_GEDIS2_v1*.TIF")))
    tif_files.extend(tif_files_upper)
    tif_files = sorted(list(set(tif_files)))
    
    if len(tif_files) == 0:
        raise ValueError(f"No TIF files found matching pattern '{pattern}' in {directory}")
    
    return tif_files


def extract_coordinates_from_tile_chunked(
    tif_path: Path,
    output_csv: Path,
    min_tall_months: int = 8,
    sample_rate: int = 100,
    chunk_size: int = CHUNK_SIZE,
    max_coords: int = MAX_COORDS_PER_TILE
) -> dict:
    """
    Extract sugarcane coordinates from a single tile using CHUNKED reading.
    
    This function reads the raster in horizontal strips (chunks) to avoid
    loading the entire file into memory at once.
    
    Args:
        tif_path: Path to the GeoTIFF file
        output_csv: Path where individual tile CSV will be saved
        min_tall_months: Minimum tall months threshold
        sample_rate: Sample every Nth pixel during extraction
        chunk_size: Number of rows to process at a time
        max_coords: Maximum coordinates to extract from this tile
    
    Returns:
        Dictionary with processing statistics
    """
    filename = tif_path.name
    stats = {
        'filename': filename,
        'status': 'pending',
        'total_sugarcane': 0,
        'extracted_coords': 0,
        'processing_time': 0,
        'error': None
    }
    
    try:
        with rasterio.open(tif_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            n_bands = src.count
            bounds = src.bounds
            
            print(f"  ✓ Opened successfully")
            print(f"  Raster shape: ({height:,}, {width:,}) - {n_bands} bands")
            print(f"  Bounds: lat [{bounds.bottom:.4f}, {bounds.top:.4f}], "
                  f"lon [{bounds.left:.4f}, {bounds.right:.4f}]")
            
            if n_bands < 3:
                print(f"  ⚠ Warning: Only {n_bands} bands, need at least 3. Skipping.")
                stats['status'] = 'skipped'
                stats['error'] = f"Only {n_bands} bands"
                return stats
            
            # Calculate number of chunks
            n_chunks = (height + chunk_size - 1) // chunk_size
            print(f"  Processing in {n_chunks} chunks of {chunk_size} rows each...")
            
            # Lists to accumulate coordinates (much smaller than full arrays)
            all_lats = []
            all_lons = []
            all_tall = []
            
            total_sugarcane = 0
            total_qualifying = 0
            pixel_counter = 0  # For sampling
            
            # Process chunk by chunk
            for chunk_idx in range(n_chunks):
                row_start = chunk_idx * chunk_size
                row_end = min((chunk_idx + 1) * chunk_size, height)
                rows_in_chunk = row_end - row_start
                
                # Create window for this chunk
                window = Window(0, row_start, width, rows_in_chunk)
                
                # Read only the bands we need for this chunk
                # Shape will be (rows_in_chunk, width) - much smaller!
                band1_chunk = src.read(BAND_TALL_MONTHS, window=window)
                band2_chunk = src.read(BAND_SUGARCANE, window=window)
                band3_chunk = src.read(BAND_ESA_CROP_MASK, window=window)
                
                # Count total sugarcane in chunk
                chunk_sugarcane = np.sum(band2_chunk == 1)
                total_sugarcane += chunk_sugarcane
                
                # Find qualifying pixels in this chunk
                # Using in-place operations to minimize memory
                mask = (band2_chunk == 1)
                mask &= (band1_chunk >= min_tall_months)
                mask &= (band3_chunk == 1)
                
                chunk_qualifying = np.sum(mask)
                total_qualifying += chunk_qualifying
                
                if chunk_qualifying == 0:
                    # Clean up chunk memory
                    del band1_chunk, band2_chunk, band3_chunk, mask
                    continue
                
                # Get local row, col indices within this chunk
                local_rows, local_cols = np.where(mask)
                
                # Convert to global row indices
                global_rows = local_rows + row_start
                
                # Apply sampling during extraction
                for i in range(len(local_rows)):
                    pixel_counter += 1
                    if pixel_counter % sample_rate == 0:
                        # Get coordinates for this pixel
                        row = global_rows[i]
                        col = local_cols[i]
                        
                        # Convert to lat/lon using transform
                        lon = transform.c + col * transform.a + row * transform.b
                        lat = transform.f + col * transform.d + row * transform.e
                        
                        # Get tall_months value
                        tall = band1_chunk[local_rows[i], local_cols[i]]
                        
                        all_lats.append(lat)
                        all_lons.append(lon)
                        all_tall.append(int(tall))
                        
                        # Check if we've hit the max coords limit
                        if len(all_lats) >= max_coords:
                            print(f"  ⚠ Reached max coords limit ({max_coords:,})")
                            break
                
                # Clean up chunk memory
                del band1_chunk, band2_chunk, band3_chunk, mask
                del local_rows, local_cols, global_rows
                gc.collect()
                
                # Progress indicator every 10 chunks
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                    print(f"    Chunk {chunk_idx + 1}/{n_chunks}: "
                          f"{len(all_lats):,} coords so far...")
                
                # Break if we've hit max coords
                if len(all_lats) >= max_coords:
                    break
            
            # Update statistics
            stats['total_sugarcane'] = total_sugarcane
            stats['total_qualifying'] = total_qualifying
            
            print(f"  Total sugarcane pixels: {total_sugarcane:,}")
            print(f"  Qualifying pixels (all filters): {total_qualifying:,}")
            print(f"  Extracted coordinates: {len(all_lats):,}")
            
            if len(all_lats) == 0:
                print(f"  ⚠ No coordinates extracted from this tile")
                stats['status'] = 'empty'
                stats['extracted_coords'] = 0
                return stats
            
            # Create DataFrame for this tile
            df = pd.DataFrame({
                'latitude': np.round(all_lats, 6),
                'longitude': np.round(all_lons, 6),
                'tall_months': all_tall,
                'confidence': np.round(np.array(all_tall) / 12.0, 2),
                'source_file': filename,
                'source': 'Di_Tommaso_2024'
            })
            
            # Save to CSV
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"  ✓ Saved to: {output_csv.name}")
            
            stats['status'] = 'success'
            stats['extracted_coords'] = len(df)
            
            # Clean up
            del all_lats, all_lons, all_tall, df
            gc.collect()
            
            return stats
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        stats['status'] = 'error'
        stats['error'] = str(e)
        return stats


def combine_tile_csvs(
    csv_dir: Path,
    output_csv: Path,
    target_coords: int = 175
) -> pd.DataFrame:
    """
    Combine individual tile CSVs into final output.
    
    Uses chunked reading to handle large combined datasets without memory issues.
    """
    print_separator()
    print("Combining tile CSVs...")
    print_separator()
    print()
    
    # Find all tile CSVs
    tile_csvs = sorted(csv_dir.glob("tile_*.csv"))
    
    if len(tile_csvs) == 0:
        raise ValueError(f"No tile CSVs found in {csv_dir}")
    
    print(f"Found {len(tile_csvs)} tile CSV files")
    
    # Read and combine all CSVs
    dfs = []
    total_coords = 0
    
    for csv_path in tile_csvs:
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            dfs.append(df)
            total_coords += len(df)
            print(f"  {csv_path.name}: {len(df):,} coordinates")
    
    if len(dfs) == 0:
        raise ValueError("No coordinates found in any tile CSV!")
    
    print(f"\nTotal coordinates: {total_coords:,}")
    
    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    # Remove duplicates efficiently
    print("Removing duplicates...")
    df_combined['lat_round'] = df_combined['latitude'].round(5)
    df_combined['lon_round'] = df_combined['longitude'].round(5)
    
    # Sort by tall_months descending, keep first (highest)
    df_combined = df_combined.sort_values('tall_months', ascending=False)
    df_combined = df_combined.drop_duplicates(subset=['lat_round', 'lon_round'], keep='first')
    df_combined = df_combined.drop(columns=['lat_round', 'lon_round'])
    
    print(f"After deduplication: {len(df_combined):,} coordinates")
    
    # Sample to target size
    if len(df_combined) > TARGET_MAX_COORDS:
        sample_rate = len(df_combined) // target_coords
        df_combined = df_combined.iloc[::sample_rate].head(TARGET_MAX_COORDS)
        print(f"Sampled to: {len(df_combined):,} coordinates")
    
    df_combined = df_combined.reset_index(drop=True)
    
    # Reorder columns
    df_combined = df_combined[['latitude', 'longitude', 'tall_months', 'confidence', 'source_file', 'source']]
    
    # Save final output
    df_combined.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n✓ Final output saved to: {output_csv}")
    
    return df_combined


def extract_sugarcane_coordinates(
    input_dir: str,
    output_csv: str,
    sample_rate: int = 100,
    min_tall_months: int = 8,
    target_coords: int = 175
) -> pd.DataFrame:
    """
    Main extraction function - processes tiles individually.
    """
    start_time = time.time()
    
    # Create temp directory for individual tile CSVs
    output_path = Path(output_csv)
    temp_dir = output_path.parent / "temp_tiles"
    temp_dir.mkdir(exist_ok=True)
    
    print(f"Temporary tile CSVs will be saved to: {temp_dir}")
    print()
    
    # Find all TIF files
    print("Scanning for TIF files...")
    tif_files = find_tif_files(input_dir)
    n_files = len(tif_files)
    
    print(f"\nFound {n_files} TIF files:")
    total_size = 0
    for i, tif_path in enumerate(tif_files, 1):
        size = tif_path.stat().st_size
        total_size += size
        print(f"  {i:2d}. {tif_path.name} ({format_file_size(size)})")
    
    print(f"\nTotal data size: {format_file_size(total_size)}")
    print()
    
    # Process each tile individually
    print_separator()
    print("Processing tiles (one at a time, memory-efficient)...")
    print_separator()
    print()
    
    all_stats = []
    successful_tiles = 0
    
    for i, tif_path in enumerate(tif_files, 1):
        tile_start = time.time()
        print(f"[{i}/{n_files}] Processing: {tif_path.name}")
        
        # Output CSV for this tile
        tile_csv = temp_dir / f"tile_{i:02d}_{tif_path.stem}.csv"
        
        # Process tile with chunked reading
        stats = extract_coordinates_from_tile_chunked(
            tif_path=tif_path,
            output_csv=tile_csv,
            min_tall_months=min_tall_months,
            sample_rate=sample_rate,
            chunk_size=CHUNK_SIZE,
            max_coords=MAX_COORDS_PER_TILE
        )
        
        tile_time = time.time() - tile_start
        stats['processing_time'] = tile_time
        all_stats.append(stats)
        
        if stats['status'] == 'success':
            successful_tiles += 1
            print(f"  ✓ Tile completed in {tile_time:.1f} seconds")
        else:
            print(f"  ⚠ Tile status: {stats['status']}")
        
        # Force garbage collection between tiles
        gc.collect()
        print()
    
    # Summary of tile processing
    print_separator()
    print("TILE PROCESSING SUMMARY")
    print_separator()
    print()
    
    total_extracted = sum(s['extracted_coords'] for s in all_stats)
    print(f"Tiles processed: {n_files}")
    print(f"Tiles successful: {successful_tiles}")
    print(f"Total coordinates extracted: {total_extracted:,}")
    print()
    
    # Per-tile summary
    print("Per-tile results:")
    for stats in all_stats:
        status_icon = "✓" if stats['status'] == 'success' else "✗"
        print(f"  {status_icon} {stats['filename']}: {stats['extracted_coords']:,} coords "
              f"({stats['processing_time']:.1f}s)")
    print()
    
    # Combine all tile CSVs into final output
    df_final = combine_tile_csvs(
        csv_dir=temp_dir,
        output_csv=output_path,
        target_coords=target_coords
    )
    
    # Final statistics
    total_time = time.time() - start_time
    
    print()
    print_separator()
    print("FINAL STATISTICS")
    print_separator()
    print()
    
    print(f"Total coordinates in final output: {len(df_final):,}")
    print()
    
    # Geographic distribution
    lat_min = df_final['latitude'].min()
    lat_max = df_final['latitude'].max()
    lon_min = df_final['longitude'].min()
    lon_max = df_final['longitude'].max()
    
    print("Geographic distribution:")
    print(f"  Latitude range: {lat_min:.4f}°N to {lat_max:.4f}°N")
    print(f"  Longitude range: {lon_min:.4f}°E to {lon_max:.4f}°E")
    
    # Validate India bounds
    outside_bounds = df_final[
        ~df_final.apply(lambda r: validate_india_bounds(r['latitude'], r['longitude']), axis=1)
    ]
    
    if len(outside_bounds) > 0:
        print(f"  ⚠ Warning: {len(outside_bounds)} coordinates outside India bounds")
    else:
        print("  ✓ All coordinates within India bounds")
    print()
    
    # Tall months statistics
    avg_tall = df_final['tall_months'].mean()
    print(f"Average tall_months: {avg_tall:.2f} (expected 8-12)")
    print(f"Average confidence: {df_final['confidence'].mean():.2f}")
    print()
    
    # Processing time
    print(f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    
    # Preview
    print("First 10 rows:")
    print(df_final.head(10).to_string(index=True))
    print()
    
    return df_final


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Define paths
    BASE_DIR = Path(r"C:\Users\rdaksh\Desktop\Agri AI")
    INPUT_DIR = BASE_DIR / "data" / "india_GEDIS2_v1"
    OUTPUT_CSV = BASE_DIR / "data" / "sugarcane_field_locations.csv"
    
    print_separator()
    print("SUGARCANE COORDINATE EXTRACTION")
    print("Memory-Optimized Version (Chunked Processing)")
    print_separator()
    print()
    print(f"Base directory: {BASE_DIR}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()
    
    # Memory info
    print("Processing settings:")
    print(f"  Chunk size: {CHUNK_SIZE} rows at a time")
    print(f"  Max coords per tile: {MAX_COORDS_PER_TILE:,}")
    print(f"  Target final coords: {TARGET_MIN_COORDS}-{TARGET_MAX_COORDS}")
    print()
    
    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    try:
        df = extract_sugarcane_coordinates(
            input_dir=str(INPUT_DIR),
            output_csv=str(OUTPUT_CSV),
            sample_rate=100,
            min_tall_months=8,
            target_coords=175
        )
        
        print_separator()
        print("✓ EXTRACTION COMPLETE")
        print_separator()
        print(f"Total coordinates: {len(df):,}")
        print(f"Output: {OUTPUT_CSV}")
        print()
        print("Ready for Phase 3: Sentinel-2 download")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
