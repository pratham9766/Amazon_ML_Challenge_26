#!/usr/bin/env python3
"""
Step 0: Convert all TSV files to Parquet with country partitioning.
Run this ONCE before the main pipeline.

Usage:
    python baseline_pipeline/scripts/00_prepare_parquet.py
"""
import sys, time, gc
from pathlib import Path
import pandas as pd

BASE = Path('d:/RESOURCES/AmazonMLChallenge')
DATA = BASE / 'dataset-20260925T050739Z-1-001' / 'dataset'
CACHE = BASE / 'baseline_pipeline' / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)

def convert_tsv_to_parquet(tsv_path: Path, name: str, chunksize: int = 500000):
    """Convert a TSV to Parquet, partitioned by country."""
    out_path = CACHE / f'{name}.parquet'
    if out_path.exists():
        print(f"  {name}.parquet already exists, skipping")
        return
    
    print(f"  Converting {tsv_path.name} -> {name}.parquet ...")
    t0 = time.time()
    
    chunks = []
    for i, chunk in enumerate(pd.read_csv(str(tsv_path), sep='\t', dtype=str, 
                                           chunksize=chunksize, keep_default_na=False)):
        chunks.append(chunk)
        print(f"    Chunk {i+1}: {len(chunk)} rows loaded ({sum(len(c) for c in chunks)} total)")
        gc.collect()
    
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    df.to_parquet(str(out_path), index=False, engine='pyarrow')
    print(f"  Done: {len(df)} rows in {time.time()-t0:.1f}s -> {out_path}")
    
    # Also save per-country parquets for efficient loading
    if 'country' in df.columns:
        for country in df['country'].unique():
            country_safe = country.replace(' ', '_')
            cpath = CACHE / f'{name}_{country_safe}.parquet'
            df[df['country'] == country].to_parquet(str(cpath), index=False)
            print(f"    {country}: {len(df[df['country']==country])} rows -> {cpath.name}")
    
    del df
    gc.collect()

def convert_gt(tsv_path: Path):
    """Convert ground truth TSV to Parquet."""
    out_path = CACHE / 'train_ground_truth.parquet'
    if out_path.exists():
        print(f"  train_ground_truth.parquet already exists, skipping")
        return
    
    print(f"  Converting ground truth...")
    t0 = time.time()
    df = pd.read_csv(str(tsv_path), sep='\t', dtype=str, keep_default_na=False)
    df.to_parquet(str(out_path), index=False)
    print(f"  Done: {len(df)} rows in {time.time()-t0:.1f}s")
    del df
    gc.collect()

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 0: Converting TSV -> Parquet (run once)")
    print("=" * 60)
    
    t_total = time.time()
    
    # Training data
    print("\n[Train Source 1]")
    convert_tsv_to_parquet(DATA / 'train' / 'train_source1.tsv', 'train_s1')
    
    print("\n[Train Source 2]")
    convert_tsv_to_parquet(DATA / 'train' / 'train_source2.tsv', 'train_s2')
    
    print("\n[Train Source 3]")
    convert_tsv_to_parquet(DATA / 'train' / 'train_source3.tsv', 'train_s3')
    
    print("\n[Train Ground Truth]")
    convert_gt(DATA / 'train' / 'train_ground_truth.tsv')
    
    # Test data
    print("\n[Test Source 1]")
    convert_tsv_to_parquet(DATA / 'test' / 'test_source1.tsv', 'test_s1')
    
    print("\n[Test Source 2]")
    convert_tsv_to_parquet(DATA / 'test' / 'test_source2.tsv', 'test_s2')
    
    print("\n[Test Source 3]")
    convert_tsv_to_parquet(DATA / 'test' / 'test_source3.tsv', 'test_s3')
    
    print(f"\nAll conversions done in {(time.time()-t_total)/60:.1f} minutes")
