import pandas as pd
import os
import gc
from pathlib import Path
from typing import Optional, Dict, Set, Tuple, List
import hashlib

BASE_DIR = Path('d:/RESOURCES/AmazonMLChallenge')
DATASET_DIR = BASE_DIR / 'dataset-20260925T050739Z-1-001' / 'dataset'
GEMINI_DIR = BASE_DIR / 'baseline_pipeline'
CACHE_DIR = GEMINI_DIR / 'cache'

def ensure_dirs():
    """Create all necessary directories."""
    for d in ['cache', 'artifacts', 'models', 'output', 'reports', 'experiments']:
        (GEMINI_DIR / d).mkdir(parents=True, exist_ok=True)

def load_tsv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a TSV file with proper settings."""
    df = pd.read_csv(path, sep='\t', dtype=str, nrows=nrows, keep_default_na=False)
    return df

def load_source(split: str, source: int, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a source file. split='train' or 'test', source=1,2,3."""
    path = DATASET_DIR / split / f'{split}_source{source}.tsv'
    return load_tsv(str(path), nrows=nrows)

def load_ground_truth(nrows: Optional[int] = None) -> pd.DataFrame:
    """Load training ground truth."""
    path = DATASET_DIR / 'train' / 'train_ground_truth.tsv'
    return load_tsv(str(path), nrows=nrows)

def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth into {s1_id: set(matched_ids)}."""
    result = {}
    for _, row in gt_df.iterrows():
        s1_id = row.get('source1_entity_id')
        matched = row.get('matched_entity_ids', '')
        if pd.notna(matched) and str(matched).strip():
            result[s1_id] = set(str(matched).split(','))
        else:
            result[s1_id] = set()
    return result

def build_reverse_mapping(gt: Dict[str, Set[str]]) -> Dict[str, str]:
    """Build reverse mapping: candidate_id -> s1_id."""
    reverse = {}
    for s1_id, matched in gt.items():
        for m_id in matched:
            reverse[m_id] = s1_id
    return reverse

def cache_to_parquet(df: pd.DataFrame, name: str) -> str:
    """Cache a DataFrame as Parquet."""
    ensure_dirs()
    path = CACHE_DIR / f'{name}.parquet'
    df.to_parquet(str(path), index=False)
    return str(path)

def load_from_parquet(name: str) -> Optional[pd.DataFrame]:
    """Load from Parquet cache if exists."""
    path = CACHE_DIR / f'{name}.parquet'
    if path.exists():
        return pd.read_parquet(str(path))
    return None

def load_or_cache_source(split: str, source: int, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load source with Parquet caching."""
    suffix = f'_n{nrows}' if nrows else ''
    cache_name = f'{split}_source{source}{suffix}'
    df = load_from_parquet(cache_name)
    if df is not None:
        return df
    df = load_source(split, source, nrows=nrows)
    cache_to_parquet(df, cache_name)
    return df

def get_country_partition(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Get records for a specific country."""
    if 'country' not in df.columns:
        return df.copy()
    return df[df['country'] == country].copy()

def get_unique_countries(df: pd.DataFrame) -> List[str]:
    """Get unique countries in a DataFrame."""
    if 'country' not in df.columns:
        return []
    # Dropna to avoid passing nans which can mess up sorting
    valid_countries = df['country'].dropna().unique().tolist()
    return sorted(valid_countries)

def chunked_load_source(split: str, source: int, chunksize: int = 500000):
    """Load a source file in chunks for memory efficiency."""
    path = DATASET_DIR / split / f'{split}_source{source}.tsv'
    return pd.read_csv(str(path), sep='\t', dtype=str, chunksize=chunksize, keep_default_na=False)
