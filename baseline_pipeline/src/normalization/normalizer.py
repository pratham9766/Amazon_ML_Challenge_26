import pandas as pd
import unicodedata
import re

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'private', 
    'corp', 'corporation', 'co', 'company', 'enterprises', 'enterprise', 
    'group', 'holdings', 'holding', 'international', 'intl', 'services', 
    'service', 'solutions', 'solution', 'technologies', 'technology', 'tech', 
    'industries', 'industry', 'associates', 'association', 'assoc', 
    'foundation', 'fund', 'trust', 'partners', 'partnership', 'sa', 'sarl', 
    'sas', 'eurl', 'sàrl', 'gmbh', 'ag', 'pty', 'nv', 'bv', 'plc', 'lp', 'na'
}

ADDR_ABBREVIATIONS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard', 
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 
    'cir': 'circle', 'hwy': 'highway', 'pkwy': 'parkway', 'apt': 'apartment', 
    'ste': 'suite', 'fl': 'floor', 'bldg': 'building', 'dept': 'department', 
    'ofc': 'office', 'rte': 'route', 'no': 'number', 'nr': 'near'
}


def _safe_str(s: pd.Series) -> pd.Series:
    """Helper to convert to string and handle NaN gracefully."""
    return s.fillna('').astype(str)


def normalize_name_nfkc(s: pd.Series) -> pd.Series:
    """NFKC + lowercase."""
    s = _safe_str(s)
    return s.apply(lambda x: unicodedata.normalize('NFKC', x)).str.lower()


def normalize_name_basic(s: pd.Series) -> pd.Series:
    """NFKC + lower + collapse whitespace + normalize punctuation (& -> and) + strip leading/trailing whitespace."""
    s = normalize_name_nfkc(s)
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s.str.strip()


def normalize_name_suffix_stripped(s: pd.Series) -> pd.Series:
    """Remove legal suffixes from basic normalized name."""
    suffixes_joined = '|'.join(sorted(LEGAL_SUFFIXES, key=len, reverse=True))
    
    # Pattern to match a legal suffix with optional dot
    suffix_pattern = rf'(?:\b(?:{suffixes_joined})\b\.?)'
    
    # Pattern to match anything in parentheses at the end (e.g. (pvt), (india))
    paren_pattern = r'(?:\([^)]*\))'
    
    # Combine them: we want to match a sequence of (suffix_pattern or paren_pattern) at the end of the string.
    combined_pattern = rf'(?:(?:{suffix_pattern}|{paren_pattern})[\s,]*)+$'
    
    res = s.str.replace(combined_pattern, '', regex=True)
    res = res.str.replace(r'[\s,]+$', '', regex=True)
    return res


def normalize_name_sorted_tokens(s: pd.Series) -> pd.Series:
    """Split basic name into tokens, sort alphabetically, rejoin with space."""
    return s.str.split().apply(lambda x: ' '.join(sorted(x)) if isinstance(x, list) else '')


def normalize_name_alnum(s: pd.Series) -> pd.Series:
    """Only keep alphanumeric and space chars from basic name."""
    s = s.str.replace(r'[^\w\s]', '', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s.str.strip()


def extract_name_numeric_tokens(s: pd.Series) -> pd.Series:
    """Extract all digit sequences, join with space."""
    return s.str.findall(r'\d+').apply(lambda x: ' '.join(x) if isinstance(x, list) else '')


def normalize_addr_nfkc(s: pd.Series) -> pd.Series:
    """NFKC + lowercase."""
    s = _safe_str(s)
    return s.apply(lambda x: unicodedata.normalize('NFKC', x)).str.lower()


def normalize_addr_basic(s: pd.Series) -> pd.Series:
    """NFKC + lower + collapse whitespace + normalize common abbreviations."""
    s = normalize_addr_nfkc(s)
    
    for abbr, expanded in ADDR_ABBREVIATIONS.items():
        # Match as a whole word, optionally followed by a period.
        pattern = rf'\b{abbr}\b\.?'
        s = s.str.replace(pattern, expanded, regex=True)
        
    s = s.str.replace(r'\s+', ' ', regex=True)
    return s.str.strip()


def normalize_addr_tokens(s: pd.Series) -> pd.Series:
    """Whitespace-split basic address into sorted unique tokens, rejoin."""
    return s.str.split().apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')


def extract_addr_numeric_tokens(s: pd.Series) -> pd.Series:
    """Extract all digit sequences from address."""
    return s.str.findall(r'\d+').apply(lambda x: ' '.join(x) if isinstance(x, list) else '')


def extract_addr_long_tokens(s: pd.Series) -> pd.Series:
    """Tokens with length >= 4 from basic address, sorted and joined."""
    return s.str.split().apply(lambda x: ' '.join(sorted([t for t in x if len(t) >= 4])) if isinstance(x, list) else '')


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ALL normalizations to a DataFrame with columns 'business_name' and 'business_address'."""
    out_df = df.copy()
    
    if 'business_name' in out_df.columns:
        out_df['name_nfkc'] = normalize_name_nfkc(out_df['business_name'])
        
        basic_name = normalize_name_basic(out_df['business_name'])
        out_df['name_basic'] = basic_name
        out_df['name_suffix_stripped'] = normalize_name_suffix_stripped(basic_name)
        out_df['name_sorted_tokens'] = normalize_name_sorted_tokens(basic_name)
        out_df['name_alnum'] = normalize_name_alnum(basic_name)
        out_df['name_numeric'] = extract_name_numeric_tokens(basic_name)
    
    if 'business_address' in out_df.columns:
        out_df['addr_nfkc'] = normalize_addr_nfkc(out_df['business_address'])
        
        basic_addr = normalize_addr_basic(out_df['business_address'])
        out_df['addr_basic'] = basic_addr
        out_df['addr_tokens'] = normalize_addr_tokens(basic_addr)
        out_df['addr_numeric'] = extract_addr_numeric_tokens(basic_addr)
        out_df['addr_long_tokens'] = extract_addr_long_tokens(basic_addr)
        
    return out_df
