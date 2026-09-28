"""Feature engineering for entity resolution candidate pairs.

Computes name similarity, address similarity, retrieval provenance,
and cross-field features for candidate pairs.

Designed for batch processing with memory efficiency.
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
from typing import List, Optional, Dict, Set
import re

def _batch_fuzz(s1_vals, c_vals, func):
    """Apply a rapidfuzz function to arrays of string pairs."""
    results = np.empty(len(s1_vals), dtype=np.float32)
    for i in range(len(s1_vals)):
        results[i] = func(s1_vals[i], c_vals[i]) / 100.0 if s1_vals[i] and c_vals[i] else 0.0
    return results

def _batch_distance(s1_vals, c_vals, func):
    """Apply a rapidfuzz distance function to arrays of string pairs."""
    results = np.empty(len(s1_vals), dtype=np.float32)
    for i in range(len(s1_vals)):
        results[i] = func(s1_vals[i], c_vals[i]) if s1_vals[i] and c_vals[i] else 0.0
    return results

def _batch_word_jaccard(s1_vals, c_vals):
    results = np.empty(len(s1_vals), dtype=np.float32)
    for i in range(len(s1_vals)):
        s1 = set(str(s1_vals[i]).split())
        c = set(str(c_vals[i]).split())
        if not s1 or not c:
            results[i] = 0.0
        else:
            results[i] = len(s1 & c) / len(s1 | c)
    return results

def _batch_char_ngrams(s1_vals, c_vals, n=3):
    results = np.empty(len(s1_vals), dtype=np.float32)
    for i in range(len(s1_vals)):
        s1 = str(s1_vals[i])
        c = str(c_vals[i])
        if len(s1) < n or len(c) < n:
            results[i] = 0.0
            continue
        s1_ngrams = set(s1[j:j+n] for j in range(len(s1)-n+1))
        c_ngrams = set(c[j:j+n] for j in range(len(c)-n+1))
        if not s1_ngrams or not c_ngrams:
            results[i] = 0.0
        else:
            results[i] = len(s1_ngrams & c_ngrams) / len(s1_ngrams | c_ngrams)
    return results

def compute_name_features(s1_names: pd.DataFrame, cand_names: pd.DataFrame) -> pd.DataFrame:
    """Compute all name similarity features."""
    features = pd.DataFrame(index=s1_names.index)
    
    # Extract string arrays
    s1_basic = s1_names.get('name_basic', pd.Series(['']*len(s1_names))).fillna('').astype(str).values
    c_basic = cand_names.get('name_basic', pd.Series(['']*len(cand_names))).fillna('').astype(str).values
    
    s1_suffix = s1_names.get('name_suffix_stripped', pd.Series(['']*len(s1_names))).fillna('').astype(str).values  
    c_suffix = cand_names.get('name_suffix_stripped', pd.Series(['']*len(cand_names))).fillna('').astype(str).values
    
    s1_sorted = s1_names.get('name_sorted_tokens', pd.Series(['']*len(s1_names))).fillna('').astype(str).values
    c_sorted = cand_names.get('name_sorted_tokens', pd.Series(['']*len(cand_names))).fillna('').astype(str).values
    
    # 1-3. Exact matches
    features['name_exact_basic'] = (s1_basic == c_basic).astype(np.int8)
    features['name_exact_suffix'] = (s1_suffix == c_suffix).astype(np.int8) 
    features['name_exact_sorted'] = (s1_sorted == c_sorted).astype(np.int8)
    
    # 4-9. Fuzz and distance
    features['name_fuzz_ratio'] = _batch_fuzz(s1_basic, c_basic, fuzz.ratio)
    features['name_fuzz_partial'] = _batch_fuzz(s1_basic, c_basic, fuzz.partial_ratio)
    features['name_fuzz_token_sort'] = _batch_fuzz(s1_basic, c_basic, fuzz.token_sort_ratio)
    features['name_fuzz_token_set'] = _batch_fuzz(s1_basic, c_basic, fuzz.token_set_ratio)
    features['name_jaro_winkler'] = _batch_distance(s1_basic, c_basic, distance.JaroWinkler.similarity)
    features['name_levenshtein_norm'] = 1.0 - _batch_distance(s1_basic, c_basic, distance.Levenshtein.normalized_distance)
    
    # 10-11. Jaccard
    features['name_word_jaccard'] = _batch_word_jaccard(s1_basic, c_basic)
    features['name_char3_jaccard'] = _batch_char_ngrams(s1_basic, c_basic, 3)
    
    # 12-15. Prefixes and tokens
    features['name_prefix_3'] = (pd.Series(s1_basic).str[:3] == pd.Series(c_basic).str[:3]).astype(np.int8)
    features['name_prefix_5'] = (pd.Series(s1_basic).str[:5] == pd.Series(c_basic).str[:5]).astype(np.int8)
    
    s1_first = [s.split()[0] if s else '' for s in s1_basic]
    c_first = [c.split()[0] if c else '' for c in c_basic]
    features['name_first_token_eq'] = (np.array(s1_first) == np.array(c_first)).astype(np.int8)
    
    s1_last = [s.split()[-1] if s else '' for s in s1_basic]
    c_last = [c.split()[-1] if c else '' for c in c_basic]
    features['name_last_token_eq'] = (np.array(s1_last) == np.array(c_last)).astype(np.int8)
    
    # 16-17. Lengths
    s1_len_tok = np.array([len(s.split()) for s in s1_basic])
    c_len_tok = np.array([len(s.split()) for s in c_basic])
    features['name_token_count_diff'] = np.abs(s1_len_tok - c_len_tok)
    
    s1_len_char = np.array([len(s) for s in s1_basic])
    c_len_char = np.array([len(s) for s in c_basic])
    min_len = np.minimum(s1_len_char, c_len_char)
    max_len = np.maximum(s1_len_char, c_len_char)
    features['name_length_ratio'] = np.where(max_len == 0, 0, min_len / max_len)
    
    # 18. Numeric agreement
    def num_agree(s1, c):
        nums1 = set(re.findall(r'\d+', str(s1)))
        nums2 = set(re.findall(r'\d+', str(c)))
        if not nums1 and not nums2:
            return 1.0
        if not nums1 or not nums2:
            return 0.0
        return len(nums1 & nums2) / len(nums1 | nums2)
        
    features['name_numeric_agreement'] = [_batch_num(a,b) for a,b in zip(s1_basic, c_basic)]
    
    # 19. Containment
    features['name_containment'] = np.array([1 if (a and b and (a in b or b in a)) else 0 for a, b in zip(s1_basic, c_basic)], dtype=np.int8)
    
    # 20-22
    features['name_len_s1'] = s1_len_char
    features['name_len_cand'] = c_len_char
    features['name_alnum_eq'] = (pd.Series(s1_basic).str.replace(r'[^a-zA-Z0-9]', '', regex=True) == pd.Series(c_basic).str.replace(r'[^a-zA-Z0-9]', '', regex=True)).astype(np.int8)
    
    return features

def _batch_num(a,b):
    nums1 = set(re.findall(r'\d+', str(a)))
    nums2 = set(re.findall(r'\d+', str(b)))
    if not nums1 and not nums2: return 1.0
    if not nums1 or not nums2: return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)

def compute_address_features(s1_addrs: pd.DataFrame, cand_addrs: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=s1_addrs.index)
    s1_basic = s1_addrs.get('address_basic', pd.Series(['']*len(s1_addrs))).fillna('').astype(str).values
    c_basic = cand_addrs.get('address_basic', pd.Series(['']*len(cand_addrs))).fillna('').astype(str).values
    
    # 23-26
    features['addr_exact_basic'] = (s1_basic == c_basic).astype(np.int8)
    features['addr_fuzz_ratio'] = _batch_fuzz(s1_basic, c_basic, fuzz.ratio)
    features['addr_fuzz_token_set'] = _batch_fuzz(s1_basic, c_basic, fuzz.token_set_ratio)
    features['addr_word_jaccard'] = _batch_word_jaccard(s1_basic, c_basic)
    
    # 27-31
    overlap = []
    num_overlap = []
    f_num = []
    l_num = []
    long_jac = []
    for s, c in zip(s1_basic, c_basic):
        st = set(s.split())
        ct = set(c.split())
        overlap.append(len(st & ct) / min(len(st), len(ct)) if st and ct else 0.0)
        
        sn = set(re.findall(r'\d+', s))
        cn = set(re.findall(r'\d+', c))
        num_overlap.append(len(sn & cn) / max(len(sn), len(cn)) if sn and cn else (1.0 if not sn and not cn else 0.0))
        
        snl = re.findall(r'\d+', s)
        cnl = re.findall(r'\d+', c)
        f_num.append(1 if snl and cnl and snl[0] == cnl[0] else 0)
        l_num.append(1 if snl and cnl and snl[-1] == cnl[-1] else 0)
        
        stl = set(t for t in s.split() if len(t) >= 4)
        ctl = set(t for t in c.split() if len(t) >= 4)
        long_jac.append(len(stl & ctl) / len(stl | ctl) if stl or ctl else 0.0)
        
    features['addr_token_overlap_ratio'] = overlap
    features['addr_numeric_overlap'] = num_overlap
    features['addr_first_number_eq'] = f_num
    features['addr_last_number_eq'] = l_num
    features['addr_long_token_overlap'] = long_jac
    
    # 32-40
    s1_len = np.array([len(x) for x in s1_basic])
    c_len = np.array([len(x) for x in c_basic])
    min_l = np.minimum(s1_len, c_len)
    max_l = np.maximum(s1_len, c_len)
    features['addr_length_ratio'] = np.where(max_l == 0, 0, min_l / max_l)
    
    features['addr_token_count_diff'] = np.abs(np.array([len(s.split()) for s in s1_basic]) - np.array([len(c.split()) for c in c_basic]))
    features['addr_missing_s1'] = (s1_len == 0).astype(np.int8)
    features['addr_missing_cand'] = (c_len == 0).astype(np.int8)
    features['addr_both_missing'] = (features['addr_missing_s1'] & features['addr_missing_cand']).astype(np.int8)
    features['addr_either_missing'] = (features['addr_missing_s1'] | features['addr_missing_cand']).astype(np.int8)
    features['addr_numeric_count_diff'] = np.abs(np.array([len(re.findall(r'\d+', s)) for s in s1_basic]) - np.array([len(re.findall(r'\d+', c)) for c in c_basic]))
    features['addr_len_s1'] = s1_len
    features['addr_len_cand'] = c_len
    
    return features

def compute_all_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    batch_size: int = 100000,
) -> pd.DataFrame:
    """Compute all features for candidate pairs."""
    s1_merged = pairs_df[['source1_entity_id']].merge(s1_df, left_on='source1_entity_id', right_index=True, how='left')
    cand_merged = pairs_df[['candidate_entity_id']].merge(candidates_df, left_on='candidate_entity_id', right_index=True, how='left')
    
    name_feat = compute_name_features(s1_merged, cand_merged)
    addr_feat = compute_address_features(s1_merged, cand_merged)
    
    feat = pd.concat([name_feat, addr_feat], axis=1)
    
    # 41-52 Cross-field
    feat['name_x_addr_product'] = feat['name_fuzz_ratio'] * feat['addr_fuzz_ratio']
    feat['name_x_addr_min'] = np.minimum(feat['name_fuzz_ratio'], feat['addr_fuzz_ratio'])
    feat['name_x_addr_max'] = np.maximum(feat['name_fuzz_ratio'], feat['addr_fuzz_ratio'])
    feat['high_name_low_addr'] = ((feat['name_fuzz_ratio'] > 0.8) & (feat['addr_fuzz_ratio'] < 0.3)).astype(np.int8)
    feat['high_addr_low_name'] = ((feat['addr_fuzz_ratio'] > 0.8) & (feat['name_fuzz_ratio'] < 0.3)).astype(np.int8)
    feat['name_missing'] = feat['name_len_s1'] == 0
    feat['both_fields_strong'] = ((feat['name_fuzz_ratio'] > 0.7) & (feat['addr_fuzz_ratio'] > 0.7)).astype(np.int8)
    feat['source_type'] = pairs_df['candidate_entity_id'].astype(str).str.startswith('S3').astype(np.int8)
    
    eps = 1e-6
    feat['name_addr_harmonic_mean'] = 2 * feat['name_fuzz_ratio'] * feat['addr_fuzz_ratio'] / (feat['name_fuzz_ratio'] + feat['addr_fuzz_ratio'] + eps)
    feat['name_addr_diff'] = np.abs(feat['name_fuzz_ratio'] - feat['addr_fuzz_ratio'])
    
    fuzz_cols = [c for c in feat.columns if 'fuzz' in c]
    if fuzz_cols:
        feat['max_fuzz_score'] = feat[fuzz_cols].max(axis=1)
    else:
        feat['max_fuzz_score'] = 0.0
    feat['min_fuzz_score'] = feat['name_x_addr_min']
    
    return feat
