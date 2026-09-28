"""
Fast, zero-swap candidate generation benchmark.
- Only loads GT for evaluated queries (~1 MB instead of 1950 MB).
- Uses batch_size=200 for sparse matrix multiplication to prevent RAM spikes.
- Uses word (1,2) for address TF-IDF.
- Measures exact pair recall, complete entity recall, and candidate counts.
"""
import sys
from pathlib import Path
BASE = Path('d:/RESOURCES/AmazonMLChallenge')
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import time
import gc
import re
import os
import psutil
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata

def log_mem(step=""):
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / 1e6
    print(f"[{step}] RAM: {rss_mb:.1f} MB")

def clean_text(s):
    if not s or not isinstance(s, str):
        return ""
    s = unicodedata.normalize('NFKC', s).lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

LEGAL_SUFFIXES = sorted([
    'inc', 'incorporated', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'corp', 'corporation', 'co', 'company', 'enterprises', 'enterprise',
    'group', 'holdings', 'holding', 'international', 'intl', 'services',
    'service', 'solutions', 'solution', 'technologies', 'technology', 'tech',
    'industries', 'industry', 'associates', 'association', 'assoc',
    'foundation', 'fund', 'trust', 'partners', 'partnership',
    'sa', 'sarl', 'sas', 'eurl', 'gmbh', 'ag', 'pty', 'nv', 'bv', 'plc', 'lp', 'na',
], key=len, reverse=True)
_SUF_PAT = re.compile(r'[\s,]*\b(?:' + '|'.join(LEGAL_SUFFIXES) + r')\b\.?[\s,]*$', re.IGNORECASE)
_PAREN_PAT = re.compile(r'\s*\([^)]*\)\s*$')

def strip_suffix(s):
    prev = None
    while s != prev:
        prev = s
        s = _SUF_PAT.sub('', s).strip()
    return _PAREN_PAT.sub('', s).strip()

def sorted_tokens(s):
    t = s.split()
    return ' '.join(sorted(t)) if t else ''

def first_two_tokens(s):
    t = s.split()
    return ' '.join(t[:2]) if len(t) >= 2 else s

def query_sparse_topk_fast(query_mat, cand_mat, cand_ids, top_k=25, batch_size=200):
    """Memory-safe sparse matrix multiplication using small batch sizes."""
    n_queries = query_mat.shape[0]
    results = [[] for _ in range(n_queries)]
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = query_mat[start:end]
        sim_batch = q_batch.dot(cand_mat.T)
        
        for i in range(sim_batch.shape[0]):
            q_idx = start + i
            r_start = sim_batch.indptr[i]
            r_end = sim_batch.indptr[i + 1]
            if r_end == r_start:
                continue
            r_data = sim_batch.data[r_start:r_end]
            r_indices = sim_batch.indices[r_start:r_end]
            
            if len(r_data) <= top_k:
                part = np.argsort(r_data)[::-1]
            else:
                p_idx = np.argpartition(r_data, -top_k)[-top_k:]
                part = p_idx[np.argsort(r_data[p_idx])[::-1]]
            
            results[q_idx] = [(cand_ids[r_indices[idx]], float(r_data[idx])) 
                              for idx in part if r_data[idx] > 0]
        del sim_batch
    return results

def main():
    log_mem("Start")
    CACHE = BASE / 'gemini_solution' / 'cache'
    
    # 1. Select 2,000 S1 queries from India
    print("\n--- 1. Loading S1 Queries ---")
    s1_df = pd.read_parquet(CACHE / 'train_s1_India.parquet').head(2000)
    q_ids = s1_df['entity_id'].values
    q_set = set(q_ids)
    q_names = [clean_text(x) for x in s1_df['business_name']]
    q_addrs = [clean_text(x) for x in s1_df['business_address']]
    q_suf = [strip_suffix(x) for x in q_names]
    q_sort = [sorted_tokens(x) for x in q_names]
    q_first2 = [first_two_tokens(x) for x in q_names]
    del s1_df
    gc.collect()
    log_mem("S1 Loaded")

    # 2. Load ground truth ONLY for these 2,000 queries (< 1 MB RAM!)
    print("\n--- 2. Loading Filtered Ground Truth ---")
    gt_df = pd.read_parquet(CACHE / 'train_ground_truth.parquet')
    mask = gt_df['source1_entity_id'].isin(q_set)
    gt_subset = gt_df[mask]
    s1_vals = gt_subset['source1_entity_id'].values
    m_vals = gt_subset['matched_entity_ids'].fillna('').values
    sample_gt = {s1: set(m.split(',')) if m else set() for s1, m in zip(s1_vals, m_vals)}
    del gt_df, gt_subset, s1_vals, m_vals, mask
    gc.collect()
    log_mem("Filtered GT Loaded")

    total_true = sum(len(v) for v in sample_gt.values())
    n_sing = sum(1 for v in sample_gt.values() if len(v) == 0)
    n_non_sing = len(sample_gt) - n_sing
    print(f"2,000 Queries: {total_true} true pairs, {n_sing} singletons, {n_non_sing} with matches")

    candidates_by_query = defaultdict(set)

    def evaluate_candidates(tag=""):
        hits = 0
        comp = 0
        c_lens = [len(candidates_by_query[q]) for q in q_ids]
        for q, true_set in sample_gt.items():
            if not true_set:
                continue
            ret = candidates_by_query[q]
            h = len(true_set & ret)
            hits += h
            if h == len(true_set):
                comp += 1
        p_rec = hits / max(total_true, 1)
        c_rec = comp / max(n_non_sing, 1)
        mean_c = np.mean(c_lens)
        p95_c = np.percentile(c_lens, 95)
        print(f"[{tag}] Pair Recall: {p_rec*100:.2f}% | Complete: {c_rec*100:.2f}% | Mean: {mean_c:.1f} | P95: {p95_c:.0f} | Max: {max(c_lens)}")
        return p_rec, c_rec, mean_c

    # 3. Process S2 India
    print("\n--- 3. Processing Source 2 India (2,017,799 records) ---")
    t0 = time.time()
    s2_df = pd.read_parquet(CACHE / 'train_s2_India.parquet')
    c2_ids = s2_df['entity_id'].values
    c2_names = [clean_text(x) for x in s2_df['business_name']]
    c2_addrs = [clean_text(x) for x in s2_df['business_address']]
    del s2_df
    gc.collect()
    log_mem("S2 In Memory")

    # 3a. Exact suffix on S2
    s2_suf_idx = defaultdict(list)
    for cid, name in zip(c2_ids, c2_names):
        suf = strip_suffix(name)
        if suf: s2_suf_idx[suf].append(cid)
    for qid, suf in zip(q_ids, q_suf):
        if suf:
            m = s2_suf_idx.get(suf, [])
            if 0 < len(m) <= 150: candidates_by_query[qid].update(m)
    del s2_suf_idx
    gc.collect()

    # 3b. Exact sorted on S2
    s2_sort_idx = defaultdict(list)
    for cid, name in zip(c2_ids, c2_names):
        srt = sorted_tokens(name)
        if srt: s2_sort_idx[srt].append(cid)
    for qid, srt in zip(q_ids, q_sort):
        if srt:
            m = s2_sort_idx.get(srt, [])
            if 0 < len(m) <= 150: candidates_by_query[qid].update(m)
    del s2_sort_idx
    gc.collect()

    # 3c. First 2 tokens exact match on S2
    s2_f2_idx = defaultdict(list)
    for cid, name in zip(c2_ids, c2_names):
        f2 = first_two_tokens(name)
        if f2 and len(f2) >= 6: s2_f2_idx[f2].append(cid)
    for qid, f2 in zip(q_ids, q_first2):
        if f2 and len(f2) >= 6:
            m = s2_f2_idx.get(f2, [])
            if 0 < len(m) <= 50: candidates_by_query[qid].update(m)
    del s2_f2_idx
    gc.collect()
    log_mem("After S2 Exact Matches")

    # 3d. TF-IDF Name on S2 (char_wb 3-5, max_features=80000)
    print("TF-IDF Name S2...")
    name_vec2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
                                sublinear_tf=True, dtype=np.float32)
    s2_name_mat = name_vec2.fit_transform(c2_names)
    q_name_mat2 = name_vec2.transform(q_names)
    del name_vec2
    gc.collect()
    log_mem("After S2 Name Mat")
    
    topk_name_s2 = query_sparse_topk_fast(q_name_mat2, s2_name_mat, c2_ids, top_k=25, batch_size=200)
    for q_idx, cands in enumerate(topk_name_s2):
        candidates_by_query[q_ids[q_idx]].update([c for c, _ in cands])
    del s2_name_mat, q_name_mat2, topk_name_s2
    gc.collect()
    log_mem("After S2 Name Query")

    # 3e. TF-IDF Addr on S2 (word 1-2, max_features=50000)
    print("TF-IDF Addr S2...")
    addr_vec2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000,
                                sublinear_tf=True, dtype=np.float32)
    s2_addr_mat = addr_vec2.fit_transform(c2_addrs)
    q_addr_mat2 = addr_vec2.transform(q_addrs)
    del addr_vec2
    gc.collect()
    log_mem("After S2 Addr Mat")
    
    topk_addr_s2 = query_sparse_topk_fast(q_addr_mat2, s2_addr_mat, c2_ids, top_k=20, batch_size=200)
    for q_idx, cands in enumerate(topk_addr_s2):
        candidates_by_query[q_ids[q_idx]].update([c for c, _ in cands])
    del s2_addr_mat, q_addr_mat2, topk_addr_s2, c2_ids, c2_names, c2_addrs
    gc.collect()
    log_mem("After S2 Complete")
    print(f"S2 finished in {time.time() - t0:.1f}s")
    evaluate_candidates("After S2")

    # 4. Process S3 India
    print("\n--- 4. Processing Source 3 India (2,115,547 records) ---")
    t1 = time.time()
    s3_df = pd.read_parquet(CACHE / 'train_s3_India.parquet')
    c3_ids = s3_df['entity_id'].values
    c3_names = [clean_text(x) for x in s3_df['business_name']]
    c3_addrs = [clean_text(x) for x in s3_df['business_address']]
    del s3_df
    gc.collect()
    log_mem("S3 In Memory")

    # 4a. Exact suffix on S3
    s3_suf_idx = defaultdict(list)
    for cid, name in zip(c3_ids, c3_names):
        suf = strip_suffix(name)
        if suf: s3_suf_idx[suf].append(cid)
    for qid, suf in zip(q_ids, q_suf):
        if suf:
            m = s3_suf_idx.get(suf, [])
            if 0 < len(m) <= 150: candidates_by_query[qid].update(m)
    del s3_suf_idx
    gc.collect()

    # 4b. Exact sorted on S3
    s3_sort_idx = defaultdict(list)
    for cid, name in zip(c3_ids, c3_names):
        srt = sorted_tokens(name)
        if srt: s3_sort_idx[srt].append(cid)
    for qid, srt in zip(q_ids, q_sort):
        if srt:
            m = s3_sort_idx.get(srt, [])
            if 0 < len(m) <= 150: candidates_by_query[qid].update(m)
    del s3_sort_idx
    gc.collect()

    # 4c. First 2 tokens exact match on S3
    s3_f2_idx = defaultdict(list)
    for cid, name in zip(c3_ids, c3_names):
        f2 = first_two_tokens(name)
        if f2 and len(f2) >= 6: s3_f2_idx[f2].append(cid)
    for qid, f2 in zip(q_ids, q_first2):
        if f2 and len(f2) >= 6:
            m = s3_f2_idx.get(f2, [])
            if 0 < len(m) <= 50: candidates_by_query[qid].update(m)
    del s3_f2_idx
    gc.collect()
    log_mem("After S3 Exact Matches")

    # 4d. TF-IDF Name on S3
    print("TF-IDF Name S3...")
    name_vec3 = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
                                sublinear_tf=True, dtype=np.float32)
    s3_name_mat = name_vec3.fit_transform(c3_names)
    q_name_mat3 = name_vec3.transform(q_names)
    del name_vec3
    gc.collect()
    log_mem("After S3 Name Mat")
    
    topk_name_s3 = query_sparse_topk_fast(q_name_mat3, s3_name_mat, c3_ids, top_k=25, batch_size=200)
    for q_idx, cands in enumerate(topk_name_s3):
        candidates_by_query[q_ids[q_idx]].update([c for c, _ in cands])
    del s3_name_mat, q_name_mat3, topk_name_s3
    gc.collect()
    log_mem("After S3 Name Query")

    # 4e. TF-IDF Addr on S3
    print("TF-IDF Addr S3...")
    addr_vec3 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50000,
                                sublinear_tf=True, dtype=np.float32)
    s3_addr_mat = addr_vec3.fit_transform(c3_addrs)
    q_addr_mat3 = addr_vec3.transform(q_addrs)
    del addr_vec3
    gc.collect()
    log_mem("After S3 Addr Mat")
    
    topk_addr_s3 = query_sparse_topk_fast(q_addr_mat3, s3_addr_mat, c3_ids, top_k=20, batch_size=200)
    for q_idx, cands in enumerate(topk_addr_s3):
        candidates_by_query[q_ids[q_idx]].update([c for c, _ in cands])
    del s3_addr_mat, q_addr_mat3, topk_addr_s3, c3_ids, c3_names, c3_addrs
    gc.collect()
    log_mem("After S3 Complete")
    print(f"S3 finished in {time.time() - t1:.1f}s")

    print("\n--- FINAL COMBINED BLOCKING EVALUATION (S2 + S3) ---")
    p_rec, c_rec, mean_c = evaluate_candidates("TOTAL UNION")
    log_mem("Done")

if __name__ == '__main__':
    main()
