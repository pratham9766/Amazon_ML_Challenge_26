"""
Benchmark and evaluate candidate generation (blocking) channels on held-out S1 queries.
Measures pair recall, complete-entity recall, candidate counts, and analyzes missed positives.
"""
import sys
from pathlib import Path

BASE = Path('d:/RESOURCES/AmazonMLChallenge')
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import time
import gc
import logging
from collections import defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from baseline_pipeline.src.normalization.normalizer import normalize_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('blocking_bench')

BASE = Path('d:/RESOURCES/AmazonMLChallenge')
CACHE = BASE / 'baseline_pipeline' / 'cache'
REPORTS = BASE / 'baseline_pipeline' / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)


def load_gt():
    df = pd.read_parquet(CACHE / 'train_ground_truth.parquet')
    s1_vals = df['source1_entity_id'].values
    m_vals = df['matched_entity_ids'].fillna('').values
    return {s1: set(m.split(',')) if m else set() for s1, m in zip(s1_vals, m_vals)}


def fast_sparse_topk(query_mat: sparse.csr_matrix, cand_mat: sparse.csr_matrix,
                     cand_ids: np.ndarray, top_k: int = 30, batch_size: int = 2000):
    """
    Batched sparse matrix multiplication with vectorized top-k extraction.
    Much faster than per-row getrow() calls.
    """
    n_queries = query_mat.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        q_batch = query_mat[start:end]
        sim_batch = q_batch.dot(cand_mat.T)  # sparse matrix of shape (b, n_cands)

        # Convert each row to top-k using sparse data structures
        for i in range(sim_batch.shape[0]):
            r_start = sim_batch.indptr[i]
            r_end = sim_batch.indptr[i + 1]
            n_nonzeros = r_end - r_start

            if n_nonzeros == 0:
                results.append([])
                continue

            r_data = sim_batch.data[r_start:r_end]
            r_indices = sim_batch.indices[r_start:r_end]

            if n_nonzeros <= top_k:
                sort_idx = np.argsort(r_data)[::-1]
            else:
                part_idx = np.argpartition(r_data, -top_k)[-top_k:]
                sort_idx = part_idx[np.argsort(r_data[part_idx])[::-1]]

            c_list = [(cand_ids[r_indices[idx]], float(r_data[idx])) 
                      for idx in sort_idx if r_data[idx] > 0]
            results.append(c_list)

        del sim_batch
        gc.collect()

    return results


def run_benchmark(country: str = 'India', n_queries: int = 5000):
    log.info(f"=== Running Blocking Benchmark for {country} (queries={n_queries}) ===")
    gt = load_gt()

    # Load S1
    s1_df = pd.read_parquet(CACHE / f'train_s1_{country}.parquet')
    s1_sample = s1_df.head(n_queries).copy()
    s1_sample = normalize_dataframe(s1_sample)
    q_ids = s1_sample['entity_id'].values
    log.info(f"Loaded {len(s1_sample)} query entities")

    # Load S2 and S3
    log.info(f"Loading {country} S2 and S3...")
    s2_df = normalize_dataframe(pd.read_parquet(CACHE / f'train_s2_{country}.parquet'))
    s3_df = normalize_dataframe(pd.read_parquet(CACHE / f'train_s3_{country}.parquet'))
    log.info(f"S2: {len(s2_df)}, S3: {len(s3_df)}")

    # Ground truth for sample
    sample_gt = {qid: gt[qid] for qid in q_ids if qid in gt}
    total_true_pairs = sum(len(v) for v in sample_gt.values())
    n_singletons = sum(1 for v in sample_gt.values() if len(v) == 0)
    n_entities_with_matches = len(sample_gt) - n_singletons
    log.info(f"Sample GT: {total_true_pairs} true pairs, {n_singletons} singletons, {n_entities_with_matches} entities with matches")

    # Evaluate Channels
    channels_results = {}

    # Helper to evaluate candidate dictionary {qid: set(cids)}
    def eval_cands(cand_dict, name):
        found = 0
        comp = 0
        c_counts = []
        for qid, t_set in sample_gt.items():
            c_set = cand_dict.get(qid, set())
            c_counts.append(len(c_set))
            if not t_set:
                continue
            hits = len(t_set & c_set)
            found += hits
            if hits == len(t_set):
                comp += 1
        pair_rec = found / max(total_true_pairs, 1)
        comp_rec = comp / max(n_entities_with_matches, 1)
        mean_c = np.mean(c_counts)
        p95_c = np.percentile(c_counts, 95)
        max_c = np.max(c_counts)
        log.info(f"[{name}] Pair Recall: {pair_rec*100:.2f}% | Complete Entity: {comp_rec*100:.2f}% | Mean Cands: {mean_c:.1f} | P95: {p95_c:.0f} | Max: {max_c}")
        return pair_rec, comp_rec, mean_c

    # 1. Exact Suffix-stripped name
    log.info("--- Testing Exact Suffix Match ---")
    s2_suf_idx = defaultdict(list)
    s3_suf_idx = defaultdict(list)
    for cid, suf in zip(s2_df['entity_id'], s2_df['name_suffix_stripped']):
        if suf: s2_suf_idx[suf].append(cid)
    for cid, suf in zip(s3_df['entity_id'], s3_df['name_suffix_stripped']):
        if suf: s3_suf_idx[suf].append(cid)

    exact_cands = defaultdict(set)
    for qid, suf in zip(s1_sample['entity_id'], s1_sample['name_suffix_stripped']):
        if suf:
            m2 = s2_suf_idx.get(suf, [])
            m3 = s3_suf_idx.get(suf, [])
            if len(m2) <= 150: exact_cands[qid].update(m2)
            if len(m3) <= 150: exact_cands[qid].update(m3)
    eval_cands(exact_cands, "Exact Suffix Name (cap=150)")

    # 2. Exact Sorted Tokens Name
    log.info("--- Testing Exact Sorted Tokens Match ---")
    s2_sort_idx = defaultdict(list)
    s3_sort_idx = defaultdict(list)
    for cid, srt in zip(s2_df['entity_id'], s2_df['name_sorted_tokens']):
        if srt: s2_sort_idx[srt].append(cid)
    for cid, srt in zip(s3_df['entity_id'], s3_df['name_sorted_tokens']):
        if srt: s3_sort_idx[srt].append(cid)

    exact_sort_cands = defaultdict(set)
    for qid, srt in zip(s1_sample['entity_id'], s1_sample['name_sorted_tokens']):
        if srt:
            m2 = s2_sort_idx.get(srt, [])
            m3 = s3_sort_idx.get(srt, [])
            if len(m2) <= 150: exact_sort_cands[qid].update(m2)
            if len(m3) <= 150: exact_sort_cands[qid].update(m3)
    eval_cands(exact_sort_cands, "Exact Sorted Tokens (cap=150)")

    # 3. TF-IDF Name Char (3-5)
    log.info("--- Testing TF-IDF Name (char_wb 3-5) ---")
    name_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=150000,
                               min_df=2, max_df=0.95, sublinear_tf=True, dtype=np.float32)
    cand_names = pd.concat([s2_df['name_basic'], s3_df['name_basic']], ignore_index=True)
    name_vec.fit(cand_names)
    s2_name_mat = name_vec.transform(s2_df['name_basic'])
    s3_name_mat = name_vec.transform(s3_df['name_basic'])
    q_name_mat = name_vec.transform(s1_sample['name_basic'])

    top_k_name = 30
    log.info(f"Querying S2 name (top_k={top_k_name})...")
    s2_name_res = fast_sparse_topk(q_name_mat, s2_name_mat, s2_df['entity_id'].values, top_k=top_k_name)
    log.info(f"Querying S3 name (top_k={top_k_name})...")
    s3_name_res = fast_sparse_topk(q_name_mat, s3_name_mat, s3_df['entity_id'].values, top_k=top_k_name)

    tfidf_name_cands = defaultdict(set)
    for i, qid in enumerate(q_ids):
        tfidf_name_cands[qid].update([cid for cid, _ in s2_name_res[i]])
        tfidf_name_cands[qid].update([cid for cid, _ in s3_name_res[i]])
    eval_cands(tfidf_name_cands, f"TF-IDF Name Top-{top_k_name} (S2+S3)")

    del cand_names, name_vec, s2_name_mat, s3_name_mat, q_name_mat
    gc.collect()

    # 4. TF-IDF Address Char (3-5)
    log.info("--- Testing TF-IDF Address (char_wb 3-5) ---")
    addr_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=150000,
                               min_df=2, max_df=0.95, sublinear_tf=True, dtype=np.float32)
    cand_addrs = pd.concat([s2_df['addr_basic'], s3_df['addr_basic']], ignore_index=True)
    addr_vec.fit(cand_addrs)
    s2_addr_mat = addr_vec.transform(s2_df['addr_basic'])
    s3_addr_mat = addr_vec.transform(s3_df['addr_basic'])
    q_addr_mat = addr_vec.transform(s1_sample['addr_basic'])

    top_k_addr = 25
    log.info(f"Querying S2 addr (top_k={top_k_addr})...")
    s2_addr_res = fast_sparse_topk(q_addr_mat, s2_addr_mat, s2_df['entity_id'].values, top_k=top_k_addr)
    log.info(f"Querying S3 addr (top_k={top_k_addr})...")
    s3_addr_res = fast_sparse_topk(q_addr_mat, s3_addr_mat, s3_df['entity_id'].values, top_k=top_k_addr)

    tfidf_addr_cands = defaultdict(set)
    for i, qid in enumerate(q_ids):
        tfidf_addr_cands[qid].update([cid for cid, _ in s2_addr_res[i]])
        tfidf_addr_cands[qid].update([cid for cid, _ in s3_addr_res[i]])
    eval_cands(tfidf_addr_cands, f"TF-IDF Addr Top-{top_k_addr} (S2+S3)")

    del cand_addrs, addr_vec, s2_addr_mat, s3_addr_mat, q_addr_mat
    gc.collect()

    # 5. Union of all channels
    log.info("--- Testing Union of Channels ---")
    union_cands = defaultdict(set)
    for qid in q_ids:
        union_cands[qid].update(exact_cands[qid])
        union_cands[qid].update(exact_sort_cands[qid])
        union_cands[qid].update(tfidf_name_cands[qid])
        union_cands[qid].update(tfidf_addr_cands[qid])
    pair_rec, comp_rec, mean_c = eval_cands(union_cands, "UNION (Exact + Suffix + Name@30 + Addr@25)")

    # 6. Missed Positives Analysis
    log.info("--- Performing Missed-Positives Analysis ---")
    missed_pairs = []
    for qid, t_set in sample_gt.items():
        if not t_set: continue
        retrieved = union_cands.get(qid, set())
        for tid in t_set:
            if tid not in retrieved:
                missed_pairs.append((qid, tid))

    log.info(f"Total Missed True Pairs: {len(missed_pairs)} out of {total_true_pairs} (Miss rate: {len(missed_pairs)/total_true_pairs*100:.2f}%)")

    # Sample and inspect 15 missed pairs
    s1_lookup = s1_sample.set_index('entity_id')
    cand_lookup = pd.concat([s2_df, s3_df]).set_index('entity_id')
    
    report_lines = [
        f"# Blocking Benchmark Report - {country}",
        f"- Sample S1 Queries: {len(s1_sample)}",
        f"- True Match Pairs: {total_true_pairs}",
        f"- Pair Recall: {pair_rec*100:.2f}%",
        f"- Complete Entity Recall: {comp_rec*100:.2f}%",
        f"- Average Candidates per S1: {mean_c:.1f}",
        f"- Total Missed Pairs: {len(missed_pairs)}",
        "",
        "## Sample Missed Pairs Inspection",
    ]
    for qid, tid in missed_pairs[:20]:
        q_row = s1_lookup.loc[qid] if qid in s1_lookup.index else None
        c_row = cand_lookup.loc[tid] if tid in cand_lookup.index else None
        if q_row is not None and c_row is not None:
            log.info(f"MISS: [{qid}] '{q_row['business_name']}' | '{q_row['business_address']}'")
            log.info(f"  VS  [{tid}] '{c_row['business_name']}' | '{c_row['business_address']}'")
            report_lines.append(f"- **Query [{qid}]**: `{q_row['business_name']}` | `{q_row['business_address']}`")
            report_lines.append(f"  - **Target [{tid}]**: `{c_row['business_name']}` | `{c_row['business_address']}`")

    report_path = REPORTS / f'blocking_report_{country}.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    log.info(f"Report written to {report_path}")

    return pair_rec, comp_rec, mean_c


if __name__ == '__main__':
    run_benchmark(country='India', n_queries=3000)
