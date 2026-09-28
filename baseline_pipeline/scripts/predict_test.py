"""
Production Test Inference Pipeline for Amazon ML Challenge 2026.
Processes test queries per country in memory-efficient batches.
Generates candidate_pairs.tsv and matching_results.tsv.
Validates outputs with the official validator.
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
import pickle
import subprocess
import logging
from collections import defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import unicodedata

from baseline_pipeline.scripts.train_model import (
    clean_text, strip_suffix, sorted_tokens, first_two_tokens,
    query_sparse_topk, compute_pair_features
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('predict_test')

CACHE = BASE / 'baseline_pipeline' / 'cache'
MODELS = BASE / 'baseline_pipeline' / 'models'
OUTPUT = BASE / 'baseline_pipeline' / 'output'
OUTPUT.mkdir(parents=True, exist_ok=True)
RES = BASE / 'resources-20260925T050740Z-1-001' / 'resources'
DATA = BASE / 'dataset-20260925T050739Z-1-001' / 'dataset'


import argparse

def main():
    parser = argparse.ArgumentParser(description='Test Inference')
    parser.add_argument('--sample', type=int, default=0, help='Number of S1 entities per country to sample (0 = all)')
    args = parser.parse_args()

    t_start = time.time()
    log.info("=" * 60)
    log.info("GEMINI SOLUTION — Final Test Inference Pipeline")
    log.info("=" * 60)

    # 1. Load trained models and config
    config_path = MODELS / 'matcher_config.pkl'
    if not config_path.exists():
        log.error("Trained models not found! Run train_model.py first.")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    threshold = config['best_threshold']
    feature_cols = config['feature_cols']
    log.info(f"Loaded config: optimal match threshold = {threshold:.2f}, features = {len(feature_cols)}")

    models = []
    for i in range(5):
        m_path = MODELS / f'lgb_model_fold{i}.txt'
        if m_path.exists():
            models.append(lgb.Booster(model_file=str(m_path)))
    log.info(f"Loaded {len(models)} LightGBM ensemble models")

    # 2. Load Test S1
    log.info("Loading test S1 entities...")
    test_s1_full = pd.read_parquet(CACHE / 'test_s1.parquet')
    log.info(f"Total Test S1 entities: {len(test_s1_full)}")

    countries = sorted(test_s1_full['country'].unique())
    log.info(f"Test countries ({len(countries)}): {countries}")

    all_matched = {}     # s1_id -> set of matched candidate IDs
    all_candidates = {}  # s1_id -> set of candidate IDs

    for country in countries:
        log.info(f"\n{'='*40}\nProcessing Country: {country}\n{'='*40}")
        s1_c = test_s1_full[test_s1_full['country'] == country].copy()
        if args.sample > 0:
            s1_c = s1_c.head(args.sample).copy()
        q_ids = s1_c['entity_id'].values
        log.info(f"Country {country}: {len(s1_c)} S1 entities")

        s1_c['name_clean'] = [clean_text(x) for x in s1_c['business_name']]
        s1_c['addr_clean'] = [clean_text(x) for x in s1_c['business_address']]
        s1_c['name_suf'] = [strip_suffix(x) for x in s1_c['name_clean']]
        s1_c['name_sort'] = [sorted_tokens(x) for x in s1_c['name_clean']]
        s1_c['name_f2'] = [first_two_tokens(x) for x in s1_c['name_clean']]

        # Load S2 and S3 for this country
        safe_country = country.replace(' ', '_')
        s2_path = CACHE / f'test_s2_{safe_country}.parquet'
        s3_path = CACHE / f'test_s3_{safe_country}.parquet'

        candidate_records = defaultdict(dict)

        def retrieve_test_source(source_num, df_path):
            if not df_path.exists():
                log.warning(f"File {df_path} does not exist, skipping")
                return
            log.info(f"Indexing & retrieving Test Source {source_num} for {country}...")
            src_df = pd.read_parquet(df_path)
            c_ids = src_df['entity_id'].values
            c_names = [clean_text(x) for x in src_df['business_name']]
            c_addrs = [clean_text(x) for x in src_df['business_address']]
            log.info(f"Source {source_num}: {len(src_df)} records")

            # Exact suffix
            suf_idx = defaultdict(list)
            for cid, name in zip(c_ids, c_names):
                suf = strip_suffix(name)
                if suf: suf_idx[suf].append(cid)
            for qid, suf in zip(q_ids, s1_c['name_suf']):
                if suf:
                    for cid in suf_idx.get(suf, [])[:150]:
                        candidate_records[(qid, cid)]['exact_suffix_flag'] = 1
            del suf_idx; gc.collect()

            # Exact sorted
            sort_idx = defaultdict(list)
            for cid, name in zip(c_ids, c_names):
                srt = sorted_tokens(name)
                if srt: sort_idx[srt].append(cid)
            for qid, srt in zip(q_ids, s1_c['name_sort']):
                if srt:
                    for cid in sort_idx.get(srt, [])[:150]:
                        candidate_records[(qid, cid)]['exact_sorted_flag'] = 1
            del sort_idx; gc.collect()

            # Word Name TF-IDF
            v_name = TfidfVectorizer(ngram_range=(1, 2), max_df=0.25, min_df=2, sublinear_tf=True, dtype=np.float32)
            c_n_mat = v_name.fit_transform(c_names)
            q_n_mat = v_name.transform(s1_c['name_clean'])
            del v_name; gc.collect()
            top_name = query_sparse_topk(q_n_mat, c_n_mat, c_ids, top_k=25, batch_size=500)
            for q_idx, cands in enumerate(top_name):
                qid = q_ids[q_idx]
                for cid, score in cands:
                    rec = candidate_records[(qid, cid)]
                    rec['ret_name_score'] = max(rec.get('ret_name_score', 0.0), score)
            del c_n_mat, q_n_mat, top_name; gc.collect()

            # Word Addr TF-IDF
            v_addr = TfidfVectorizer(ngram_range=(1, 2), max_df=0.25, min_df=2, sublinear_tf=True, dtype=np.float32)
            c_a_mat = v_addr.fit_transform(c_addrs)
            q_a_mat = v_addr.transform(s1_c['addr_clean'])
            del v_addr; gc.collect()
            top_addr = query_sparse_topk(q_a_mat, c_a_mat, c_ids, top_k=20, batch_size=500)
            for q_idx, cands in enumerate(top_addr):
                qid = q_ids[q_idx]
                for cid, score in cands:
                    rec = candidate_records[(qid, cid)]
                    rec['ret_addr_score'] = max(rec.get('ret_addr_score', 0.0), score)
            del c_a_mat, q_a_mat, top_addr; gc.collect()
            del src_df, c_ids, c_names, c_addrs; gc.collect()

        retrieve_test_source(2, s2_path)
        retrieve_test_source(3, s3_path)

        # Assemble candidate pairs for country
        pairs_list = []
        for (qid, cid), meta in candidate_records.items():
            pairs_list.append({
                'source1_entity_id': qid,
                'candidate_entity_id': cid,
                'ret_name_score': meta.get('ret_name_score', 0.0),
                'ret_addr_score': meta.get('ret_addr_score', 0.0),
                'ret_channel_count': len(meta),
                'exact_suffix_flag': meta.get('exact_suffix_flag', 0),
                'exact_sorted_flag': meta.get('exact_sorted_flag', 0),
            })
        del candidate_records; gc.collect()
        pairs_c = pd.DataFrame(pairs_list)
        del pairs_list; gc.collect()

        log.info(f"Total candidate pairs for {country}: {len(pairs_c)}")

        # Store candidate sets for candidate_pairs.tsv
        cand_grouped = defaultdict(set)
        for q, c in zip(pairs_c['source1_entity_id'], pairs_c['candidate_entity_id']):
            cand_grouped[q].add(c)
        for q in q_ids:
            all_candidates[q] = cand_grouped.get(q, set())

        # Load S2/S3 records needed for feature engineering
        needed_ids = set(pairs_c['candidate_entity_id'])
        s2_c = pd.read_parquet(s2_path) if s2_path.exists() else pd.DataFrame()
        s3_c = pd.read_parquet(s3_path) if s3_path.exists() else pd.DataFrame()
        c_sub = pd.concat([s2_c[s2_c['entity_id'].isin(needed_ids)],
                           s3_c[s3_c['entity_id'].isin(needed_ids)]], ignore_index=True)
        del s2_c, s3_c; gc.collect()

        c_sub['name_clean'] = [clean_text(x) for x in c_sub['business_name']]
        c_sub['addr_clean'] = [clean_text(x) for x in c_sub['business_address']]
        c_sub['name_suf'] = [strip_suffix(x) for x in c_sub['name_clean']]
        c_sub['name_sort'] = [sorted_tokens(x) for x in c_sub['name_clean']]

        # Score in chunks of 250,000 candidate pairs
        log.info("Computing features and predicting matches...")
        chunk_size = 250000
        for start_idx in range(0, len(pairs_c), chunk_size):
            end_idx = min(start_idx + chunk_size, len(pairs_c))
            chunk_pairs = pairs_c.iloc[start_idx:end_idx]

            feats = compute_pair_features(chunk_pairs, s1_c, c_sub)
            for col in feature_cols:
                if col not in feats.columns:
                    feats[col] = 0.0
            X_chunk = feats[feature_cols].values.astype(np.float32)
            del feats; gc.collect()

            # Ensemble prediction
            preds = np.zeros(len(X_chunk), dtype=np.float32)
            for m in models:
                preds += m.predict(X_chunk) / len(models)
            del X_chunk; gc.collect()

            # Extract matches above threshold
            for q, c, prob in zip(chunk_pairs['source1_entity_id'], chunk_pairs['candidate_entity_id'], preds):
                if prob >= threshold:
                    if q not in all_matched:
                        all_matched[q] = set()
                    all_matched[q].add(c)

            del chunk_pairs, preds; gc.collect()

        del pairs_c, c_sub, s1_c; gc.collect()

    # 3. Ensure ALL test S1 entities are present in results
    all_s1_test_ids = test_s1_full['entity_id'].values
    for s1_id in all_s1_test_ids:
        if s1_id not in all_matched:
            all_matched[s1_id] = set()
        if s1_id not in all_candidates:
            all_candidates[s1_id] = set()

    del test_s1_full; gc.collect()

    # 4. Write matching_results.tsv
    log.info("\nWriting matching_results.tsv...")
    out_matching = OUTPUT / 'matching_results.tsv'
    with open(out_matching, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(all_matched.keys()):
            m_set = all_matched[s1_id]
            m_str = ",".join(sorted(m_set)) if m_set else ""
            f.write(f"{s1_id}\t{m_str}\n")
    log.info(f"Written {out_matching}")

    # 5. Write candidate_pairs.tsv
    log.info("Writing candidate_pairs.tsv...")
    out_candidates = OUTPUT / 'candidate_pairs.tsv'
    with open(out_candidates, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(all_candidates.keys()):
            c_set = all_candidates[s1_id]
            c_str = ",".join(sorted(c_set)) if c_set else ""
            f.write(f"{s1_id}\t{c_str}\n")
    log.info(f"Written {out_candidates}")

    # Stats
    n_singletons = sum(1 for v in all_matched.values() if len(v) == 0)
    n_matched = len(all_matched) - n_singletons
    total_matches = sum(len(v) for v in all_matched.values())
    log.info(f"Summary: {len(all_matched)} Total S1 Entities | {n_matched} Matched Entities ({total_matches} pairs) | {n_singletons} Singletons ({n_singletons/len(all_matched)*100:.2f}%)")

    # 6. Run Official Validator
    log.info("\n" + "=" * 60)
    log.info("RUNNING OFFICIAL VALIDATOR")
    log.info("=" * 60)
    val_script = RES / 'utils' / 'validate_submission.py'
    test_dir = DATA / 'test'

    res = subprocess.run([
        sys.executable, str(val_script),
        '--matching', str(out_matching),
        '--candidate', str(out_candidates),
        '--test-dir', str(test_dir),
    ], capture_output=True, text=True, timeout=300)

    log.info("Validator Output:\n" + res.stdout)
    if res.stderr:
        log.warning("Validator Stderr:\n" + res.stderr)

    if res.returncode == 0:
        log.info("✓ OFFICIAL SUBMISSION VALIDATION PASSED!")
    else:
        log.error("✗ SUBMISSION VALIDATION FAILED!")

    log.info(f"Test inference pipeline completed in {(time.time()-t_start)/60:.1f} minutes!")

if __name__ == '__main__':
    main()
