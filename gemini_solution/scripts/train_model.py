"""
Train and validate the entity matching classifier with Grouped 5-Fold CV.
Computes engineered features, samples hard negatives, tunes LightGBM,
and optimizes threshold on OOF predictions to maximize official Macro F0.5.
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
import pickle
import logging
from collections import defaultdict
from typing import Dict, Set, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from rapidfuzz import fuzz, distance as rfuzz_distance
import lightgbm as lgb
import unicodedata

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('train_matcher')

CACHE = BASE / 'gemini_solution' / 'cache'
MODELS = BASE / 'gemini_solution' / 'models'
REPORTS = BASE / 'gemini_solution' / 'reports'
EXPDIR = BASE / 'gemini_solution' / 'experiments'
for d in [CACHE, MODELS, REPORTS, EXPDIR]:
    d.mkdir(parents=True, exist_ok=True)


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

def query_sparse_topk(query_mat, cand_mat, cand_ids, top_k=25, batch_size=500):
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

def compute_pair_features(pairs_df, s1_df, cand_df):
    """Compute rich pairwise features in memory-efficient vectorized batches."""
    s1_ix = s1_df.set_index('entity_id')
    c_ix = cand_df.set_index('entity_id')

    s1_ids = pairs_df['source1_entity_id'].values
    c_ids = pairs_df['candidate_entity_id'].values

    def get_str(ix, ids, col):
        return [str(ix.at[eid, col]) if eid in ix.index and pd.notna(ix.at[eid, col]) else '' for eid in ids]

    s1_names = get_str(s1_ix, s1_ids, 'name_clean')
    c_names = get_str(c_ix, c_ids, 'name_clean')
    s1_suf = get_str(s1_ix, s1_ids, 'name_suf')
    c_suf = get_str(c_ix, c_ids, 'name_suf')
    s1_sort = get_str(s1_ix, s1_ids, 'name_sort')
    c_sort = get_str(c_ix, c_ids, 'name_sort')
    s1_addrs = get_str(s1_ix, s1_ids, 'addr_clean')
    c_addrs = get_str(c_ix, c_ids, 'addr_clean')

    n = len(pairs_df)
    feat = pd.DataFrame(index=pairs_df.index)

    # 1. Exact string matches
    feat['name_exact_basic'] = (np.array(s1_names) == np.array(c_names)).astype(np.int8)
    feat['name_exact_suffix'] = (np.array(s1_suf) == np.array(c_suf)).astype(np.int8)
    feat['name_exact_sorted'] = (np.array(s1_sort) == np.array(c_sort)).astype(np.int8)
    feat['addr_exact'] = (np.array(s1_addrs) == np.array(c_addrs)).astype(np.int8)

    # 2. RapidFuzz Name Similarities
    n_ratio = np.empty(n, dtype=np.float32)
    n_part = np.empty(n, dtype=np.float32)
    n_tsort = np.empty(n, dtype=np.float32)
    n_tset = np.empty(n, dtype=np.float32)
    n_jw = np.empty(n, dtype=np.float32)
    n_lev = np.empty(n, dtype=np.float32)
    n_jac = np.empty(n, dtype=np.float32)

    for i in range(n):
        a, b = s1_names[i], c_names[i]
        if a and b:
            n_ratio[i] = fuzz.ratio(a, b) / 100.0
            n_part[i] = fuzz.partial_ratio(a, b) / 100.0
            n_tsort[i] = fuzz.token_sort_ratio(a, b) / 100.0
            n_tset[i] = fuzz.token_set_ratio(a, b) / 100.0
            n_jw[i] = rfuzz_distance.JaroWinkler.similarity(a, b)
            n_lev[i] = 1.0 - rfuzz_distance.Levenshtein.normalized_distance(a, b)
            sa, sb = set(a.split()), set(b.split())
            n_jac[i] = len(sa & sb) / max(len(sa | sb), 1)
        else:
            n_ratio[i] = n_part[i] = n_tsort[i] = n_tset[i] = n_jw[i] = n_lev[i] = n_jac[i] = 0.0

    feat['name_ratio'] = n_ratio
    feat['name_part'] = n_part
    feat['name_tsort'] = n_tsort
    feat['name_tset'] = n_tset
    feat['name_jw'] = n_jw
    feat['name_lev'] = n_lev
    feat['name_jac'] = n_jac

    # 3. Name Structural features
    s1_len = np.array([len(s) for s in s1_names], dtype=np.float32)
    c_len = np.array([len(s) for s in c_names], dtype=np.float32)
    max_l = np.maximum(s1_len, c_len)
    feat['name_len_ratio'] = np.where(max_l > 0, np.minimum(s1_len, c_len) / max_l, 0.0)
    feat['name_len_diff'] = np.abs(s1_len - c_len)
    feat['name_pre3_eq'] = np.array([1 if a and b and a[:3] == b[:3] else 0 for a, b in zip(s1_names, c_names)], dtype=np.int8)
    feat['name_ft_eq'] = np.array([1 if a and b and a.split()[:1] == b.split()[:1] else 0 for a, b in zip(s1_names, c_names)], dtype=np.int8)
    feat['name_lt_eq'] = np.array([1 if a and b and a.split()[-1:] == b.split()[-1:] else 0 for a, b in zip(s1_names, c_names)], dtype=np.int8)
    feat['name_contain'] = np.array([1 if a and b and (a in b or b in a) else 0 for a, b in zip(s1_names, c_names)], dtype=np.int8)

    # 4. RapidFuzz Address Similarities
    a_ratio = np.empty(n, dtype=np.float32)
    a_tset = np.empty(n, dtype=np.float32)
    a_jac = np.empty(n, dtype=np.float32)
    a_novlp = np.empty(n, dtype=np.float32)

    for i in range(n):
        a, b = s1_addrs[i], c_addrs[i]
        if a and b:
            a_ratio[i] = fuzz.ratio(a, b) / 100.0
            a_tset[i] = fuzz.token_set_ratio(a, b) / 100.0
            sa, sb = set(a.split()), set(b.split())
            a_jac[i] = len(sa & sb) / max(len(sa | sb), 1)
            na = set(re.findall(r'\d+', a))
            nb = set(re.findall(r'\d+', b))
            if not na and not nb:
                a_novlp[i] = 1.0
            elif not na or not nb:
                a_novlp[i] = 0.0
            else:
                a_novlp[i] = len(na & nb) / max(len(na | nb), 1)
        else:
            a_ratio[i] = a_tset[i] = a_jac[i] = 0.0
            a_novlp[i] = 0.0

    feat['addr_ratio'] = a_ratio
    feat['addr_tset'] = a_tset
    feat['addr_jac'] = a_jac
    feat['addr_num_ovlp'] = a_novlp

    a1_len = np.array([len(s) for s in s1_addrs], dtype=np.float32)
    a2_len = np.array([len(s) for s in c_addrs], dtype=np.float32)
    amax = np.maximum(a1_len, a2_len)
    feat['addr_len_ratio'] = np.where(amax > 0, np.minimum(a1_len, a2_len) / amax, 0.0)
    feat['addr_miss_s1'] = (a1_len == 0).astype(np.int8)
    feat['addr_miss_cand'] = (a2_len == 0).astype(np.int8)
    feat['addr_either_miss'] = ((a1_len == 0) | (a2_len == 0)).astype(np.int8)

    # 5. Cross-field interactions
    feat['name_x_addr'] = n_ratio * a_ratio
    feat['name_addr_min'] = np.minimum(n_ratio, a_ratio)
    feat['name_addr_max'] = np.maximum(n_ratio, a_ratio)
    feat['hi_name_lo_addr'] = ((n_ratio > 0.8) & (a_ratio < 0.3)).astype(np.int8)
    feat['hi_addr_lo_name'] = ((a_ratio > 0.8) & (n_ratio < 0.3)).astype(np.int8)
    feat['both_strong'] = ((n_ratio > 0.7) & (a_ratio > 0.7)).astype(np.int8)
    feat['source_is_s3'] = np.array([1 if str(c).startswith('S3') else 0 for c in c_ids], dtype=np.int8)

    # 6. Retrieval Provenance features
    for col in ['ret_name_score', 'ret_addr_score', 'ret_channel_count', 'exact_suffix_flag', 'exact_sorted_flag']:
        if col in pairs_df.columns:
            feat[col] = pairs_df[col].values

    return feat


def evaluate_f05_macro(pred_dict: Dict[str, Set[str]], gt_dict: Dict[str, Set[str]]):
    """Compute official macro-averaged F0.5 per S1 entity."""
    scores = []
    precs, recs = [], []
    sing_ok, sing_tot = 0, 0
    complete = 0
    non_sing_tot = 0

    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if len(true_set) == 0:
            sing_tot += 1
            if len(pred_set) == 0:
                scores.append(1.0)
                sing_ok += 1
            else:
                scores.append(0.0)
        else:
            non_sing_tot += 1
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
            scores.append(f)
            precs.append(p)
            recs.append(r)
            if fp == 0 and fn == 0:
                complete += 1

    macro = float(np.mean(scores)) if scores else 0.0
    return macro, {
        'macro_f05': macro,
        'macro_precision': float(np.mean(precs)) if precs else 0.0,
        'macro_recall': float(np.mean(recs)) if recs else 0.0,
        'singleton_accuracy': sing_ok / max(sing_tot, 1),
        'complete_match_rate': complete / max(non_sing_tot, 1),
        'singletons': sing_tot,
        'entities_with_matches': non_sing_tot,
    }


def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("GEMINI SOLUTION — Model Training & Grouped 5-Fold CV")
    log.info("=" * 60)

    # 1. Load S1 training queries (e.g. 10,000 for training & CV)
    N_TRAIN = 10000
    log.info(f"Loading {N_TRAIN} S1 training entities (India)...")
    s1_full = pd.read_parquet(CACHE / 'train_s1_India.parquet').head(N_TRAIN)
    q_ids = s1_full['entity_id'].values
    q_set = set(q_ids)

    s1_full['name_clean'] = [clean_text(x) for x in s1_full['business_name']]
    s1_full['addr_clean'] = [clean_text(x) for x in s1_full['business_address']]
    s1_full['name_suf'] = [strip_suffix(x) for x in s1_full['name_clean']]
    s1_full['name_sort'] = [sorted_tokens(x) for x in s1_full['name_clean']]
    s1_full['name_f2'] = [first_two_tokens(x) for x in s1_full['name_clean']]

    # 2. Load ground truth for these entities
    gt_df = pd.read_parquet(CACHE / 'train_ground_truth.parquet')
    mask = gt_df['source1_entity_id'].isin(q_set)
    gt_sub = gt_df[mask]
    gt_dict = {row['source1_entity_id']: set(row['matched_entity_ids'].split(',')) if row['matched_entity_ids'] else set()
               for _, row in gt_sub.iterrows()}
    del gt_df, gt_sub
    gc.collect()

    n_sing = sum(1 for v in gt_dict.values() if len(v) == 0)
    total_true = sum(len(v) for v in gt_dict.values())
    log.info(f"Training GT: {len(gt_dict)} entities, {total_true} true pairs, {n_sing} singletons")

    # 3. Multi-Channel Candidate Retrieval
    candidate_records = defaultdict(dict)  # (qid, cid) -> metadata

    def retrieve_from_source(source_num, df_path):
        log.info(f"Retrieving from Source {source_num} India...")
        src_df = pd.read_parquet(df_path)
        c_ids = src_df['entity_id'].values
        c_names = [clean_text(x) for x in src_df['business_name']]
        c_addrs = [clean_text(x) for x in src_df['business_address']]

        # Exact suffix
        suf_idx = defaultdict(list)
        for cid, name in zip(c_ids, c_names):
            suf = strip_suffix(name)
            if suf: suf_idx[suf].append(cid)
        for qid, suf in zip(q_ids, s1_full['name_suf']):
            if suf:
                for cid in suf_idx.get(suf, [])[:150]:
                    rec = candidate_records[(qid, cid)]
                    rec['exact_suffix_flag'] = 1
                    rec['ret_channel_count'] = rec.get('ret_channel_count', 0) + 1
        del suf_idx; gc.collect()

        # Exact sorted
        sort_idx = defaultdict(list)
        for cid, name in zip(c_ids, c_names):
            srt = sorted_tokens(name)
            if srt: sort_idx[srt].append(cid)
        for qid, srt in zip(q_ids, s1_full['name_sort']):
            if srt:
                for cid in sort_idx.get(srt, [])[:150]:
                    rec = candidate_records[(qid, cid)]
                    rec['exact_sorted_flag'] = 1
                    rec['ret_channel_count'] = rec.get('ret_channel_count', 0) + 1
        del sort_idx; gc.collect()

        # Word-level Name TF-IDF
        v_name = TfidfVectorizer(ngram_range=(1, 2), max_df=0.25, min_df=2, sublinear_tf=True, dtype=np.float32)
        c_n_mat = v_name.fit_transform(c_names)
        q_n_mat = v_name.transform(s1_full['name_clean'])
        del v_name; gc.collect()
        top_name = query_sparse_topk(q_n_mat, c_n_mat, c_ids, top_k=25, batch_size=500)
        for q_idx, cands in enumerate(top_name):
            qid = q_ids[q_idx]
            for cid, score in cands:
                rec = candidate_records[(qid, cid)]
                rec['ret_name_score'] = max(rec.get('ret_name_score', 0.0), score)
                rec['ret_channel_count'] = rec.get('ret_channel_count', 0) + 1
        del c_n_mat, q_n_mat, top_name; gc.collect()

        # Word-level Address TF-IDF
        v_addr = TfidfVectorizer(ngram_range=(1, 2), max_df=0.25, min_df=2, sublinear_tf=True, dtype=np.float32)
        c_a_mat = v_addr.fit_transform(c_addrs)
        q_a_mat = v_addr.transform(s1_full['addr_clean'])
        del v_addr; gc.collect()
        top_addr = query_sparse_topk(q_a_mat, c_a_mat, c_ids, top_k=20, batch_size=500)
        for q_idx, cands in enumerate(top_addr):
            qid = q_ids[q_idx]
            for cid, score in cands:
                rec = candidate_records[(qid, cid)]
                rec['ret_addr_score'] = max(rec.get('ret_addr_score', 0.0), score)
                rec['ret_channel_count'] = rec.get('ret_channel_count', 0) + 1
        del c_a_mat, q_a_mat, top_addr; gc.collect()
        del src_df, c_ids, c_names, c_addrs; gc.collect()

    retrieve_from_source(2, CACHE / 'train_s2_India.parquet')
    retrieve_from_source(3, CACHE / 'train_s3_India.parquet')

    # Build pairs DataFrame
    pairs_list = []
    for (qid, cid), meta in candidate_records.items():
        pairs_list.append({
            'source1_entity_id': qid,
            'candidate_entity_id': cid,
            'ret_name_score': meta.get('ret_name_score', 0.0),
            'ret_addr_score': meta.get('ret_addr_score', 0.0),
            'ret_channel_count': meta.get('ret_channel_count', 0),
            'exact_suffix_flag': meta.get('exact_suffix_flag', 0),
            'exact_sorted_flag': meta.get('exact_sorted_flag', 0),
        })
    del candidate_records; gc.collect()
    pairs_df = pd.DataFrame(pairs_list)
    del pairs_list; gc.collect()
    log.info(f"Generated {len(pairs_df)} candidate pairs for {len(q_ids)} queries ({len(pairs_df)/len(q_ids):.1f} per query)")

    # Label candidates
    labels = np.array([1 if c in gt_dict.get(q, set()) else 0 
                       for q, c in zip(pairs_df['source1_entity_id'], pairs_df['candidate_entity_id'])], dtype=np.int8)
    pairs_df['label'] = labels
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    log.info(f"Candidates: {n_pos} Positive pairs ({n_pos/max(total_true,1)*100:.2f}% recall), {n_neg} Negative pairs")

    # Hard Negative Sampling (keep ALL positives, sample 10x hard negatives by retrieval score)
    pos_df = pairs_df[pairs_df['label'] == 1]
    neg_df = pairs_df[pairs_df['label'] == 0].copy()
    neg_df['ret_max'] = np.maximum(neg_df['ret_name_score'], neg_df['ret_addr_score'])
    n_sample_neg = min(len(neg_df), n_pos * 10)
    hard_neg_df = neg_df.nlargest(n_sample_neg, 'ret_max').drop(columns=['ret_max'])
    train_pairs = pd.concat([pos_df, hard_neg_df], ignore_index=True)
    del pairs_df, pos_df, neg_df, hard_neg_df; gc.collect()
    log.info(f"Training dataset: {len(train_pairs)} pairs ({n_pos} positive, {n_sample_neg} hard negative)")

    # Load candidate records for feature calculation
    needed_cand_ids = set(train_pairs['candidate_entity_id'])
    s2_full = pd.read_parquet(CACHE / 'train_s2_India.parquet')
    s3_full = pd.read_parquet(CACHE / 'train_s3_India.parquet')
    c_full = pd.concat([s2_full[s2_full['entity_id'].isin(needed_cand_ids)],
                        s3_full[s3_full['entity_id'].isin(needed_cand_ids)]], ignore_index=True)
    del s2_full, s3_full; gc.collect()
    c_full['name_clean'] = [clean_text(x) for x in c_full['business_name']]
    c_full['addr_clean'] = [clean_text(x) for x in c_full['business_address']]
    c_full['name_suf'] = [strip_suffix(x) for x in c_full['name_clean']]
    c_full['name_sort'] = [sorted_tokens(x) for x in c_full['name_clean']]

    # Compute features
    log.info("Computing feature matrix...")
    features_df = compute_pair_features(train_pairs, s1_full, c_full)
    del c_full, s1_full; gc.collect()

    feat_cols = [c for c in features_df.columns]
    X = features_df.values.astype(np.float32)
    y = train_pairs['label'].values.astype(np.int8)
    groups = train_pairs['source1_entity_id'].values
    log.info(f"Feature matrix: shape {X.shape}, features: {len(feat_cols)}")

    # 4. LightGBM Grouped 5-Fold Cross-Validation
    log.info("--- Starting Grouped 5-Fold CV ---")
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    models = []
    feature_importances = np.zeros(len(feat_cols), dtype=np.float32)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 45,
        'max_depth': 7,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)
        model = lgb.train(
            lgb_params, dtrain, num_boost_round=600,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)]
        )
        val_pred = model.predict(X[val_idx])
        oof_preds[val_idx] = val_pred
        models.append(model)
        feature_importances += model.feature_importance(importance_type='gain') / 5.0
        log.info(f"Fold {fold+1}/5 trained (best iteration: {model.best_iteration})")

    # 5. Out-Of-Fold Threshold Sweep to Maximize Macro F0.5
    log.info("--- Sweeping Threshold on OOF Predictions ---")
    train_pairs['pred_prob'] = oof_preds

    # Group candidate predictions by S1 entity
    cand_by_s1 = defaultdict(list)
    for qid, cid, prob in zip(train_pairs['source1_entity_id'], train_pairs['candidate_entity_id'], oof_preds):
        cand_by_s1[qid].append((cid, prob))

    best_thresh = 0.5
    best_f05 = 0.0
    best_metrics = {}

    threshold_results = []
    for thresh in np.arange(0.35, 0.95, 0.02):
        pred_dict = {}
        for qid in q_ids:
            cands = cand_by_s1.get(qid, [])
            pred_dict[qid] = {cid for cid, prob in cands if prob >= thresh}
        f05, m = evaluate_f05_macro(pred_dict, gt_dict)
        threshold_results.append((thresh, f05, m['macro_precision'], m['macro_recall'], m['singleton_accuracy']))
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(thresh)
            best_metrics = m

    log.info("\n" + "=" * 60)
    log.info("OOF VALIDATION RESULTS (OFFICIAL MACRO F0.5)")
    log.info("=" * 60)
    log.info(f"Best Match Threshold:      {best_thresh:.2f}")
    log.info(f"Macro F0.5 Score:          {best_f05:.4f}")
    log.info(f"Macro Precision:           {best_metrics['macro_precision']:.4f}")
    log.info(f"Macro Recall:              {best_metrics['macro_recall']:.4f}")
    log.info(f"Singleton Accuracy:        {best_metrics['singleton_accuracy']:.4f}")
    log.info(f"Complete Match Recovery:   {best_metrics['complete_match_rate']:.4f}")
    log.info("=" * 60)

    # 6. Save Models & Config
    log.info("Saving trained models and config...")
    for i, m in enumerate(models):
        m.save_model(str(MODELS / f'lgb_model_fold{i}.txt'))
    with open(MODELS / 'matcher_config.pkl', 'wb') as f:
        pickle.dump({
            'feature_cols': feat_cols,
            'best_threshold': best_thresh,
            'lgb_params': lgb_params,
            'oof_macro_f05': best_f05,
            'oof_metrics': best_metrics,
        }, f)

    # 7. Write Model Report & Experiments CSV
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': feature_importances}).sort_values('importance', ascending=False)
    log.info("\nTop 15 Most Informative Features:")
    for _, r in imp_df.head(15).iterrows():
        log.info(f"  {r['feature']:<20}: {r['importance']:.1f}")

    report_lines = [
        "# Gemini Solution — Matching Model Report",
        "",
        "## 1. Validation Performance",
        f"- **Model**: LightGBM GBDT (Grouped 5-Fold Cross-Validation by Source 1 Entity)",
        f"- **Evaluation Metric**: Official Macro F0.5 (per S1 entity)",
        f"- **Best Threshold**: `{best_thresh:.2f}`",
        f"- **Macro F0.5**: **`{best_f05:.4f}`**",
        f"- **Macro Precision**: **`{best_metrics['macro_precision']:.4f}`**",
        f"- **Macro Recall**: **`{best_metrics['macro_recall']:.4f}`**",
        f"- **Singleton Accuracy**: **`{best_metrics['singleton_accuracy']:.4f}`**",
        f"- **Complete Match Recovery**: **`{best_metrics['complete_match_rate']:.4f}`**",
        "",
        "## 2. Threshold Calibration Sweep",
        "| Threshold | Macro F0.5 | Precision | Recall | Singleton Accuracy |",
        "|---|---|---|---|---|",
    ]
    for t, f05, p, r, s in threshold_results:
        report_lines.append(f"| {t:.2f} | {f05:.4f} | {p:.4f} | {r:.4f} | {s:.4f} |")

    report_lines.extend([
        "",
        "## 3. Top Feature Importances",
        "| Rank | Feature | Importance (Gain) |",
        "|---|---|---|",
    ])
    for rank, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        report_lines.append(f"| {rank} | `{row['feature']}` | {row['importance']:.1f} |")

    with open(REPORTS / 'model_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Log to experiments.csv
    exp_file = EXPDIR / 'experiments.csv'
    exp_entry = pd.DataFrame([{
        'experiment_id': 'exp_lgb_v1',
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'blocker_version': 'v1_word_tfidf_exact',
        'feature_version': f'{len(feat_cols)}_features',
        'model': 'LightGBM_5Fold_GBDT',
        'negative_strategy': '10x_hard_retrieval',
        'threshold': best_thresh,
        'singleton_strategy': 'optimal_f05_threshold',
        'pair_recall': n_pos / max(total_true, 1),
        'complete_entity_recall': best_metrics['complete_match_rate'],
        'macro_f0_5': best_f05,
        'macro_precision': best_metrics['macro_precision'],
        'macro_recall': best_metrics['macro_recall'],
        'singleton_accuracy': best_metrics['singleton_accuracy'],
        'candidate_mean': len(train_pairs) / len(q_ids),
        'runtime': f'{(time.time()-t_start)/60:.1f}min',
        'notes': f'5-fold grouped CV on {len(q_ids)} S1 training queries',
    }])
    if exp_file.exists():
        existing = pd.read_csv(str(exp_file))
        exp_entry = pd.concat([existing, exp_entry], ignore_index=True)
    exp_entry.to_csv(str(exp_file), index=False)
    log.info(f"Model report written to {REPORTS / 'model_report.md'}")
    log.info(f"Experiment logged to {exp_file}")
    log.info(f"Total training pipeline completed in {(time.time()-t_start)/60:.1f} minutes!")

if __name__ == '__main__':
    main()
