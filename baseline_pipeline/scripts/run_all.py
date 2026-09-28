#!/usr/bin/env python3
"""
Gemini Solution — Main Pipeline (Parquet-based, per-country)
============================================================

Prerequisite: Run 00_prepare_parquet.py first.

Usage:
    python baseline_pipeline/scripts/run_all.py [--sample N] [--skip-test]
"""
import sys, os, gc, time, argparse, warnings, re, pickle, logging, subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from rapidfuzz import fuzz, distance as rfuzz_distance
from tqdm import tqdm
import lightgbm as lgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('gemini')

# ── Paths ──
BASE   = Path('d:/RESOURCES/AmazonMLChallenge')
DATA   = BASE / 'dataset-20260925T050739Z-1-001' / 'dataset'
RES    = BASE / 'resources-20260925T050740Z-1-001' / 'resources'
SOL    = BASE / 'baseline_pipeline'
CACHE  = SOL / 'cache'
MODELS = SOL / 'models'
OUTPUT = SOL / 'output'
REPORTS= SOL / 'reports'
EXPDIR = SOL / 'experiments'
for d in [CACHE, MODELS, OUTPUT, REPORTS, EXPDIR, SOL/'artifacts']:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════

LEGAL_SUFFIXES = sorted([
    'inc', 'incorporated', 'llc', 'llp', 'ltd', 'limited', 'pvt', 'private',
    'corp', 'corporation', 'co', 'company', 'enterprises', 'enterprise',
    'group', 'holdings', 'holding', 'international', 'intl', 'services',
    'service', 'solutions', 'solution', 'technologies', 'technology', 'tech',
    'industries', 'industry', 'associates', 'association', 'assoc',
    'foundation', 'fund', 'trust', 'partners', 'partnership',
    'sa', 'sarl', 'sas', 'eurl', 'gmbh', 'ag', 'pty', 'nv', 'bv', 'plc', 'lp', 'na',
], key=len, reverse=True)

_SUFFIX_RE = re.compile(
    r'[\s,]*\b(?:' + '|'.join(LEGAL_SUFFIXES) + r')\b\.?[\s,]*$', re.IGNORECASE
)
_PAREN_RE = re.compile(r'\s*\([^)]*\)\s*$')

def _norm_name(s):
    if not s: return ''
    s = unicodedata.normalize('NFKC', s).lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _strip_suffix(s):
    prev = None
    while s != prev:
        prev = s
        s = _SUFFIX_RE.sub('', s).strip()
    s = _PAREN_RE.sub('', s).strip()
    return s

def _norm_addr(s):
    if not s: return ''
    s = unicodedata.normalize('NFKC', s).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _sorted_toks(s):
    t = s.split()
    return ' '.join(sorted(t)) if t else ''

def _num_toks(s):
    return ' '.join(re.findall(r'\d+', s))

def normalize_df(df):
    df = df.copy()
    n = df['business_name'].fillna('').astype(str)
    a = df['business_address'].fillna('').astype(str)
    df['name_basic']  = n.map(_norm_name)
    df['name_suffix'] = df['name_basic'].map(_strip_suffix)
    df['name_sorted'] = df['name_basic'].map(_sorted_toks)
    df['name_numeric']= df['name_basic'].map(_num_toks)
    df['addr_basic']  = a.map(_norm_addr)
    df['addr_numeric']= df['addr_basic'].map(_num_toks)
    df['combined']    = df['name_basic'] + ' ' + df['addr_basic']
    return df

# ═══════════════════════════════════════════════════════════════
# BLOCKING
# ═══════════════════════════════════════════════════════════════

def tfidf_topk(q_texts, q_ids, c_texts, c_ids, c_src,
               top_k=30, ngram_range=(3,5), max_features=300000, batch_sz=5000):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range,
                          max_features=max_features, min_df=2, max_df=0.95,
                          sublinear_tf=True, dtype=np.float32)
    c_mat = vec.fit_transform(c_texts)
    
    src_data = {}
    for src in np.unique(c_src):
        mask = c_src == src
        src_data[src] = (c_mat[mask], c_ids[mask])
    
    results = {}
    for start in range(0, len(q_texts), batch_sz):
        end = min(start + batch_sz, len(q_texts))
        q_mat = vec.transform(q_texts[start:end])
        for src, (cm, ci) in src_data.items():
            sim = q_mat.dot(cm.T)
            for i in range(sim.shape[0]):
                qid = q_ids[start + i]
                row = sim.getrow(i)
                if row.nnz == 0:
                    results.setdefault(qid, [])
                    continue
                d, ix = row.data, row.indices
                k = min(top_k, len(d))
                if k < len(d):
                    pi = np.argpartition(d, -k)[-k:]
                else:
                    pi = np.arange(len(d))
                pi = pi[np.argsort(d[pi])[::-1]]
                results.setdefault(qid, []).extend(
                    [(ci[ix[j]], float(d[j])) for j in pi if d[j] > 0])
            del sim
        del q_mat
        gc.collect()
    del c_mat, vec, src_data
    gc.collect()
    return results

def exact_block(q_keys, q_ids, c_keys, c_ids, fanout=200):
    idx = defaultdict(list)
    for k, cid in zip(c_keys, c_ids):
        k = str(k).strip()
        if k: idx[k].append(cid)
    results = {}
    for k, qid in zip(q_keys, q_ids):
        k = str(k).strip()
        m = idx.get(k, [])
        results[qid] = [(c, 1.0) for c in m] if k and 0 < len(m) <= fanout else []
    return results

def block_country(s1, cand, name_k=30, addr_k=20, comb_k=20):
    if len(s1) == 0 or len(cand) == 0:
        return pd.DataFrame(columns=['source1_entity_id','candidate_entity_id'])
    
    qi = s1['entity_id'].values
    ci = cand['entity_id'].values
    cs = np.array([e[:2] for e in ci])
    
    log.info(f"    Blocking {len(qi)} S1 x {len(ci)} cands")
    
    ch = {}
    log.info("    Ch1: Name TF-IDF")
    ch['name'] = tfidf_topk(s1['name_basic'].values, qi,
                             cand['name_basic'].values, ci, cs, top_k=name_k)
    log.info("    Ch2: Addr TF-IDF")
    ch['addr'] = tfidf_topk(s1['addr_basic'].values, qi,
                             cand['addr_basic'].values, ci, cs, top_k=addr_k)
    log.info("    Ch3: Combined TF-IDF")
    ch['comb'] = tfidf_topk(s1['combined'].values, qi,
                             cand['combined'].values, ci, cs, top_k=comb_k)
    log.info("    Ch4: Exact suffix")
    ch['ex_suf'] = exact_block(s1['name_suffix'].values, qi,
                                cand['name_suffix'].values, ci)
    log.info("    Ch5: Exact sorted")
    ch['ex_sort'] = exact_block(s1['name_sorted'].values, qi,
                                 cand['name_sorted'].values, ci)
    
    # Merge
    pairs = {}
    for cname, cdata in ch.items():
        for qid, cands in cdata.items():
            for cid, score in cands:
                key = (qid, cid)
                if key not in pairs:
                    pairs[key] = {'ns':0,'as':0,'cs':0,'esf':0,'est':0,'cc':0}
                p = pairs[key]
                p['cc'] += 1
                if cname=='name': p['ns'] = max(p['ns'], score)
                elif cname=='addr': p['as'] = max(p['as'], score)
                elif cname=='comb': p['cs'] = max(p['cs'], score)
                elif cname=='ex_suf': p['esf'] = 1
                elif cname=='ex_sort': p['est'] = 1
    del ch
    gc.collect()
    
    if not pairs:
        return pd.DataFrame(columns=['source1_entity_id','candidate_entity_id'])
    
    rows = [{'source1_entity_id':k[0], 'candidate_entity_id':k[1],
             'name_score':v['ns'], 'addr_score':v['as'], 'combined_score':v['cs'],
             'exact_suffix_flag':v['esf'], 'exact_sorted_flag':v['est'],
             'channel_count':v['cc']} for k,v in pairs.items()]
    del pairs
    gc.collect()
    
    df = pd.DataFrame(rows)
    for c in ['name_score','addr_score','combined_score']:
        df[c] = df[c].astype(np.float32)
    df['exact_suffix_flag'] = df['exact_suffix_flag'].astype(np.int8)
    df['exact_sorted_flag'] = df['exact_sorted_flag'].astype(np.int8)
    df['channel_count'] = df['channel_count'].astype(np.int8)
    
    log.info(f"    -> {len(df)} pairs ({len(df)/max(len(s1),1):.1f} avg/S1)")
    return df

# ═══════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════

def _fb(a, b, fn):
    n = len(a)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = fn(a[i], b[i]) / 100.0 if a[i] and b[i] else 0.0
    return out

def _db(a, b, fn):
    n = len(a)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = fn(a[i], b[i]) if a[i] and b[i] else 0.0
    return out

def _jac(a, b):
    n = len(a)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sa, sb = set(str(a[i]).split()), set(str(b[i]).split())
        out[i] = len(sa & sb) / max(len(sa | sb), 1)
    return out

def _novlp(a, b):
    n = len(a)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        na = set(re.findall(r'\d+', str(a[i])))
        nb = set(re.findall(r'\d+', str(b[i])))
        if not na and not nb: out[i] = 1.0
        elif not na or not nb: out[i] = 0.0
        else: out[i] = len(na & nb) / max(len(na | nb), 1)
    return out

def compute_features(pairs, s1_df, cand_df):
    s1_ix = s1_df.set_index('entity_id')
    c_ix = cand_df.set_index('entity_id')
    
    s1_ids = pairs['source1_entity_id'].values
    c_ids = pairs['candidate_entity_id'].values
    
    def gf(ix, ids, field):
        vals = []
        for eid in ids:
            try: v = ix.at[eid, field]; vals.append(str(v) if v else '')
            except: vals.append('')
        return np.array(vals, dtype=object)
    
    sn = gf(s1_ix, s1_ids, 'name_basic')
    cn = gf(c_ix, c_ids, 'name_basic')
    ss = gf(s1_ix, s1_ids, 'name_suffix')
    cs = gf(c_ix, c_ids, 'name_suffix')
    sst = gf(s1_ix, s1_ids, 'name_sorted')
    cst = gf(c_ix, c_ids, 'name_sorted')
    sa = gf(s1_ix, s1_ids, 'addr_basic')
    ca = gf(c_ix, c_ids, 'addr_basic')
    
    f = pd.DataFrame(index=pairs.index)
    
    # Name features
    f['n_eq_basic'] = (sn == cn).astype(np.int8)
    f['n_eq_suf'] = (ss == cs).astype(np.int8)
    f['n_eq_sort'] = (sst == cst).astype(np.int8)
    f['n_ratio'] = _fb(sn, cn, fuzz.ratio)
    f['n_partial'] = _fb(sn, cn, fuzz.partial_ratio)
    f['n_tsort'] = _fb(sn, cn, fuzz.token_sort_ratio)
    f['n_tset'] = _fb(sn, cn, fuzz.token_set_ratio)
    f['n_jw'] = _db(sn, cn, rfuzz_distance.JaroWinkler.similarity)
    f['n_lev'] = 1.0 - _db(sn, cn, rfuzz_distance.Levenshtein.normalized_distance)
    f['n_jac'] = _jac(sn, cn)
    f['n_novlp'] = _novlp(sn, cn)
    
    slen = np.array([len(s) for s in sn])
    clen = np.array([len(s) for s in cn])
    mx = np.maximum(slen, clen).astype(np.float32)
    f['n_lenr'] = np.where(mx > 0, np.minimum(slen, clen) / mx, 0)
    f['n_pre3'] = np.array([1 if a and b and a[:3]==b[:3] else 0 for a,b in zip(sn,cn)], np.int8)
    f['n_ft_eq'] = np.array([1 if a and b and a.split()[0]==b.split()[0] else 0 for a,b in zip(sn,cn)], np.int8)
    f['n_lt_eq'] = np.array([1 if a and b and a.split()[-1]==b.split()[-1] else 0 for a,b in zip(sn,cn)], np.int8)
    f['n_tdiff'] = np.abs(np.array([len(s.split()) for s in sn]) - np.array([len(s.split()) for s in cn])).astype(np.int8)
    f['n_contain'] = np.array([1 if a and b and (a in b or b in a) else 0 for a,b in zip(sn,cn)], np.int8)
    
    # Address features
    f['a_eq'] = (sa == ca).astype(np.int8)
    f['a_ratio'] = _fb(sa, ca, fuzz.ratio)
    f['a_tset'] = _fb(sa, ca, fuzz.token_set_ratio)
    f['a_jac'] = _jac(sa, ca)
    f['a_novlp'] = _novlp(sa, ca)
    
    alen1 = np.array([len(s) for s in sa])
    alen2 = np.array([len(s) for s in ca])
    amx = np.maximum(alen1, alen2).astype(np.float32)
    f['a_lenr'] = np.where(amx > 0, np.minimum(alen1, alen2) / amx, 0)
    f['a_miss1'] = (alen1 == 0).astype(np.int8)
    f['a_miss2'] = (alen2 == 0).astype(np.int8)
    f['a_eith_miss'] = ((alen1==0)|(alen2==0)).astype(np.int8)
    
    f['a_tok_ovlp'] = np.array([
        len(set(a.split())&set(b.split()))/max(min(len(a.split()),len(b.split())),1)
        if a and b else 0.0 for a,b in zip(sa,ca)], np.float32)
    
    def _fn(s):
        m = re.search(r'\d+', str(s))
        return m.group() if m else ''
    def _ln(s):
        ns = re.findall(r'\d+', str(s))
        return ns[-1] if ns else ''
    f['a_fn_eq'] = np.array([1 if _fn(a)==_fn(b) and _fn(a) else 0 for a,b in zip(sa,ca)], np.int8)
    f['a_ln_eq'] = np.array([1 if _ln(a)==_ln(b) and _ln(a) else 0 for a,b in zip(sa,ca)], np.int8)
    
    f['a_long_jac'] = np.array([
        len(set(t for t in a.split() if len(t)>=4) & set(t for t in b.split() if len(t)>=4)) /
        max(len(set(t for t in a.split() if len(t)>=4) | set(t for t in b.split() if len(t)>=4)), 1)
        if a and b else 0.0 for a,b in zip(sa,ca)], np.float32)
    
    # Retrieval features
    for c in ['name_score','addr_score','combined_score','exact_suffix_flag','exact_sorted_flag','channel_count']:
        if c in pairs.columns:
            f['r_'+c] = pairs[c].values
    
    # Cross features
    nr = f['n_ratio'].values
    ar = f['a_ratio'].values
    f['x_prod'] = nr * ar
    f['x_min'] = np.minimum(nr, ar)
    f['x_max'] = np.maximum(nr, ar)
    f['x_hn_la'] = ((nr > 0.8) & (ar < 0.3)).astype(np.int8)
    f['x_ha_ln'] = ((ar > 0.8) & (nr < 0.3)).astype(np.int8)
    f['x_both'] = ((nr > 0.7) & (ar > 0.7)).astype(np.int8)
    eps = 1e-6
    f['x_hmean'] = 2*nr*ar / (nr + ar + eps)
    f['src_type'] = np.array([1 if str(c).startswith('S3') else 0 for c in c_ids], np.int8)
    
    return f

# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def macro_f05(pred: Dict[str, Set[str]], gt: Dict[str, Set[str]]):
    scores = []
    precs, recs = [], []
    s_ok, s_tot = 0, 0
    comp = 0
    ns = 0
    for s1, ts in gt.items():
        ps = pred.get(s1, set())
        if len(ts) == 0:
            s_tot += 1
            if len(ps) == 0: scores.append(1.0); s_ok += 1
            else: scores.append(0.0)
        else:
            ns += 1
            tp = len(ps & ts)
            fp = len(ps - ts)
            fn = len(ts - ps)
            p = tp/(tp+fp) if tp+fp>0 else 0.0
            r = tp/(tp+fn) if tp+fn>0 else 0.0
            f = (1.25*p*r)/(0.25*p+r) if p+r>0 else 0.0
            scores.append(f); precs.append(p); recs.append(r)
            if fp==0 and fn==0: comp += 1
    mf = float(np.mean(scores)) if scores else 0.0
    return mf, {
        'macro_f05': mf,
        'macro_precision': float(np.mean(precs)) if precs else 0.0,
        'macro_recall': float(np.mean(recs)) if recs else 0.0,
        'singleton_accuracy': s_ok/max(s_tot,1),
        'complete_match_rate': comp/max(ns,1),
        'singletons': s_tot, 'non_singletons': ns,
    }

def apply_thresh(df, t, all_s1=None):
    pred = {}
    above = df[df['pred_score'] >= t]
    for s1, grp in above.groupby('source1_entity_id'):
        pred[s1] = set(grp['candidate_entity_id'].tolist())
    if all_s1:
        for s1 in all_s1:
            pred.setdefault(s1, set())
    return pred

def sweep_thresh(df, gt, lo=0.30, hi=0.95, step=0.01):
    best_t, best_f, best_d = 0.5, 0.0, {}
    all_s1 = set(gt.keys())
    for t in np.arange(lo, hi+step/2, step):
        p = apply_thresh(df, t, all_s1)
        f, d = macro_f05(p, gt)
        if f > best_f: best_f, best_t, best_d = f, float(t), d
    return best_t, best_f, best_d

def blocking_recall(pairs_df, gt):
    ps = set(zip(pairs_df['source1_entity_id'], pairs_df['candidate_entity_id']))
    tp_total, found = 0, 0
    comp, part, tot = 0, 0, 0
    for s1, ts in gt.items():
        if not ts: continue
        tot += 1
        f = sum(1 for t in ts if (s1,t) in ps)
        tp_total += len(ts); found += f
        if f == len(ts): comp += 1
        elif f > 0: part += 1
    return {
        'pair_recall': found/max(tp_total,1),
        'complete_recall': comp/max(tot,1),
        'total_pairs': len(pairs_df),
        'avg_cands': len(pairs_df)/max(len(gt),1),
    }

# ═══════════════════════════════════════════════════════════════
# LOAD PARQUET HELPERS
# ═══════════════════════════════════════════════════════════════

def load_pq(name):
    return pd.read_parquet(str(CACHE / f'{name}.parquet'))

def load_pq_country(name, country):
    safe = country.replace(' ','_')
    p = CACHE / f'{name}_{safe}.parquet'
    if p.exists():
        return pd.read_parquet(str(p))
    # Fallback: load full and filter
    df = load_pq(name)
    return df[df['country'] == country].copy()

def load_gt():
    df = load_pq('train_ground_truth')
    gt = {}
    for _, row in df.iterrows():
        s1 = row['source1_entity_id']
        m = row['matched_entity_ids']
        gt[s1] = set(m.split(',')) if m.strip() else set()
    return gt

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(sample_n=0, skip_test=False):
    t0 = time.time()
    log.info("=" * 60)
    log.info("GEMINI SOLUTION — Entity Resolution Pipeline")
    log.info("=" * 60)

    # Check parquet cache
    if not (CACHE / 'train_s1.parquet').exists():
        log.error("Parquet cache not found. Run 00_prepare_parquet.py first!")
        sys.exit(1)

    # ── Load S1 and GT ──
    log.info("\n[STAGE 1] Loading S1 + ground truth...")
    s1_all = load_pq('train_s1')
    gt_full = load_gt()
    
    if sample_n > 0:
        np.random.seed(42)
        sampled = np.random.choice(s1_all['entity_id'].values,
                                    min(sample_n, len(s1_all)), replace=False)
        sampled_set = set(sampled)
        s1_all = s1_all[s1_all['entity_id'].isin(sampled_set)]
        gt_full = {k: v for k, v in gt_full.items() if k in sampled_set}
        log.info(f"  Sampled {len(s1_all)} S1 entities")
    
    log.info(f"  S1: {len(s1_all)}, GT entries: {len(gt_full)}")
    singletons = sum(1 for v in gt_full.values() if len(v)==0)
    log.info(f"  Singletons: {singletons} ({100*singletons/max(len(gt_full),1):.1f}%)")
    
    countries = sorted(s1_all['country'].unique())
    log.info(f"  Countries: {countries}")
    
    # ── Normalize S1 ──
    log.info("\n[STAGE 2] Normalizing S1...")
    s1_all = normalize_df(s1_all)
    
    # ── Per-country blocking + features ──
    log.info("\n[STAGE 3] Per-country blocking & features...")
    all_train_data = []
    
    for country in countries:
        log.info(f"\n  ═══ Country: {country} ═══")
        s1_c = s1_all[s1_all['country'] == country]
        
        # Load country-specific S2, S3
        log.info(f"    Loading S2/S3 for {country}...")
        s2_c = load_pq_country('train_s2', country)
        s3_c = load_pq_country('train_s3', country)
        log.info(f"    S1: {len(s1_c)}, S2: {len(s2_c)}, S3: {len(s3_c)}")
        
        if len(s2_c) == 0 and len(s3_c) == 0:
            log.info(f"    No S2/S3 records, skipping")
            continue
        
        # Normalize S2/S3
        s2_c = normalize_df(s2_c)
        s3_c = normalize_df(s3_c)
        cand_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()
        
        # Blocking
        pairs_c = block_country(s1_c, cand_c, name_k=30, addr_k=20, comb_k=20)
        
        if len(pairs_c) == 0:
            del cand_c
            gc.collect()
            continue
        
        # Blocking recall
        gt_c = {s1: gt_full[s1] for s1 in s1_c['entity_id'] if s1 in gt_full}
        br = blocking_recall(pairs_c, gt_c)
        log.info(f"    Blocking: pair_recall={br['pair_recall']:.4f}, "
                 f"complete={br['complete_recall']:.4f}, "
                 f"avg={br['avg_cands']:.1f}")
        
        # Features (in chunks)
        log.info(f"    Computing features for {len(pairs_c)} pairs...")
        chunk_sz = 200000
        feat_list = []
        for cs in range(0, len(pairs_c), chunk_sz):
            ce = min(cs + chunk_sz, len(pairs_c))
            chunk = pairs_c.iloc[cs:ce]
            feat_list.append(compute_features(chunk, s1_c, cand_c))
            gc.collect()
        feats = pd.concat(feat_list, ignore_index=True)
        del feat_list
        gc.collect()
        
        # Labels
        s1_arr = pairs_c['source1_entity_id'].values
        c_arr = pairs_c['candidate_entity_id'].values
        labels = np.zeros(len(pairs_c), dtype=np.int8)
        for i in range(len(pairs_c)):
            if c_arr[i] in gt_full.get(s1_arr[i], set()):
                labels[i] = 1
        
        # Combine pairs + features + labels
        train_chunk = pd.concat([
            pairs_c[['source1_entity_id','candidate_entity_id']].reset_index(drop=True),
            feats.reset_index(drop=True),
        ], axis=1)
        train_chunk['label'] = labels
        all_train_data.append(train_chunk)
        
        log.info(f"    Pos: {labels.sum()}, Neg: {len(labels)-labels.sum()}")
        
        del pairs_c, feats, cand_c, train_chunk
        gc.collect()
    
    # ── Combine all training data ──
    log.info("\n[STAGE 4] Combining training data...")
    train_df = pd.concat(all_train_data, ignore_index=True)
    del all_train_data
    gc.collect()
    log.info(f"  Total pairs: {len(train_df)}")
    log.info(f"  Positives: {train_df['label'].sum()}, Negatives: {(train_df['label']==0).sum()}")
    
    # Hard negative sampling
    pos_mask = train_df['label'] == 1
    neg_mask = train_df['label'] == 0
    n_pos = pos_mask.sum()
    n_neg_target = min(neg_mask.sum(), n_pos * 10)
    
    if n_neg_target < neg_mask.sum():
        log.info(f"  Sampling {n_neg_target} hard negatives (from {neg_mask.sum()} total)")
        neg = train_df[neg_mask].copy()
        ret_cols = [c for c in neg.columns if c.startswith('r_') and 'flag' not in c]
        if ret_cols:
            neg['_h'] = neg[ret_cols].max(axis=1)
            neg = neg.nlargest(n_neg_target, '_h').drop(columns=['_h'])
        else:
            neg = neg.sample(n=n_neg_target, random_state=42)
        train_df = pd.concat([train_df[pos_mask], neg], ignore_index=True)
        del neg
        gc.collect()
    
    log.info(f"  Final: {len(train_df)} ({train_df['label'].sum()} pos, "
             f"{(train_df['label']==0).sum()} neg)")
    
    # ── Feature columns ──
    meta = {'source1_entity_id','candidate_entity_id','label','pred_score'}
    feat_cols = [c for c in train_df.columns if c not in meta]
    for c in feat_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors='coerce').fillna(0).astype(np.float32)
    log.info(f"  Features: {len(feat_cols)}")
    
    # ── LightGBM Training ──
    log.info("\n[STAGE 5] LightGBM 5-Fold Grouped CV...")
    params = {
        'objective':'binary', 'metric':'binary_logloss', 'boosting_type':'gbdt',
        'num_leaves':63, 'learning_rate':0.05, 'feature_fraction':0.8,
        'bagging_fraction':0.8, 'bagging_freq':5, 'min_child_samples':50,
        'lambda_l1':0.1, 'lambda_l2':0.1, 'verbose':-1, 'n_jobs':-1, 'seed':42,
    }
    
    gkf = GroupKFold(n_splits=5)
    groups = train_df['source1_entity_id'].values
    oof = np.zeros(len(train_df))
    models = []
    
    for fold, (tri, vai) in enumerate(gkf.split(train_df, groups=groups)):
        log.info(f"\n  === Fold {fold+1}/5 ===")
        Xtr = train_df.iloc[tri][feat_cols]
        ytr = train_df.iloc[tri]['label']
        Xva = train_df.iloc[vai][feat_cols]
        yva = train_df.iloc[vai]['label']
        log.info(f"  Train: {len(Xtr)} ({ytr.sum()} pos), Val: {len(Xva)} ({yva.sum()} pos)")
        
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dval = lgb.Dataset(Xva, label=yva, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=1000,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        oof[vai] = model.predict(Xva)
        models.append(model)
        
        # Fold F0.5
        vdf = train_df.iloc[vai][['source1_entity_id','candidate_entity_id']].copy()
        vdf['pred_score'] = oof[vai]
        vs1 = set(vdf['source1_entity_id'].unique())
        vgt = {s: gt_full[s] for s in vs1 if s in gt_full}
        bt, bf, bd = sweep_thresh(vdf, vgt, lo=0.30, hi=0.95, step=0.02)
        log.info(f"  Fold {fold+1}: t={bt:.2f} F0.5={bf:.4f} P={bd['macro_precision']:.4f} R={bd['macro_recall']:.4f}")
        del Xtr, ytr, Xva, yva, dtrain, dval, vdf
        gc.collect()
    
    # ── Threshold optimization ──
    log.info("\n[STAGE 6] Threshold optimization...")
    train_df['pred_score'] = oof
    oof_df = train_df[['source1_entity_id','candidate_entity_id','pred_score']].copy()
    best_t, best_f, best_d = sweep_thresh(oof_df, gt_full, lo=0.30, hi=0.95, step=0.005)
    
    log.info(f"\n  *** OOF Results ***")
    log.info(f"  Best threshold: {best_t:.3f}")
    log.info(f"  Macro F0.5:     {best_f:.4f}")
    log.info(f"  Precision:      {best_d['macro_precision']:.4f}")
    log.info(f"  Recall:         {best_d['macro_recall']:.4f}")
    log.info(f"  Singleton acc:  {best_d['singleton_accuracy']:.4f}")
    log.info(f"  Complete match: {best_d['complete_match_rate']:.4f}")
    
    # Feature importance
    imp = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': imp}).sort_values('importance', ascending=False)
    log.info(f"\n  Top 15 features:")
    for _, r in imp_df.head(15).iterrows():
        log.info(f"    {r['feature']}: {r['importance']:.0f}")
    
    # Save models
    log.info("\n[STAGE 7] Saving models...")
    for i, m in enumerate(models):
        m.save_model(str(MODELS / f'model_fold{i}.txt'))
    with open(MODELS / 'config.pkl', 'wb') as f:
        pickle.dump({'feature_cols': feat_cols, 'threshold': best_t,
                     'params': params, 'n_models': len(models),
                     'oof_f05': best_f, 'oof_detail': best_d}, f)
    
    # Experiment log
    exp = pd.DataFrame([{
        'experiment_id': 'v1', 'date': time.strftime('%Y-%m-%d %H:%M'),
        'model': 'LightGBM', 'threshold': best_t,
        'macro_f0_5': best_f,
        'macro_precision': best_d['macro_precision'],
        'macro_recall': best_d['macro_recall'],
        'singleton_accuracy': best_d['singleton_accuracy'],
        'notes': f'sample={sample_n}' if sample_n else 'full',
    }])
    exp.to_csv(str(EXPDIR / 'experiments.csv'), index=False)
    
    del train_df, oof_df
    gc.collect()
    
    # ── Test Inference ──
    if not skip_test:
        log.info("\n[STAGE 8] Test Inference...")
        run_test(models, feat_cols, best_t)
    else:
        log.info("\n[SKIPPED] Test inference (--skip-test)")
    
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"Done in {elapsed/60:.1f} min | F0.5={best_f:.4f} | t={best_t:.3f}")
    log.info(f"{'='*60}")
    return best_f, best_d


def run_test(models, feat_cols, threshold):
    log.info("  Loading test S1...")
    ts1 = load_pq('test_s1')
    log.info(f"  Test S1: {len(ts1)}")
    ts1 = normalize_df(ts1)
    
    countries = sorted(ts1['country'].unique())
    log.info(f"  Countries: {countries}")
    
    all_results = {}
    all_cands = {}
    
    for country in countries:
        log.info(f"\n  --- {country} ---")
        s1_c = ts1[ts1['country'] == country]
        
        log.info(f"    Loading S2/S3...")
        s2_c = load_pq_country('test_s2', country)
        s3_c = load_pq_country('test_s3', country)
        log.info(f"    S1:{len(s1_c)} S2:{len(s2_c)} S3:{len(s3_c)}")
        
        if len(s2_c)==0 and len(s3_c)==0:
            for sid in s1_c['entity_id']: all_results[sid] = set(); all_cands[sid] = set()
            continue
        
        s2_c = normalize_df(s2_c)
        s3_c = normalize_df(s3_c)
        cand_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c; gc.collect()
        
        # Process in S1 batches
        batch_sz = 50000
        s1_ids_all = s1_c['entity_id'].values
        
        for bs in range(0, len(s1_ids_all), batch_sz):
            be = min(bs + batch_sz, len(s1_ids_all))
            s1b = s1_c.iloc[bs:be]
            log.info(f"    Batch {bs}-{be}...")
            
            pairs = block_country(s1b, cand_c, name_k=30, addr_k=20, comb_k=20)
            
            if len(pairs) == 0:
                for sid in s1b['entity_id']: all_results[sid] = set(); all_cands[sid] = set()
                continue
            
            for s1id, grp in pairs.groupby('source1_entity_id'):
                all_cands[s1id] = set(grp['candidate_entity_id'])
            
            # Features
            f_list = []
            for fs in range(0, len(pairs), 200000):
                fe = min(fs+200000, len(pairs))
                f_list.append(compute_features(pairs.iloc[fs:fe], s1b, cand_c))
                gc.collect()
            feats = pd.concat(f_list, ignore_index=True)
            del f_list; gc.collect()
            
            for c in feat_cols:
                if c not in feats.columns: feats[c] = 0.0
            feats = feats[feat_cols]
            for c in feat_cols:
                feats[c] = pd.to_numeric(feats[c], errors='coerce').fillna(0).astype(np.float32)
            
            preds = np.mean([m.predict(feats) for m in models], axis=0)
            scored = pairs[['source1_entity_id','candidate_entity_id']].copy()
            scored['pred_score'] = preds
            
            for s1id, grp in scored.groupby('source1_entity_id'):
                matches = grp[grp['pred_score'] >= threshold]['candidate_entity_id'].tolist()
                all_results[s1id] = set(matches)
            
            del pairs, feats, scored; gc.collect()
        
        del cand_c; gc.collect()
    
    # Ensure all S1 present
    for sid in ts1['entity_id']:
        all_results.setdefault(sid, set())
        all_cands.setdefault(sid, set())
    
    # Write outputs
    log.info("  Writing matching_results.tsv...")
    with open(OUTPUT / 'matching_results.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1 in sorted(all_results):
            m = all_results[s1]
            f.write(f'{s1}\t{",".join(sorted(m)) if m else ""}\n')
    
    log.info("  Writing candidate_pairs.tsv...")
    with open(OUTPUT / 'candidate_pairs.tsv', 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1 in sorted(all_cands):
            c = all_cands[s1]
            f.write(f'{s1}\t{",".join(sorted(c)) if c else ""}\n')
    
    nm = sum(1 for v in all_results.values() if v)
    ns = sum(1 for v in all_results.values() if not v)
    log.info(f"  Matched: {nm}, Singletons: {ns}")
    
    # Validate
    log.info("  Running validator...")
    vp = RES / 'utils' / 'validate_submission.py'
    td = DATA / 'test'
    r = subprocess.run([sys.executable, str(vp),
                        '--matching', str(OUTPUT/'matching_results.tsv'),
                        '--candidate', str(OUTPUT/'candidate_pairs.tsv'),
                        '--test-dir', str(td)],
                       capture_output=True, text=True, timeout=300)
    log.info(f"  {r.stdout.strip()}")
    if r.stderr: log.info(f"  stderr: {r.stderr.strip()}")
    if r.returncode == 0: log.info("  ✓ VALIDATION PASSED")
    else: log.warning("  ✗ VALIDATION FAILED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--skip-test', action='store_true')
    args = parser.parse_args()
    run_pipeline(sample_n=args.sample, skip_test=args.skip_test)
