"""Threshold optimization and singleton detection.

Optimizes the global match threshold and singleton detection
for the F0.5 metric using OOF predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple, Optional, List

def sweep_global_threshold(
    oof_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[np.ndarray] = None,
    score_col: str = 'pred_score',
) -> Tuple[float, float, pd.DataFrame]:
    """Sweep global threshold to maximize macro F0.5."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.99, 0.005)
    
    results = []
    # Avoid recalculating grouping inside loop by converting to dict first
    groups = oof_df.groupby('source1_entity_id')
    group_dict = {s1: group[score_col].values for s1, group in groups}
    cand_dict = {s1: group['candidate_entity_id'].values for s1, group in groups}
    
    from ..validation.evaluator import macro_f05, detailed_metrics
    
    for t in thresholds:
        predicted = {s1: set() for s1 in ground_truth}
        
        for s1, scores in group_dict.items():
            mask = scores >= t
            if np.any(mask):
                predicted[s1] = set(cand_dict[s1][mask])
        
        details = detailed_metrics(predicted, ground_truth)
        results.append({
            'threshold': t,
            'macro_f05': details.get('macro_f05', 0),
            'precision': details.get('macro_precision', 0),
            'recall': details.get('macro_recall', 0),
            'singleton_accuracy': details.get('singleton_accuracy', 0),
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['macro_f05'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f05 = results_df.loc[best_idx, 'macro_f05']
    
    return best_threshold, best_f05, results_df

def compute_entity_level_features(
    scores_df: pd.DataFrame,
    score_col: str = 'pred_score',
) -> pd.DataFrame:
    """Compute entity-level features for singleton detection."""
    grouped = scores_df.groupby('source1_entity_id')
    
    # We can use agg with custom functions or compute vector-wise if sorting
    # A simple approach: sort and pick top 3
    sorted_df = scores_df.sort_values(['source1_entity_id', score_col], ascending=[True, False])
    
    def get_top_k(g, k):
        return g[score_col].iloc[k] if len(g) > k else 0.0
        
    res = grouped.agg(
        candidate_count=('candidate_entity_id', 'count'),
        strong_candidate_count=(score_col, lambda x: (x > 0.5).sum()),
        score_std=(score_col, 'std'),
        score_mean=(score_col, 'mean')
    )
    
    # Get top1, top2, top3
    top_scores = sorted_df.groupby('source1_entity_id')[score_col].apply(lambda x: list(x)[:3]).to_dict()
    
    b1, b2, b3 = [], [], []
    for s1 in res.index:
        scores = top_scores.get(s1, [])
        b1.append(scores[0] if len(scores) > 0 else 0)
        b2.append(scores[1] if len(scores) > 1 else 0)
        b3.append(scores[2] if len(scores) > 2 else 0)
        
    res['best_score'] = b1
    res['second_score'] = b2
    res['third_score'] = b3
    
    res['top1_top2_margin'] = res['best_score'] - res['second_score']
    res['top2_top3_margin'] = res['second_score'] - res['third_score']
    
    res['score_std'] = res['score_std'].fillna(0)
    
    return res

def apply_threshold_with_singleton_logic(
    scores_df: pd.DataFrame,
    threshold: float,
    singleton_threshold: Optional[float] = None,
    score_col: str = 'pred_score',
) -> Dict[str, Set[str]]:
    """Apply threshold with additional singleton detection logic."""
    predicted = {}
    
    if singleton_threshold is not None:
        entity_feats = compute_entity_level_features(scores_df, score_col)
        singletons = set(entity_feats[entity_feats['best_score'] < singleton_threshold].index)
    else:
        singletons = set()
        
    above_thresh = scores_df[scores_df[score_col] >= threshold]
    
    for s1_id in scores_df['source1_entity_id'].unique():
        predicted[s1_id] = set()
        
    for _, row in above_thresh.iterrows():
        s1 = row['source1_entity_id']
        if s1 not in singletons:
            cand = row['candidate_entity_id']
            predicted[s1].add(cand)
            
    return predicted
