"""Entity-level evaluation for the competition.

Implements the exact macro-averaged F0.5 metric,
grouped cross-validation, and comprehensive diagnostics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from sklearn.model_selection import GroupKFold

def compute_f05_per_entity(
    predicted: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute F0.5 for each S1 entity."""
    scores = {}
    for s1_id in ground_truth:
        gt = ground_truth[s1_id]
        pred = predicted.get(s1_id, set())
        
        if len(gt) == 0:
            # True singleton
            scores[s1_id] = 1.0 if len(pred) == 0 else 0.0
        elif len(pred) == 0:
            # Predicted singleton but has true matches
            scores[s1_id] = 0.0
        else:
            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall == 0:
                scores[s1_id] = 0.0
            else:
                scores[s1_id] = (1.25 * precision * recall) / (0.25 * precision + recall)
    
    return scores

def macro_f05(predicted: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> float:
    """Compute macro-averaged F0.5."""
    scores = compute_f05_per_entity(predicted, ground_truth)
    return np.mean(list(scores.values())) if scores else 0.0

def detailed_metrics(predicted: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Dict:
    """Compute comprehensive metrics breakdown."""
    metrics = {
        'macro_f05': 0.0,
        'macro_precision': 0.0,
        'macro_recall': 0.0,
        'singleton_accuracy': 0.0,
        'non_singleton_f05': 0.0,
        'complete_match_rate': 0.0,
        'false_positives_per_1000': 0.0
    }
    
    if not ground_truth:
        return metrics
        
    precisions = []
    recalls = []
    f05_scores = []
    ns_f05 = []
    singletons_correct = 0
    singletons_total = 0
    complete_matches = 0
    total_fp = 0
    
    for s1_id, gt in ground_truth.items():
        pred = predicted.get(s1_id, set())
        
        if len(gt) == 0:
            singletons_total += 1
            if len(pred) == 0:
                singletons_correct += 1
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                total_fp += len(pred)
        else:
            tp = len(pred & gt)
            fp = len(pred - gt)
            fn = len(gt - pred)
            total_fp += fp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            
            if p + r > 0:
                f05 = (1.25 * p * r) / (0.25 * p + r)
            else:
                f05 = 0.0
            f05_scores.append(f05)
            ns_f05.append(f05)
            
            if fp == 0 and fn == 0:
                complete_matches += 1
                
    metrics['macro_f05'] = np.mean(f05_scores) if f05_scores else 0.0
    metrics['macro_precision'] = np.mean(precisions) if precisions else 0.0
    metrics['macro_recall'] = np.mean(recalls) if recalls else 0.0
    metrics['singleton_accuracy'] = singletons_correct / singletons_total if singletons_total > 0 else 0.0
    metrics['non_singleton_f05'] = np.mean(ns_f05) if ns_f05 else 0.0
    ns_total = len(ground_truth) - singletons_total
    metrics['complete_match_rate'] = complete_matches / ns_total if ns_total > 0 else 0.0
    metrics['false_positives_per_1000'] = (total_fp / len(ground_truth)) * 1000
    
    return metrics

def create_grouped_cv_splits(
    s1_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create grouped CV splits by S1 entity ID."""
    gkf = GroupKFold(n_splits=n_splits)
    # create dummy X
    X = np.zeros(len(s1_ids))
    return list(gkf.split(X, groups=s1_ids))

def predictions_from_threshold(
    scores_df: pd.DataFrame,
    threshold: float,
    score_col: str = 'pred_score',
) -> Dict[str, Set[str]]:
    """Convert scores to predictions using a threshold."""
    predicted = {}
    
    # Init empty sets for all sources
    for s1_id in scores_df['source1_entity_id'].unique():
        predicted[s1_id] = set()
        
    above_thresh = scores_df[scores_df[score_col] >= threshold]
    for _, row in above_thresh.iterrows():
        s1 = row['source1_entity_id']
        cand = row['candidate_entity_id']
        predicted[s1].add(cand)
        
    return predicted

def evaluate_at_threshold(
    scores_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    threshold: float,
    score_col: str = 'pred_score',
) -> Dict:
    pred = predictions_from_threshold(scores_df, threshold, score_col)
    
    # Ensure all gt keys exist
    for k in ground_truth:
        if k not in pred:
            pred[k] = set()
            
    return detailed_metrics(pred, ground_truth)

def sweep_thresholds(
    scores_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[np.ndarray] = None,
    score_col: str = 'pred_score',
):
    if thresholds is None:
        thresholds = np.arange(0.3, 1.0, 0.05)
    
    best_t = 0.0
    best_f05 = -1.0
    
    for t in thresholds:
        metrics = evaluate_at_threshold(scores_df, ground_truth, t, score_col)
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_t = t
            
    return best_t, best_f05
