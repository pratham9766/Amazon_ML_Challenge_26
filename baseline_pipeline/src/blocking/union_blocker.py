"""Union blocker that merges multiple blocking channels.

Combines results from TF-IDF, exact match, and other blocking channels
into a single candidate set with provenance tracking.
"""
import gc
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm

from src.blocking.tfidf_blocker import TfidfBlocker
from src.blocking.exact_blocker import ExactBlocker

logger = logging.getLogger(__name__)


def merge_candidates(
    channel_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    all_query_ids: List[str],
) -> pd.DataFrame:
    """Merge candidates from multiple channels into a single DataFrame.
    
    Args:
        channel_results: {channel_name: {query_id: [(candidate_id, score), ...]}}
        all_query_ids: Complete list of query IDs
    
    Returns:
        DataFrame with columns:
        - source1_entity_id
        - candidate_entity_id
        - One score column per channel
        - One rank column per channel
        - channel_count: number of channels that found this pair
    """
    pair_data = defaultdict(lambda: {})
    
    for channel_name, results in channel_results.items():
        for qid, candidates in results.items():
            for rank, (cid, score) in enumerate(candidates, 1):
                if qid not in pair_data or cid not in pair_data[qid]:
                    pair_data[qid][cid] = {}
                pair_data[qid][cid][f'{channel_name}_score'] = score
                pair_data[qid][cid][f'{channel_name}_rank'] = rank
    
    rows = []
    channel_names = list(channel_results.keys())
    
    for qid in all_query_ids:
        if qid in pair_data:
            for cid, scores in pair_data[qid].items():
                row = {
                    'source1_entity_id': qid,
                    'candidate_entity_id': cid,
                }
                channel_count = 0
                for ch_name in channel_names:
                    score_key = f'{ch_name}_score'
                    rank_key = f'{ch_name}_rank'
                    row[score_key] = scores.get(score_key, 0.0)
                    row[rank_key] = scores.get(rank_key, 999)
                    if score_key in scores:
                        channel_count += 1
                row['channel_count'] = channel_count
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Convert numeric columns to float32/int16 for memory
    for col in df.columns:
        if col.endswith('_score'):
            df[col] = df[col].astype(np.float32)
        elif col.endswith('_rank'):
            df[col] = df[col].astype(np.int16)
    df['channel_count'] = df['channel_count'].astype(np.int8)
    
    return df


class UnionBlocker:
    """Orchestrates multiple blocking strategies across datasets."""
    
    def __init__(self, top_k_tfidf: int = 30, batch_size: int = 5000):
        self.top_k_tfidf = top_k_tfidf
        self.batch_size = batch_size
        
    def _measure_recall(self, candidates_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
        """Measure the recall of the blocking stage."""
        if truth_df is None or truth_df.empty:
            return 0.0
            
        merged = truth_df.merge(
            candidates_df[['source1_entity_id', 'candidate_entity_id']], 
            left_on=['source_1', 'source_2'], 
            right_on=['source1_entity_id', 'candidate_entity_id'], 
            how='left',
            indicator=True
        )
        
        found = (merged['_merge'] == 'both').sum()
        total = len(truth_df)
        return found / total if total > 0 else 0.0

    def run(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        truth_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Run union blocking logic per country.
        
        Args:
            df_s1: Source 1 data (Queries).
            df_s2: Source 2 data (Candidates).
            df_s3: Source 3 data (Candidates).
            truth_df: Optional ground truth for recall calculation.
            
        Returns:
            pd.DataFrame: Merged candidate pairs.
        """
        all_candidates = []
        countries = df_s1['country'].dropna().unique()
        
        # Combine S2 and S3 for candidates pool
        df_s2_pool = df_s2.copy()
        df_s2_pool['source_name'] = 'S2'
        
        df_s3_pool = df_s3.copy()
        df_s3_pool['source_name'] = 'S3'
        
        df_candidates = pd.concat([df_s2_pool, df_s3_pool], ignore_index=True)
        
        for country in sorted(countries):
            logger.info(f"Processing country: {country}")
            
            c_s1 = df_s1[df_s1['country'] == country]
            c_cand = df_candidates[df_candidates['country'] == country]
            
            if c_s1.empty or c_cand.empty:
                continue
                
            channel_results = {}
            
            # --- 1. TF-IDF Name Blocking ---
            if 'normalized_name' in c_s1.columns and 'normalized_name' in c_cand.columns:
                logger.info(f"[{country}] Running TF-IDF on names...")
                tfidf_name = TfidfBlocker()
                tfidf_name.fit(c_cand['normalized_name'], c_cand['entity_id'], c_cand['source_name'])
                channel_results['tfidf_name'] = tfidf_name.query_batch(
                    c_s1['normalized_name'], 
                    c_s1['entity_id'], 
                    top_k=self.top_k_tfidf,
                    batch_size=self.batch_size
                )
                tfidf_name.free()
                
            # --- 2. Exact Name Blocking ---
            if 'normalized_name' in c_s1.columns and 'normalized_name' in c_cand.columns:
                logger.info(f"[{country}] Running Exact on names...")
                exact_name = ExactBlocker(fanout_cap=200)
                exact_name.fit(c_cand['normalized_name'], c_cand['entity_id'])
                channel_results['exact_name'] = exact_name.query(c_s1['normalized_name'], c_s1['entity_id'])
                exact_name.free()
                
            # Combine for country
            if channel_results:
                logger.info(f"[{country}] Merging candidates...")
                country_candidates_df = merge_candidates(
                    channel_results, 
                    c_s1['entity_id'].tolist()
                )
                
                # Check subset recall if truth provided
                if truth_df is not None:
                    truth_sub = truth_df[truth_df['source_1'].isin(c_s1['entity_id'])]
                    if not truth_sub.empty:
                        recall = self._measure_recall(country_candidates_df, truth_sub)
                        logger.info(f"[{country}] Blocking Recall: {recall:.4f} ({len(country_candidates_df)} pairs)")
                
                all_candidates.append(country_candidates_df)
                
            # Free memory
            del c_s1, c_cand, channel_results
            gc.collect()
            
        if not all_candidates:
            return pd.DataFrame()
            
        final_df = pd.concat(all_candidates, ignore_index=True)
        
        # Overall recall check
        if truth_df is not None:
            overall_recall = self._measure_recall(final_df, truth_df)
            logger.info(f"[Overall] Total Blocking Recall: {overall_recall:.4f} ({len(final_df)} pairs generated)")
            
        return final_df
