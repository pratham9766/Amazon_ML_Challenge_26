"""End-to-end test inference pipeline.

Processes test data through:
1. Load test data
2. Normalize
3. Block per country
4. Generate features
5. Score with model
6. Apply threshold
7. Write output files

Designed for 8 GB RAM: processes one country at a time,
batched operations, incremental output writing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
import gc
import time
from tqdm import tqdm
import logging

from ..data.loader import (
    load_or_cache_source, ensure_dirs, GEMINI_DIR, DATASET_DIR
)
from ..normalization.normalizer import normalize_dataframe
from ..blocking.tfidf_blocker import TfidfBlocker
from ..blocking.exact_blocker import ExactBlocker
from ..blocking.union_blocker import merge_candidates
from ..features.feature_engineer import compute_all_features
from ..models.trainer import EntityResolutionTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end test inference."""
    
    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.5,
        singleton_threshold: Optional[float] = None,
        batch_size: int = 50000,
    ):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.singleton_threshold = singleton_threshold
        self.batch_size = batch_size
        self.trainer = EntityResolutionTrainer()
    
    def run(self):
        """Run full inference pipeline."""
        logger.info("=== GEMINI SOLUTION: Test Inference ===")
        start_time = time.time()
        
        # 1. Load test data
        logger.info("[1/7] Loading test data...")
        test_s1 = load_or_cache_source('test', 1)
        test_s2 = load_or_cache_source('test', 2)
        test_s3 = load_or_cache_source('test', 3)
        logger.info(f"  S1: {len(test_s1)}, S2: {len(test_s2)}, S3: {len(test_s3)}")
        
        # 2. Normalize
        logger.info("[2/7] Normalizing...")
        test_s1 = normalize_dataframe(test_s1)
        test_s2 = normalize_dataframe(test_s2)
        test_s3 = normalize_dataframe(test_s3)
        
        # 3. Load model
        logger.info("[3/7] Loading model...")
        self.trainer.load(str(self.model_dir))
        
        # 4. Process per country
        countries = sorted(test_s1['country'].dropna().unique())
        if not countries:
            logger.warning("No countries found, defaulting to processing all data together.")
            countries = [None]
            
        logger.info(f"[4/7] Processing {len(countries)} countries: {countries}")
        
        all_results = {}  # s1_id -> set(matched_ids)
        all_candidates = {}  # s1_id -> set(candidate_ids)
        
        for country in countries:
            logger.info(f"\n  --- Country: {country} ---")
            if country is not None:
                s1_c = test_s1[test_s1['country'] == country]
                s2_c = test_s2[test_s2['country'] == country]
                s3_c = test_s3[test_s3['country'] == country]
            else:
                s1_c, s2_c, s3_c = test_s1, test_s2, test_s3
                
            results, candidates = self._process_country(s1_c, s2_c, s3_c)
            all_results.update(results)
            all_candidates.update(candidates)
            gc.collect()
        
        # Ensure all S1 entities are present
        for s1_id in test_s1['entity_id'].values:
            if s1_id not in all_results:
                all_results[s1_id] = set()
            if s1_id not in all_candidates:
                all_candidates[s1_id] = set()
        
        # 5. Write outputs
        logger.info("[5/7] Writing outputs...")
        self._write_matching_results(all_results)
        self._write_candidate_pairs(all_candidates)
        
        elapsed = time.time() - start_time
        logger.info(f"=== Inference complete in {elapsed:.0f}s ===")
        logger.info(f"  Total S1: {len(all_results)}")
        logger.info(f"  Singletons: {sum(1 for v in all_results.values() if len(v) == 0)}")
        logger.info(f"  With matches: {sum(1 for v in all_results.values() if len(v) > 0)}")
    
    def _process_country(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """Process one country: blocking -> features -> scoring -> thresholding."""
        # Combine S2+S3
        candidates_pool = pd.concat([s2_df, s3_df], ignore_index=True)
        
        if len(s1_df) == 0 or len(candidates_pool) == 0:
            return {}, {}
        
        logger.info(f"    S1: {len(s1_df)}, S2+S3: {len(candidates_pool)}")
        
        all_results = {}
        all_candidates = {}
        
        # 1. Blocking
        logger.info("    Blocking...")
        # TF-IDF on names
        tfidf_name = TfidfBlocker(field='name', analyzer='char_wb', ngram_range=(3, 5), top_k=20)
        tfidf_name.fit(candidates_pool)
        cands_name = tfidf_name.get_candidates(s1_df)
        
        # TF-IDF on addresses
        tfidf_addr = TfidfBlocker(field='address', analyzer='char_wb', ngram_range=(3, 5), top_k=10)
        tfidf_addr.fit(candidates_pool)
        cands_addr = tfidf_addr.get_candidates(s1_df)
        
        # Exact on suffix-stripped names
        exact_name_stripped = ExactBlocker(field='name_normalized_stripped')
        exact_name_stripped.fit(candidates_pool)
        cands_exact_stripped = exact_name_stripped.get_candidates(s1_df)
        
        # Merge candidates
        merged_cands = merge_candidates([cands_name, cands_addr, cands_exact_stripped])
        
        if merged_cands.empty:
            logger.warning("    No candidates found after blocking.")
            return all_results, all_candidates
            
        # Group candidates by S1 for easy lookup later
        for _, row in merged_cands.iterrows():
            s1_id = row['source1_entity_id']
            if s1_id not in all_candidates:
                all_candidates[s1_id] = set()
            all_candidates[s1_id].add(row['candidate_entity_id'])

        # Split into sub-batches for S1 to manage memory during feature engineering
        s1_ids = s1_df['entity_id'].unique()
        for i in range(0, len(s1_ids), self.batch_size):
            batch_s1_ids = s1_ids[i:i + self.batch_size]
            batch_cands = merged_cands[merged_cands['source1_entity_id'].isin(batch_s1_ids)]
            
            if batch_cands.empty:
                continue
                
            batch_s1 = s1_df[s1_df['entity_id'].isin(batch_s1_ids)]
            batch_candidates_df = candidates_pool[candidates_pool['entity_id'].isin(batch_cands['candidate_entity_id'])]
            
            logger.info(f"    Computing features for batch {i//self.batch_size + 1} ({len(batch_cands)} pairs)...")
            
            # 2. Features
            features_df = compute_all_features(batch_cands, batch_s1, batch_candidates_df)
            
            # 3. Scoring
            logger.info("    Scoring...")
            # Predict only on pairs we successfully computed features for
            if features_df.empty:
                continue
            
            preds = self.trainer.predict(features_df)
            features_df['pred_score'] = preds
            
            # 4. Thresholding
            for s1_id in batch_s1_ids:
                s1_pairs = features_df[features_df['source1_entity_id'] == s1_id]
                
                matched = set()
                if not s1_pairs.empty:
                    # Apply threshold
                    above_threshold = s1_pairs[s1_pairs['pred_score'] >= self.threshold]
                    
                    if not above_threshold.empty:
                        matched.update(above_threshold['candidate_entity_id'].tolist())
                    elif self.singleton_threshold is not None:
                        # Fallback for singleton threshold
                        best_match = s1_pairs.loc[s1_pairs['pred_score'].idxmax()]
                        if best_match['pred_score'] >= self.singleton_threshold:
                            matched.add(best_match['candidate_entity_id'])
                
                all_results[s1_id] = matched

        return all_results, all_candidates
    
    def _write_matching_results(self, results: Dict[str, Set[str]]):
        """Write matching_results.tsv."""
        output_dir = GEMINI_DIR / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'matching_results.tsv'
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('source1_entity_id\tmatched_entity_ids\n')
            for s1_id in sorted(results.keys()):
                matched = results[s1_id]
                matched_str = ','.join(sorted(matched)) if matched else ''
                f.write(f'{s1_id}\t{matched_str}\n')
        logger.info(f"  Written: {path}")
    
    def _write_candidate_pairs(self, candidates: Dict[str, Set[str]]):
        """Write candidate_pairs.tsv."""
        output_dir = GEMINI_DIR / 'output'
        path = output_dir / 'candidate_pairs.tsv'
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('source1_entity_id\tcandidate_entity_ids\n')
            for s1_id in sorted(candidates.keys()):
                cands = candidates[s1_id]
                cands_str = ','.join(sorted(cands)) if cands else ''
                f.write(f'{s1_id}\t{cands_str}\n')
        logger.info(f"  Written: {path}")
