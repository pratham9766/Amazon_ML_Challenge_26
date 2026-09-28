"""TF-IDF based blocking for entity resolution.

Uses sparse character n-gram TF-IDF vectors and batched cosine similarity
to find top-K candidates for each query entity.

Designed for 8 GB RAM constraint:
- Processes queries in batches (default 5000)
- Uses scipy.sparse throughout
- Supports per-country processing
- Separate top-K retrieval for S2 and S3 candidates
"""
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class TfidfBlocker:
    """TF-IDF based candidate retrieval."""
    
    def __init__(
        self,
        analyzer: str = 'char_wb',
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: int = 500000,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        self.candidate_ids: Optional[np.ndarray] = None
        self.candidate_sources: Optional[np.ndarray] = None
        self.candidate_matrix: Optional[sparse.csr_matrix] = None
        
        # Source-specific matrices and ids for efficient source-specific top-K
        self.source_matrices: Dict[str, sparse.csr_matrix] = {}
        self.source_ids: Dict[str, np.ndarray] = {}
    
    def fit(self, candidate_texts: pd.Series, candidate_ids: pd.Series, candidate_sources: Optional[pd.Series] = None):
        """Fit TF-IDF on candidate texts and store the matrix.
        
        Args:
            candidate_texts: Texts to index.
            candidate_ids: IDs of the candidates.
            candidate_sources: Optional source identifiers (e.g., 'S2', 'S3'). 
                               If provided, enables source-specific Top-K.
        """
        texts = candidate_texts.fillna('').astype(str).tolist()
        self.candidate_matrix = self.vectorizer.fit_transform(texts)
        self.candidate_ids = candidate_ids.values
        
        if candidate_sources is not None:
            self.candidate_sources = candidate_sources.values
            unique_sources = np.unique(self.candidate_sources)
            
            for source in unique_sources:
                mask = (self.candidate_sources == source)
                self.source_matrices[source] = self.candidate_matrix[mask]
                self.source_ids[source] = self.candidate_ids[mask]
                
        return self
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the fitted vectorizer."""
        if hasattr(self.vectorizer, 'vocabulary_'):
            return len(self.vectorizer.vocabulary_)
        return 0
        
    def get_matrix_shape(self) -> Tuple[int, int]:
        """Get the shape of the candidate document-term matrix."""
        if self.candidate_matrix is not None:
            return self.candidate_matrix.shape
        return (0, 0)
    
    def _get_top_k_from_row(
        self, 
        row: sparse.csr_matrix, 
        ids: np.ndarray, 
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Extract top K candidates from a single sparse row."""
        if row.nnz == 0:
            return []
            
        row_data = row.data
        row_indices = row.indices
        
        if row.nnz <= top_k:
            sorted_idx = np.argsort(row_data)[::-1]
        else:
            partition_idx = np.argpartition(row_data, -top_k)[-top_k:]
            sorted_idx = partition_idx[np.argsort(row_data[partition_idx])[::-1]]
            
        candidates = []
        for idx in sorted_idx:
            score = float(row_data[idx])
            if score > 0:
                original_idx = row_indices[idx]
                candidates.append((ids[original_idx], score))
                
        return candidates

    def query_batch(
        self,
        query_texts: pd.Series,
        query_ids: pd.Series,
        top_k: int = 30,
        batch_size: int = 5000,
        show_progress: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Find top-K candidates for each query.
        
        If candidate_sources were provided during fit, this retrieves top_k 
        candidates *per source*.
        
        Returns:
            Dict mapping query_id -> list of (candidate_id, score) tuples,
            sorted by score descending.
        """
        results = {}
        texts = query_texts.fillna('').astype(str).tolist()
        ids = query_ids.values
        
        num_batches = int(np.ceil(len(texts) / batch_size))
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Querying batches")
            
        for start in iterator:
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_ids = ids[start:end]
            
            # Transform query batch
            query_matrix = self.vectorizer.transform(batch_texts)
            
            if self.source_matrices:
                # Source-specific retrieval
                for source, target_matrix in self.source_matrices.items():
                    target_ids = self.source_ids[source]
                    sim_matrix = query_matrix.dot(target_matrix.T)
                    
                    for i in range(sim_matrix.shape[0]):
                        qid = batch_ids[i]
                        row = sim_matrix.getrow(i)
                        
                        candidates = self._get_top_k_from_row(row, target_ids, top_k)
                        if qid not in results:
                            results[qid] = []
                        results[qid].extend(candidates)
            else:
                # Global retrieval
                sim_matrix = query_matrix.dot(self.candidate_matrix.T)
                for i in range(sim_matrix.shape[0]):
                    qid = batch_ids[i]
                    row = sim_matrix.getrow(i)
                    
                    candidates = self._get_top_k_from_row(row, self.candidate_ids, top_k)
                    results[qid] = candidates
            
            # Sort combined results if doing source-specific retrieval
            if self.source_matrices:
                for qid in batch_ids:
                    results[qid].sort(key=lambda x: x[1], reverse=True)
            
            del query_matrix
            if 'sim_matrix' in locals():
                del sim_matrix
            gc.collect()
        
        return results
    
    def free(self):
        """Free memory from stored matrices."""
        self.candidate_matrix = None
        self.candidate_ids = None
        self.candidate_sources = None
        self.source_matrices.clear()
        self.source_ids.clear()
        gc.collect()
