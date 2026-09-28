"""Exact match blocking using inverted index.

Finds candidates that share exact normalized names with queries.
Includes fanout control to prevent explosion on common names.
"""
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class ExactBlocker:
    """Inverted index exact match blocker."""
    
    def __init__(self, fanout_cap: int = 200):
        """Initialize the ExactBlocker.
        
        Args:
            fanout_cap: Maximum number of matches to allow for a single key.
                       Prevents explosion on extremely common names/attributes.
        """
        self.fanout_cap = fanout_cap
        self.index: Dict[str, List[str]] = defaultdict(list)
    
    def fit(self, candidate_keys: pd.Series, candidate_ids: pd.Series):
        """Build inverted index: normalized_name -> list of candidate_ids.
        
        Args:
            candidate_keys: Series of keys to index.
            candidate_ids: Series of candidate IDs corresponding to keys.
        """
        for key, cid in zip(candidate_keys.fillna(''), candidate_ids):
            key = str(key).strip()
            if key and key.lower() not in ('nan', 'none', 'null'):
                self.index[key].append(cid)
        return self
    
    def query(self, query_keys: pd.Series, query_ids: pd.Series) -> Dict[str, List[Tuple[str, float]]]:
        """Find exact matches for each query.
        
        Args:
            query_keys: Series of keys to look up.
            query_ids: Series of query IDs.
            
        Returns:
            Dict mapping query_id -> list of (candidate_id, 1.0) tuples.
            Applies fanout cap to prevent explosion on common names.
        """
        results = {}
        for key, qid in zip(query_keys.fillna(''), query_ids):
            key = str(key).strip()
            
            # Skip invalid or empty keys
            if not key or key.lower() in ('nan', 'none', 'null'):
                results[qid] = []
                continue
                
            matches = self.index.get(key, [])
            if len(matches) > self.fanout_cap:
                # Skip keys with too many matches (common/generic names)
                results[qid] = []
            elif matches:
                # Return matches with a score of 1.0
                results[qid] = [(cid, 1.0) for cid in matches]
            else:
                results[qid] = []
                
        return results
    
    def free(self):
        """Free the inverted index memory."""
        self.index.clear()
