"""LightGBM model training for entity resolution.

Features:
- Grouped K-fold CV (split by source1_entity_id)
- OOF prediction collection
- Hard negative sampling
- Feature importance tracking
- Model serialization
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pickle
import gc
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_DIR = Path('d:/RESOURCES/AmazonMLChallenge/gemini_solution')


class EntityResolutionTrainer:
    """Train LightGBM for entity resolution with grouped CV."""
    
    FEATURE_COLS = None  # Set during training
    
    LGB_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    def __init__(self, params: Optional[Dict] = None, n_estimators: int = 1000,
                 early_stopping_rounds: int = 50):
        self.params = {**self.LGB_PARAMS, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.models = []
        self.feature_importance = None
    
    def prepare_training_data(
        self,
        candidates_df: pd.DataFrame,
        features_df: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        neg_ratio: int = 10,
    ) -> pd.DataFrame:
        """Prepare training data with hard negative sampling.
        
        1. Merge candidates with features
        2. Label pairs using ground truth
        3. Keep ALL positive pairs
        4. Sample hard negatives (top-ranked false candidates)
        5. Return balanced training DataFrame
        """
        logger.info("Merging candidates with features...")
        # Merge features
        train_df = candidates_df.merge(features_df, 
                                        on=['source1_entity_id', 'candidate_entity_id'], 
                                        how='inner')
        
        logger.info("Creating labels...")
        # Create labels
        labels = []
        for _, row in train_df.iterrows():
            s1_id = row['source1_entity_id']
            c_id = row['candidate_entity_id']
            gt = ground_truth.get(s1_id, set())
            labels.append(1 if c_id in gt else 0)
        train_df['label'] = labels
        
        # Separate positives and negatives
        pos_df = train_df[train_df['label'] == 1]
        neg_df = train_df[train_df['label'] == 0]
        
        logger.info(f"Total positive pairs: {len(pos_df)}")
        logger.info(f"Total negative pairs: {len(neg_df)}")
        
        # Sample negatives: prioritize hard negatives (higher retrieval scores)
        n_neg = min(len(neg_df), max(len(pos_df) * neg_ratio, 1))
        if n_neg < len(neg_df):
            # Sort by max retrieval score descending to get hardest negatives
            score_cols = [c for c in neg_df.columns if c.endswith('_score') and 'pred' not in c]
            if score_cols:
                logger.info("Sampling hard negatives based on retrieval scores...")
                neg_df = neg_df.copy()
                neg_df['_max_ret_score'] = neg_df[score_cols].max(axis=1)
                neg_df = neg_df.nlargest(n_neg, '_max_ret_score')
                neg_df = neg_df.drop(columns=['_max_ret_score'])
            else:
                logger.info("Sampling random negatives...")
                neg_df = neg_df.sample(n=n_neg, random_state=42)
        
        result = pd.concat([pos_df, neg_df], ignore_index=True)
        logger.info(f"Final training data shape: {result.shape}")
        return result
    
    def train_cv(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        n_splits: int = 5,
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ) -> pd.DataFrame:
        """Train with grouped K-fold CV.
        
        Groups by source1_entity_id so all pairs for an S1 are in the same fold.
        Returns DataFrame with OOF predictions.
        """
        self.FEATURE_COLS = feature_cols
        oof_preds = np.zeros(len(train_df))
        self.models = []
        
        groups = train_df['source1_entity_id'].values
        gkf = GroupKFold(n_splits=n_splits)
        
        importances = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
            logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")
            logger.info(f"  Train: {len(train_idx)} pairs, Val: {len(val_idx)} pairs")
            
            X_train = train_df.iloc[train_idx][feature_cols]
            y_train = train_df.iloc[train_idx]['label']
            X_val = train_df.iloc[val_idx][feature_cols]
            y_val = train_df.iloc[val_idx]['label']
            
            logger.info(f"  Train pos rate: {y_train.mean():.4f}, Val pos rate: {y_val.mean():.4f}")
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds),
                    lgb.log_evaluation(100),
                ],
            )
            
            oof_preds[val_idx] = model.predict(X_val)
            self.models.append(model)
            
            # Feature importance
            imp = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importance(importance_type='gain'),
                'fold': fold,
            })
            importances.append(imp)
            
            # If ground truth provided, compute fold F0.5
            if ground_truth:
                try:
                    from ..validation.evaluator import macro_f05, predictions_from_scores
                    val_df = train_df.iloc[val_idx].copy()
                    val_df['pred_score'] = oof_preds[val_idx]
                    # Sweep a few thresholds
                    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                        preds = predictions_from_scores(val_df, t)
                        fold_gt = {s1: ground_truth[s1] for s1 in preds if s1 in ground_truth}
                        f05 = macro_f05(preds, fold_gt)
                        logger.info(f"    Fold {fold+1} @ t={t:.1f}: F0.5={f05:.4f}")
                except ImportError:
                    logger.warning("Could not import validation metrics. Skipping fold evaluation.")
            
            gc.collect()
        
        # Aggregate feature importance
        if importances:
            self.feature_importance = pd.concat(importances).groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # Add OOF predictions
        train_df = train_df.copy()
        train_df['pred_score'] = oof_preds
        
        return train_df
    
    def train_full(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> lgb.Booster:
        """Train on full data for final model."""
        logger.info("Training full final model...")
        self.FEATURE_COLS = feature_cols
        X = train_df[feature_cols]
        y = train_df['label']
        
        train_data = lgb.Dataset(X, label=y)
        
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
        )
        
        self.models = [model]
        return model
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using ensemble of CV models."""
        if not self.models:
            raise ValueError("No models found. Train or load models before predicting.")
        
        preds = np.zeros(len(df))
        for model in self.models:
            preds += model.predict(df[self.FEATURE_COLS])
        return preds / len(self.models)
    
    def save(self, path: str):
        """Save models and config."""
        try:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(self.models):
                model.save_model(str(save_dir / f'model_fold{i}.txt'))
            
            with open(save_dir / 'config.pkl', 'wb') as f:
                pickle.dump({
                    'params': self.params,
                    'feature_cols': self.FEATURE_COLS,
                    'n_models': len(self.models),
                    'feature_importance': self.feature_importance,
                }, f)
            logger.info(f"Models and config saved successfully to {save_dir}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load(self, path: str):
        """Load models and config."""
        try:
            save_dir = Path(path)
            with open(save_dir / 'config.pkl', 'rb') as f:
                config = pickle.load(f)
            self.params = config['params']
            self.FEATURE_COLS = config['feature_cols']
            self.feature_importance = config.get('feature_importance')
            self.models = []
            for i in range(config['n_models']):
                model = lgb.Booster(model_file=str(save_dir / f'model_fold{i}.txt'))
                self.models.append(model)
            logger.info(f"Loaded {len(self.models)} models from {save_dir}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
