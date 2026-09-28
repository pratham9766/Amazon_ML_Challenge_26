# Gemini Solution — Business Entity Resolution

## Overview
Competition-grade entity resolution system for the Amazon ML Challenge 2026.
Identifies matching business records across three data sources using
TF-IDF blocking, feature engineering, and LightGBM classification.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (development sample)
python scripts/run_all.py --sample 50000

# Run full pipeline (all data)
python scripts/run_all.py

# Run without test inference (training/validation only)
python scripts/run_all.py --sample 50000 --skip-test
```

## Architecture

```
Raw TSV Data
    ↓
[Normalization] NFKC + case + suffix strip + tokenization
    ↓
[Blocking] Per-country, multi-channel (5 channels):
  1. Name char TF-IDF (3-5 grams, top-30 per source)
  2. Address char TF-IDF (3-5 grams, top-20 per source)
  3. Combined name+address TF-IDF (top-20 per source)
  4. Exact suffix-stripped name match
  5. Exact sorted-token name match
    ↓
[Feature Engineering] ~50 features:
  - Name similarity (exact, fuzzy, Jaccard, Jaro-Winkler)
  - Address similarity (fuzzy, numeric overlap, token features)
  - Retrieval provenance (scores, ranks, channel count)
  - Cross-field interactions
    ↓
[LightGBM] Grouped 5-fold CV, hard negative mining
    ↓
[Threshold Optimization] Sweep for max macro F0.5
    ↓
[Test Inference] Batched per-country processing
    ↓
matching_results.tsv + candidate_pairs.tsv
```

## Key Design Decisions

1. **Country gating**: Same-country blocking only (validated: zero cross-country matches in training)
2. **Source-specific retrieval**: Separate top-K from S2 and S3 to prevent crowding
3. **Hard negative mining**: Top retrieval negatives, not random negatives
4. **Precision-first**: F0.5 weights precision 2x over recall; threshold is optimized accordingly
5. **8 GB RAM**: Per-country processing, chunked features, sparse matrices

## Directory Structure

```
baseline_pipeline/
├── scripts/
│   └── run_all.py          # Main entry point
├── src/                     # Modular source code
│   ├── data/               # Data loading, caching
│   ├── normalization/      # Text normalization
│   ├── blocking/           # TF-IDF + exact blocking
│   ├── features/           # Feature engineering
│   ├── models/             # LightGBM trainer
│   ├── validation/         # F0.5 evaluation
│   ├── calibration/        # Threshold optimization
│   └── inference/          # Test pipeline
├── config/                  # YAML configs
├── models/                  # Saved LightGBM models
├── output/                  # Final TSV outputs
├── reports/                 # Analysis reports
├── experiments/             # Experiment tracking
└── cache/                   # Parquet caches
```

## Output Files

- `output/matching_results.tsv` — Final scored submission
- `output/candidate_pairs.tsv` — Candidate set for blocking analysis
- `experiments/experiments.csv` — Experiment tracking log
- `models/` — Saved LightGBM models and config

## License
All models used are MIT/Apache-2.0 licensed and ≤ 8B parameters.
LightGBM: MIT License.
