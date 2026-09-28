# Amazon ML Challenge 2026 — Business Entity Resolution

An end-to-end, CPU-friendly entity-resolution system for matching Source 1 businesses to records in Sources 2 and 3.

> **Final public leaderboard score: 96.1%**
>
> Model: **M8 Ownership** · Metric: entity-level macro F0.5 · Amazon ML Challenge 2026

This repository contains the implementation, experiment history, and compact performance evidence. Raw competition data and rebuildable model artifacts remain local and are excluded from Git.

## Performance

| System | Evaluation | Macro F0.5 | Precision | Recall | Public leaderboard |
|---|---|---:|---:|---:|---:|
| **M8 Ownership (final)** | dense-region grouped CV | **0.9733** | — | — | **0.961 (96.1%)** |
| M5 Competition | dense-region grouped CV | 0.9707 | 0.9856 | 0.9367 | 0.959 |
| M2 | 20k grouped OOF | 0.9622 | 0.9802 | 0.9221 | 0.957 with one-to-one resolution |
| Gemini baseline | 10k grouped OOF | 0.9204 | 0.9565 | 0.8539 | — |

The final M8 system improved the public leaderboard score from 95.7% for M2 and 95.9% for M5 to **96.1%**. The frozen blocker reaches **97.14% pair recall** and **90.89% complete-entity recall** on the untouched 7.5k confirmation set with 52.6 candidates per Source 1 entity on average. Public leaderboard results are kept distinct from offline validation metrics.

For a compact comparison, see the [model card](MODEL_CARD.md). Full methodology and caveats are in the [main final report](primary_pipeline/reports/final_report.md), [leaderboard analysis](primary_pipeline/reports/leaderboard_analysis.md), and [experiment log](primary_pipeline/experiments/experiment_log.csv).

## Repository map

```text
analysis/          Reproducible dataset audit, metrics, and figures
blocking/          Initial blocking benchmark and ablation evidence
primary_pipeline/  Main pipeline, reports, frozen decisions, and metrics
baseline_pipeline/ Earlier baseline implementation and its reports
```

Large local-only directories—datasets, Parquet caches, trained LightGBM files, OOF predictions, candidate pairs, submissions, and logs—are intentionally excluded by `.gitignore`. They are reproducible from the tracked code or are competition-provided inputs.

## Approach

```text
Raw TSV records
      ↓
Unicode normalization and training-learned transliteration
      ↓
Country/source-aware TF-IDF blocking + missing-address rescue
      ↓
RapidFuzz pair features and retrieval context
      ↓
LightGBM matcher with grouped out-of-fold validation
      ↓
Ownership/competition features and one-to-one target resolution
      ↓
Officially validated submission
```

The system is designed for approximately 8 GB RAM and four CPU cores. Candidate generation is chunked and resumable, and pair features are computed only inside the bounded candidate pool.

## Main pipeline

Python 3.11 is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r primary_pipeline\requirements.txt
```

Place the official data at:

```text
dataset-20260925T050739Z-1-001/dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

The supplied validator is expected at `resources-20260925T050740Z-1-001/resources/utils/validate_submission.py`.

Run the production workflow from `primary_pipeline/src/`:

```powershell
python prep.py
python translit.py
python -c "from bench import build_truth; build_truth()"
python run_channels.py --queries bench --tag diag --channels comb_t --k 60 --batch 500
python run_channels.py --queries bench --tag diag --channels name_na --k 20
python pairs.py --split train --tag diag --qchunk 10000
python train_matcher.py --name M2 --lr 0.1
```

Test retrieval and submission generation are documented in [primary_pipeline/README.md](primary_pipeline/README.md). The pipeline uses only competition-provided data; its transliteration dictionary is learned from training pairs.

## What belongs in Git

Commit source, configuration, Markdown reports, compact JSON/CSV metrics, and small explanatory figures. Do not commit raw challenge data, caches, model weights, pair-level predictions, candidate tables, submission files, or runtime logs. If a future artifact is necessary but too large for ordinary Git, document how to reproduce it or use an external artifact store rather than weakening `.gitignore`.

