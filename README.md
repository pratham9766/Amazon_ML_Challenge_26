# Amazon ML Challenge 2026 — Business Entity Resolution

CPU-friendly entity resolution for matching Source 1 businesses to records in Sources 2 and 3. The repository keeps the implementation, experiment history, and compact performance evidence in Git; raw data and rebuildable model artifacts remain local.

## Results

| System | Evaluation | Macro F0.5 | Precision | Recall | Notes |
|---|---:|---:|---:|---:|---|
| M8 ownership | dense-region grouped CV | **0.9733** | — | — | Best offline experiment; stage on M3F with one-to-one resolution |
| M5 competition | dense-region grouped CV | 0.9707 | 0.9856 | 0.9367 | Public leaderboard **0.959** |
| M2 | 20k grouped OOF | 0.9622 | 0.9802 | 0.9221 | Confirmation F0.5 0.9607 |
| Gemini baseline | 10k grouped OOF | 0.9204 | 0.9565 | 0.8539 | Earlier reference implementation |

The frozen blocker reaches **97.14% pair recall** and **90.89% complete-entity recall** on the untouched 7.5k confirmation set with 52.6 candidates per Source 1 entity on average. Leaderboard values are user-reported and are kept distinct from offline validation metrics.

For a compact comparison, see the [model card](MODEL_CARD.md). Full methodology and caveats are in the [main final report](codex_results/reports/final_report.md), [leaderboard analysis](codex_results/reports/leaderboard_analysis.md), and [experiment log](codex_results/experiments/experiment_log.csv).

## Repository map

```text
analysis/          Reproducible dataset audit, metrics, and figures
blocking/          Initial blocking benchmark and ablation evidence
codex_results/     Main pipeline, reports, frozen decisions, and metrics
gemini_solution/   Earlier baseline implementation and its reports
```

Large local-only directories—datasets, Parquet caches, trained LightGBM files, OOF predictions, candidate pairs, submissions, and logs—are intentionally excluded by `.gitignore`. They are reproducible from the tracked code or are competition-provided inputs.

## Main pipeline

Python 3.11 is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r codex_results\requirements.txt
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

Run the production workflow from `codex_results/src/`:

```powershell
python prep.py
python translit.py
python -c "from bench import build_truth; build_truth()"
python run_channels.py --queries bench --tag diag --channels comb_t --k 60 --batch 500
python run_channels.py --queries bench --tag diag --channels name_na --k 20
python pairs.py --split train --tag diag --qchunk 10000
python train_matcher.py --name M2 --lr 0.1
```

Test retrieval and submission generation are documented in [codex_results/README.md](codex_results/README.md). The pipeline uses only competition-provided data; its transliteration dictionary is learned from training pairs.

## What belongs in Git

Commit source, configuration, Markdown reports, compact JSON/CSV metrics, and small explanatory figures. Do not commit raw challenge data, caches, model weights, pair-level predictions, candidate tables, submission files, or runtime logs. If a future artifact is necessary but too large for ordinary Git, document how to reproduce it or use an external artifact store rather than weakening `.gitignore`.

