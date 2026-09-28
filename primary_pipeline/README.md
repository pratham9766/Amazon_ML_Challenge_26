# Business Entity Resolution: primary pipeline

Only competition-provided data is used. There are no external lookups, geocoding or registries, and nothing is downloaded. All outputs stay inside `primary_pipeline/`.

Environment: Python 3.11 and `requirements.txt`. It runs on about 8 GB RAM with 4 CPU cores (hashed sparse TF-IDF, chunked Parquet, bounded candidate pools). Run the commands from `primary_pipeline/src/` with `PYTHONIOENCODING=utf-8`.

## End-to-end

```bash
python prep.py                                   # 7 TSV -> Parquet (+ normalized name/addr, address codes), ~5 min
python translit.py                               # learn native-script->Latin token dictionary from TRAIN pairs
# --- benchmark (dev 20k / confirmation 7.5k, reused from blocking/artifacts) ---
python -c "from bench import build_truth; build_truth()"
python run_channels.py --queries bench --tag diag --channels comb_t --k 60 --batch 500
python run_channels.py --queries bench --tag diag --channels name_na --k 20
python pairs.py --split train --tag diag --qchunk 10000          # features for bench candidates
python train_matcher.py --name M2 --lr 0.1                      # OOF GroupKFold(5) + rule sweep + confirmation
# --- test ---
python run_channels.py --split test --queries all --tag test --channels name_na --k 3 --batch 2000
python run_channels.py --split test --queries all --tag test --channels comb_t --k 25 --batch 500
#   (optional parallel workers: --countries US  /  --countries India --srcs 3 ; rerun without them to concatenate)
python pairs.py --split test --tag test --country France --qchunk 25000   # likewise India, US
python make_submission.py --model M2 --sub_id S001              # predicts, applies the frozen rule, runs the official validator
```

`make_submission.py` refuses to package a submission (exit 1) if `resources/utils/validate_submission.py` fails.

## Layout

| path | content |
|---|---|
| `src/prep.py`, `src/common.py` | ingestion, normalization expressions, paths, experiment log |
| `src/retrieval.py`, `src/run_channels.py` | hashed TF-IDF, batched sparse top-k, channels, resumable parts |
| `src/translit.py` | train-learned transliteration dictionary |
| `src/bench.py`, `src/analyze_channels.py`, `src/fusion.py` | blocker evaluation, rank diagnosis, ablations |
| `src/features.py`, `src/pairs.py` | RapidFuzz pair features, chunked by query |
| `src/metric.py` | official macro F0.5 (singletons included) |
| `src/train_matcher.py`, `src/stage2.py` | LightGBM OOF, decision-rule sweep, stage-2 experiment |
| `src/dense_eval.py` | one-to-one conflict evaluation on a dense region (prepared, not completed) |
| `src/make_submission.py` | test scoring, validator gate, versioned submission folder |
| `src/make_partial_submission.py` | helper found in the folder (not written by this session); partial-coverage submissions |
| `blocker/frozen_blocker.json` | frozen blocker definition; `frozen_blocker_eval.json` has dev and confirmation results |
| `matcher/M2_decision.json` | frozen features, threshold and singleton gate |
| `models/` | fold models and `*_full.txt` final models |
| `oof/` | out-of-fold predictions |
| `experiments/experiment_log.csv` | every blocker and matcher experiment (append-only) |
| `experiments/leaderboard_log.csv` | offline and reported public-leaderboard results for submitted models |
| `reports/` | rank failure, rescue channels, blocking v2, matcher, error analysis, leaderboard, final |
| `output/`, `submissions/submission_<id>/` | validated TSVs and a manifest |
