# Final report: Business Entity Resolution (codex_results)

## Summary

| stage | metric | previous state | this work |
|---|---|---|---|
| blocker (confirmation 7.5k, untouched) | pair recall | 79.35% | **97.14%** |
| | complete-entity recall | 61.06% | **90.89%** |
| | mean candidates / S1 | 123.7 (cap 150) | **52.6 (cap 56)** |
| matcher M2 (dev 20k, grouped OOF) | macro F0.5 | – | **0.9622** (P 0.980, R 0.922) |
| | singleton accuracy | – | **0.961** |
| matcher M2 (confirmation, untouched) | macro F0.5 | – | **0.9607** (singleton 0.965) |

Only competition data is used. There are no external lookups; dictionaries are learned from training pairs only. Country is handled as an open string label and is never used as a feature, so France runs through the same code path.

## Methodology

1. **Ingestion** (`prep.py`): streaming Polars TSV → Parquet (quote-free, empty strings preserved). Normalization is NFKC, lowercase, Latin-only accent folding (Indic vowel signs kept) and `&`→`and`. Compact alphanumeric address codes are extracted (`8-2-351/N/3 → 82351n3`).
2. **Blocking diagnosis:** 98.7% of true pairs were retrievable within the top 200 of a proper TF-IDF channel. The previous blocker lost recall to its approximate inverted index (6 features per query), not to the data. Rank buckets by subgroup show that high multiplicity is not a failure mode under per-source top-k (`blocker_rank_failure_analysis.md`).
3. **Blocker** (frozen, `blocking_report_v2.md`):
   * The backbone is exact sparse TF-IDF over hashed word 1–2 grams of name + address (`comb_t`), per country and per source, top 25.
   * Targets are transliterated with a **1,347-token native-script → Latin dictionary learned from aligned training pairs**. This takes non-ASCII target-name recall from 89.8% to 97.9% at zero extra candidates.
   * A **missing-address rescue** (`name_na`) adds a name index over address-less targets, top 3, raising that subgroup's recall from 82.7% to 87.6%.
   * Rejected on measured evidence: rank fusion (RRF, min-rank, weighted), the structured-code channel (0.0006 pairs per extra candidate), char-n-gram retrieval (≈15× cost, infeasible at test scale) and dynamic caps (no rank penalty to fix). See `rescue_channel_analysis.md`.
4. **Pair features** (`features.py`): about 95 features computed with RapidFuzz `cpdist`, only inside the candidate pool. They cover name (full and suffix-reduced), address, codes, house-number agreement, retrieval scores and ranks, per-query context and interactions.
5. **Matcher** (`matcher_report.md`): LightGBM, `GroupKFold(5)` by S1, OOF-driven choices. Every blocker candidate is a hard negative by construction. Error analysis showed FPs are near-twin decoys (one-letter name edits, off-by-a-few house numbers). The added numeric features cut FPs 944 → 825 and raised singleton accuracy 0.934 → 0.961.
6. **Entity decision:** a global threshold t = 0.69 plus a **singleton gate** (a query predicts nothing unless its best candidate has p ≥ 0.79). Both were chosen on OOF macro F0.5 with the exact metric. The optimum is flat (±0.0002 over t ∈ [0.64, 0.72]), and fold F0.5 spans 0.9601–0.9646.
7. **One-to-one resolution (final submission):** ground truth assigns each target to at most one S1. On test, a noticeable fraction of predicted targets are claimed by several S1s (29.8k in France alone), and each such target carries a guaranteed false positive. The final submission keeps each target only for its highest-probability S1. This step could not be measured offline: the sparse benchmark has only 2 conflicts. `S001` (no resolution, fully OOF-validated) is packaged as the fallback, and the LB comparison is the deciding signal (`leaderboard_analysis.md`).

## Experiments considered and not promoted

| experiment | result | decision |
|---|---|---|
| Stage-2 model on OOF stage-1 p + query context (M2S2) | OOF 0.96199 vs 0.96216 | not promoted |
| Relative-to-max rule (keep p ≥ r·max) | no gain for any r | not used |
| 40k extra training queries (M3) | features built (`features/train_extra`), training stopped to free RAM for test retrieval | open |
| Dense-region conflict evaluation (`dense_eval.py`) | prepared, not run to completion | open |

## Runtime and RAM (4-core i3, 7.6 GB RAM, shared with other applications)

* **Prep:** 5 min.
* **Benchmark retrieval:** ~20 min per channel.
* **Training features:** 2 min per 1.4M pairs.
* **M2 training:** ~18 min, 5 folds.
* **Test retrieval:** matmul-bound, about queries × targets per country. France took 12.5 min per source; India and US took several hours each and ran as parallel resumable per-country/source parts. Running more than two retrieval processes at once caused paging and was counter-productive.
* **Test features:** about 100 s per 1.3M pairs with all cores.
* **Peak RSS:** about 1.3–1.6 GB per retrieval process (the lean version) and under 1 GB for feature parts.

## Risks

* France precision is unmeasured (no labels). Its behaviour is consistent with train (95.0% of S1s matched, 3.56 matches per matched S1, versus 93.8% and 3.44 on dev).
* One-to-one resolution is logically grounded but not measured offline (see above).
* The test mix is more India-heavy (47% vs 40%). India and US scored equally on OOF (0.958 each for M1).

## Reproduction

See `README.md`. In order: `prep.py` → `translit.py` → `run_channels.py` (bench + test) → `pairs.py` → `train_matcher.py --name M2 --lr 0.1` → `make_submission.py --model M2 --sub_id S002_M2_all_1to1 --one_to_one`. Every experiment is recorded in `experiments/experiment_log.csv`.
