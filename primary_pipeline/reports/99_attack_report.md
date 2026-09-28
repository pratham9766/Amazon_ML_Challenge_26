# 0.99 attack report

All numbers are measured. Nothing here is estimated unless labelled as an estimate. The metric is the official per-S1 macro F0.5, with singletons included and blocker-missed true pairs counted as recall loss.

## 1. Baseline and leaderboard history

| submission | model | offline evidence | public LB |
|---|---|---|---|
| S002 | M2 + one-to-one | OOF 0.9622 (sparse dev), confirmation 0.9607 | **0.950** |
| **S005** | **M5: competition stage 2 on M2 + one-to-one** | dense 0.9707 vs 0.9665 for S002's method | **0.959** |
| **S008** | **M8: ownership stage on M3F + one-to-one** | dense 0.9732 across both regions | **0.961** |

The offline gain from S002 to S005 was +0.0042 on the dense region; the LB gain was **+0.009**. The competition signal helps France (unlabelled, with 8–14× more conflicts than US/India) more than the labelled regions show.

## 2. Why sparse validation was misleading and dense validation is needed

The dev benchmark samples about 1% of S1s, so the S1s that compete for the same record are almost never present. Measured on training labels:
* **83.4%** of all candidate negatives are the true matches of *another* S1.
* 46% of M2's false positives belong to another S1.
* Dense regions (every S1 of the chosen states queried, as on test):
  * region 1: KS, WV, Punjab, Odisha — 62.5k S1s, 60.7k scored after excluding every S1 used to train any stage-1 model;
  * region 2: CT, OK, Bihar, Madhya Pradesh — 87k S1s.

## 3. Experiments (same region-1 queries = "subset B", 60,651 S1s; grouped 5-fold CV by S1)

| model | what | dense F0.5 | P | R | singleton | decision |
|---|---|---|---|---|---|---|
| M2 (best rule) | pair model | 0.96683 | | | | baseline |
| M3F (best rule) | M2 + 40k extra training queries | 0.97001 | | | | stage-1 upgrade |
| M5B | competition stage on M2 (+1:1) | 0.97078 | 0.9863 | 0.9349 | 0.9797 | = S005 method |
| **M6B** | competition stage on M3F (+1:1) | **0.97169** | 0.9862 | 0.9383 | 0.9785 | +0.0009 vs M5B: **below the +0.001 gate** |
| R2 | 2nd competition round on M6B probabilities | 0.97143 | 0.9850 | 0.9409 | 0.9773 | **rejected** (worse than M6B) |
| **M8** | ownership model on M3F: rich claimant tables (top-3 claimant p, margins, rank, n≥0.5/0.7/0.9, mean/std, unique), trained on regions 1+2 (~148k S1s), iso calibration, threshold 0.68, one-to-one | **0.97298** | 0.9874 | 0.9396 | 0.9808 | **+0.0013 vs M6B → promoted (S008)** |

Earlier sparse-benchmark ladder (dev 20k OOF → untouched confirmation 7.5k): M1 0.9581 → M2 0.9622 (confirmation 0.9607). M2S2, a query-context stage 2, was not promoted. M1/M2/M2S2 blends gave +0.0004 at most and were not promoted. The extra-data A/B was +0.0034 and +0.0047 on folds 0/1, so M3F was promoted.

## 4. Error budget (M5B, subset B, 60,651 S1s, total loss = Σ(1 − F0.5) = 1,772)

| category | S1s | share of loss |
|---|---|---|
| true-match S1 predicted **empty** | 385 | **21.7%** |
| missed pairs never generated (**blocker**) | 4,633 | **25.9%** |
| missed pairs **rejected by the model** | 6,559 | **29.5%** |
| false positives (non-singletons) | 1,376 | 19.0% |
| singleton false positives | 70 | 3.9% |

Pair counts: FP 1,476 · FN (blocker) 5,642 · FN (model) 8,700.
* Model-rejected true pairs: 47.8% have a target with no address, 48.0% a repeated S1 name, 31.2% a house-number conflict.
* False positives: 28.9% house-number conflict, 29.4% repeated S1 name, 26.7% exact core-name match (near-twins).

**Precision is essentially solved (0.986). About 77% of the remaining loss is recall.**

## 5. The 0.99 question (evidence, not opinion)

* **Oracle ceiling with the frozen blocker** (a perfect matcher, zero errors on the candidates): **0.9916** on subset B, and 0.9908/0.9909 on sparse dev/confirmation.
* Our best model is 0.9717 on the same queries. Reaching 0.99 therefore needs **~95% of all remaining matcher error removed and the blocker improved.** The blocker-missed share alone (25.9% of loss) exceeds the gap budget between 0.99 and the ceiling.
* Stated plainly, per the brief: **0.99 is not supported by the evidence for this pipeline.** Top-LB teams at 0.98–0.99 must have both a higher-recall blocker and near-perfect ownership resolution. The first requires re-running test retrieval with larger candidate sets (≈6+ hours on this machine), which did not fit before the deadline.

## 6. Plan items not executed, and why

* **Reverse target→S1 retrieval:** its cost is test-retrieval scale (10M targets). Forward claimant tables cover every S1 that retrieved the record, which captures the same ownership competition among plausible owners.
* **300–500k-S1 retraining, leave-one-country-out, synthetic-corruption likelihoods:** they did not fit the remaining time alongside the gated runs.

## 7. Final submission

**S008 = M8 + isotonic calibration + threshold 0.68 + global one-to-one** → `primary_pipeline/output/` (copy in `submissions/submission_S008_M8_ownership/`).
The official validator passed on matching_results.tsv, and the streaming candidate check passed with 0 errors.
Test output: 1,629,667 matched S1s and 5,710,090 predicted pairs. The candidates are unchanged (frozen blocker, 92.7M pairs).

M8 details (grouped 5-fold OOF):

| subset | threshold rule F0.5 | expected-F0.5 selector F0.5 | selector singleton acc |
|---|---|---|---|
| both regions (iso) | 0.97320 (singletons 0.9787) | 0.97334 | 0.9633 |
| region 1 | 0.97296 | 0.97301 | 0.9637 |
| region 2 | 0.97337 | 0.97357 | 0.9630 |

* **Calibration:** isotonic (cross-fitted on OOF) changed F0.5 by ≤0.0001, so the raw scores were already calibrated. Iso was kept because it is harmless.
* **Expected-F0.5 list selection:** +0.0001 to +0.0002 over the threshold rule, but singleton accuracy fell from 0.979 to 0.963. **Rejected for singleton safety and robustness in the unseen country.**
* **Singleton logic:** re-optimized from scratch. The best rule on M8 needs no separate top-candidate gate (t_top = none), and singleton accuracy is 0.979.

**Public leaderboard result for S008:** **0.961 (96.1%)**. Before submission, the score was estimated at 0.962–0.964 from the offline gain over S005; the observed result was slightly below that narrow estimate while still improving on S005's 0.959.

**Conclusion on 0.99:** the full structural stack reached 0.9730 offline, against a hard ceiling of 0.9916 with this blocker. By the brief's own criterion (cannot exceed ~0.98 offline), **0.99 is not supported by evidence for this pipeline**. The next levers would be a higher-recall blocker (26% of the remaining loss) and recall on address-less targets (48% of model-rejected positives).
