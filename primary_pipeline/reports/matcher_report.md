# Matcher report

## Setup

* **Candidates:** the frozen blocker (`comb_t@25 + name_na@3` per source; ≤56 per S1; 97.0% dev pair recall). The same code path runs on train and test.
* **Training data:** all blocker candidates of the 20,000 dev queries (1.05M pairs, ~6.6% positive). Every negative is a *top-ranked retrieval neighbour*, so the training set is hard-negative by construction.
* **Validation:** `GroupKFold(5)` by S1 on dev, producing out-of-fold probabilities. All decisions (features, rule, threshold) were made on **OOF macro F0.5** using the exact entity-level metric (`src/metric.py`). The metric counts every dev S1, including singletons (empty prediction = 1.0) and true pairs the blocker missed (they count as recall loss).
* **Holdout:** the 7,500 confirmation queries were never used for selection. Each model's frozen rule is applied once to a model trained on all dev data.
* **Model:** LightGBM (binary, 63 leaves, min_data_in_leaf 50, feature/bagging fraction 0.8, early stopping per fold). The final model uses 1.1 × the mean best iteration. No large hyperparameter search was run.

## Features (`src/features.py`; all RapidFuzz `cpdist`, computed only inside the candidate pool)

* **Name:** ratio, partial_ratio, token_set, token_sort, Jaro-Winkler and normalized Levenshtein on the full name and on the legal-suffix-reduced core. Also exact / core-exact equality, token Jaccard and coverage (both directions), lengths and length difference. Targets use the transliterated name.
* **Address:** ratio, partial, token_set, token_sort, token Jaccard and coverage. These are NaN when the target address is missing, and a missing flag is kept.
* **Structured address:** code (alphanumeric id) overlap, Jaccard and coverage, a code-conflict flag, and all target codes present in the query. *(M2)* House-number agreement: first number equal, first target number contained in the query, and counts of numbers only in the query, only in the target, and in both.
* **Retrieval:** comb_t score and rank, name_na score and rank, source.
* **Query context:** per-query max, gap-to-max, rank and near-max count for each retrieval score, plus candidate count. Per-query gap, is-top and n≥0.9 for the key similarities. S1 name repetition count (chains or generic names).
* **Interactions:** `name_tset × addr_tset` and `strong name × missing address`.
* **Excluded:** `country` is not a feature (France is unseen) and IDs are never used.

## Results (all logged in `experiments/experiment_log.csv`)

| model | change | OOF F0.5 | P | R | singleton acc | fold range | **confirmation F0.5** | conf singleton |
|---|---|---|---|---|---|---|---|---|
| M1 | baseline, 81 features, lr 0.05 | 0.95812 | 0.9780 | 0.9143 | 0.9339 | 0.9555–0.9609 | 0.95702 | 0.9323 |
| **M2** | + house-number agreement, lr 0.1 | **0.96216** | 0.9802 | 0.9221 | **0.9612** | 0.9601–0.9646 | **0.96069** | 0.9649 |
| M2S2 | stage 2 on OOF stage-1 p + query context | 0.96199 | 0.9787 | 0.9246 | 0.9595 | 0.9604–0.9634 | 0.96136 | 0.9649 |
| M3 | M2 + 40k extra training queries | not completed: stopped to free RAM for test retrieval under the 8 GB limit | | | | | | |

Precision and recall are macro averages. A correctly empty singleton counts as P=R=1.

**Selected: M2.** M2S2's OOF is not better (−0.00017) and its confirmation gain (+0.0007) is within noise. Under the promotion rule, the extra stage is not kept.

## Entity-level decision logic

Swept on OOF (`train_matcher.sweep`): a global threshold t ∈ [0.20, 0.90] in 0.05 steps, then a fine ±0.05 search in 0.01 steps, crossed with a relative rule (keep p ≥ r·max_p, r ∈ {0, 0.2, 0.4}) and a singleton gate (the query predicts nothing unless max_p ≥ t_top).

* M1 best: t = 0.67 with no gate. The relative rule never helped (r = 0/0.2/0.4 tie).
* **M2 best: t = 0.69 with singleton gate t_top = 0.79.** Singleton accuracy is 0.961 OOF and 0.965 on confirmation.
* The optimum is flat: t ∈ [0.64, 0.72] stays within 0.0002 F0.5, and fold scores span only 0.0045. The threshold is not knife-edge.
* 0.5 would not be optimal: F0.5 favours precision, so the optimal cut sits well above it.

## Feature importance (M2, gain; top)

`name_x_addr` ≫ `num_t_only` > `comb_score_qrank` > `name_x_addr_gap` > `comb_rank` > `num_q_only` > `n_tsort` > `nc_partial` > `nc_tset` > `nt_jacc`. The new house-number features rank 2nd and 6th, confirming the error analysis that motivated them (near-twin decoys that differ in digits).

## Unseen country (France)

France was scored with M2 as a behavioural check (no labels exist). 94.96% of French S1s receive at least one match, with 3.56 matches per matched S1. Dev OOF gives 93.78% and 3.44, and the dev truth is 94.3% non-empty. The model does not collapse or over-fire on the unseen country.

## Reproduce

`python src/pairs.py --split train --tag diag --qchunk 10000` → `python src/train_matcher.py --name M2 --lr 0.1`
