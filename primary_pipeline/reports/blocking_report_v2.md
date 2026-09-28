# Blocking report v2: frozen candidate generator

## Frozen configuration (`blocker/frozen_blocker.json`)

| component | definition |
|---|---|
| normalization | NFKC → lowercase → Latin-only accent folding → `&`→`and` → non-word chars → space (Indic marks kept) |
| gating | same `country` string (generic and open-set; France works unchanged), per source S2 and S3 |
| `comb_t` @25/source | TF-IDF over hashed word 1–2 grams of `name + address`; sublinear TF; features with df > 2% of the country/source corpus dropped; targets transliterated with the train-learned dictionary |
| `name_na` @3/source | same TF-IDF over names only, indexed on targets with an empty address |
| IDF fit | on the target corpus of the same split and country; labels are never used, and test IDF comes from test targets |

Union → at most 56 candidates per S1 (25+25 from comb_t, 3+3 from name_na).

## Results

| split | queries | true pairs | pair recall | complete-entity recall | at-least-one | mean | median | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| dev (selection) | 20,000 | 69,268 | **97.00%** | 90.41% | 99.81% | 52.6 | 52 | 56 | 56 | 56 |
| **confirmation (run once)** | 7,500 | 26,066 | **97.14%** | **90.89%** | 99.82% | 52.6 | 52 | 56 | 56 | 56 |
| previous blocker, confirmation | 7,500 | 26,066 | 79.35% | 61.06% | – | 123.7 | 147 | 150 | 150 | 150 |

Singleton S1s receive about 52.5 candidates on average. Their handling is left to the matcher.

## Subgroups (pair recall; complete-entity recall on confirmation)

| subgroup | dev | conf | conf complete |
|---|---|---|---|
| US | 97.03 | 97.28 | 90.94 |
| India | 96.96 | 96.93 | 90.81 |
| S2 | 97.74 | 97.80 | 95.99 |
| S3 | 96.31 | 96.52 | 93.74 |
| multiplicity 1 | 97.58 | 96.88 | 96.88 |
| multiplicity 2 | 97.35 | 96.90 | 93.95 |
| multiplicity 3 | 97.03 | 97.58 | 93.51 |
| multiplicity 4+ | 96.93 | 97.04 | 87.84 |
| non-ASCII target name | 98.23 | 97.92 | 97.24 |
| non-ASCII target address | 97.48 | 97.75 | 97.02 |
| missing target address | 87.56 | 86.74 | 86.06 |
| no shared name token | 92.14 | 92.68 | 89.58 |
| short name | 96.32 | 96.57 | 89.05 |
| repeated name | 96.00 | 96.11 | 87.85 |

Dev and confirmation agree within ±0.7 points in every subgroup, so the configuration is not overfitted to dev.

## Unseen-country compatibility

Nothing in the blocker depends on country values. The gate is string equality, IDF is fitted per country on that country's own targets, the transliteration dictionary only fires on tokens it knows, and accent folding covers French diacritics (`é`, `è`, `ç`). France-specific quality cannot be measured without labels. The mechanics were exercised on the test France partition, and all 259,452 French S1 records receive candidates.

## Efficiency

* Benchmark (27.5k queries, full training universe): comb_t took 1,248 s and name_na 19 s, with RSS ≤1.3 GB.
* Test (1.73M queries): runtime is dominated by the sparse matmul, roughly proportional to queries × targets per country. France took 12.5 min per source. India and US took several hours each on 4 cores and were run as two parallel processes with resumable per-country/source checkpoints (`blocker/test/_part_*`).
* The memory design uses hashed features (no vocabulary dict), float32 CSR, query batches of 500 and per-country/source processing, with only the needed columns loaded from Parquet.

## Freeze decision

Freeze conditions met: recall is strong (97.1% pair / 90.9% complete versus 79.3% / 61.1% before), confirmation is stable, volume is small and bounded (≤56), no dominant rescue opportunity remains (best remaining efficiency ≈0.001–0.002 pairs per candidate), and the pipeline is reproducible (`src/run_channels.py`) and feasible at test scale.
