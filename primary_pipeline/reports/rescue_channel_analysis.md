# Rescue-channel analysis (development 20k)

All numbers are dev-split pair recall over all 69,268 true pairs. Candidate counts are per S1 (S2+S3 combined). `rescue_efficiency` = additional true pairs recovered per additional candidate generated (an engineering metric, not the competition metric). All experiments are logged in `experiments/experiment_log.csv` (ids B001–B005).

## 1. Channel comparison (per-source top-k, union over S2+S3)

| channel @k/source | pair recall | complete entity | mean cands |
|---|---|---|---|
| previous Codex RRF(address_char, name_char), cap 150 | 79.74 | 60.98 | 123.7 |
| addr_w12 @10 | 80.90 | 56.08 | 20 |
| name_w12 @10 | 58.76 | 28.50 | 20 |
| codes @10 (structured/rare address ids) | 21.65 | 10.63 | 17.7 |
| **comb_w12 @10** | **93.76** | 82.03 | 20 |
| comb_w12 @20 | 95.40 | 86.42 | 40 |
| all four @20 | 96.45 | 89.18 | 123.4 |
| all four @50 | 97.56 | 92.36 | 307.9 |

The IDF-weighted **address** channel already acts as the "rare address token" retriever that Phase 3 asked for. Common tokens (road, street, india, near) get low IDF, and tokens in more than 2% of a country's corpus are dropped. The **structured-address/codes** channel (compact identifiers such as `8-2-351/N/3 → 82351n3`, `H.No 10/A → 10a`) is precise but has low coverage. Its rescue efficiency over the backbone is 0.0006, so it was **discarded**.

## 2. Fusion vs union

Fusing the four channels through RRF, min-rank or weighted score sums was **worse** than the combined channel alone at every budget. For example, RRF top-20/source gave 95.08% against 95.40% for comb_w12@20. Fusion was discarded.

## 3. Rescue slices on top of the comb backbone

| config | pair | complete | mean | rescue eff. vs comb@20 |
|---|---|---|---|---|
| comb_w12@20 | 95.40 | 86.42 | 40.0 | – |
| + name_w12@5 | 95.54 | 86.79 | 45.5 | 0.0009 |
| + addr_w12@5 | 95.56 | 86.80 | 41.3 | 0.0041 |
| + codes@5 | 95.53 | 86.74 | 48.1 | 0.0006 |
| + name@5 + addr@5 | 95.70 | 87.16 | 46.9 | 0.0015 |

## 4. Unicode, accent and transliteration rescue (Phase 6)

* Normalization is NFKC plus lowercasing. Accents are folded **only on Latin letters** (`Ássets → assets`, `Cíty → city`). Combining marks of Indic scripts are preserved, because they are vowel signs.
* No transliteration library was installed. Instead, a **native-script → Latin token dictionary was learned from training match pairs only**: positional alignment of equal-length S1/target names gave 1,347 entries across Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Gurmukhi, Odia and Malayalam. Address co-occurrence mapping (≥60% support, n≥20) gave 17 state/region entries. The dictionary is applied to targets as an extra representation (`comb_t`), and the original text is kept for features.

| config | pair | complete | non-ASCII target names | India |
|---|---|---|---|---|
| comb_w12@20 | 95.40 | 86.42 | 89.81 | 93.62 |
| **comb_t@20** | **96.45** | **88.81** | **97.89** | **96.27** |

This gives +1.05 points of recall at **zero extra candidates**. It is the largest single gain after the TF-IDF fix.

## 5. Missing-target-address rescue (Phase 11)

`name_na` is a name word-TF-IDF index built only over targets whose address is empty (about 3% of targets), queried with the S1 name.

| config | pair | complete | missing-addr subgroup | mean | eff. |
|---|---|---|---|---|---|
| comb_t@25 | 96.78 | 89.70 | 82.66 | 50.0 | – |
| + name_na@2 | 96.97 | 90.30 | 86.82 | 51.5 | 0.004 |
| **+ name_na@3** | **97.00** | **90.41** | **87.56** | **52.6** | 0.003 |
| + name_na@5 | 97.06 | 90.60 | 88.88 | 55.4 | 0.002 |
| comb_t@30 + name_na@3 | 97.21 | 91.00 | 87.66 | 62.5 | 0.001 |

## 6. Phases considered and not pursued (evidence-based)

* **Dynamic multiplicity caps (Phase 9):** multiplicity has no rank penalty under per-source top-k (see rank analysis), so there was nothing to fix.
* **Character n-gram retrieval:** profiled at about 15× the matmul cost and about 100× the top-k cost of word n-grams. On test (1.73M queries × ~2M targets per country) that is infeasible on 4 cores and 8 GB. Character similarity is used in the matcher instead.
* **Conditional router (Phase 12):** the two surviving rescues are already conditional by construction (non-Latin text is translated; the address-less index only contains address-less targets). No further routing had a measurable target.
