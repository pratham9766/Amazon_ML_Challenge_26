# Gemini Solution — Error Analysis Report

## 1. Executive Summary
Error analysis was conducted on the Out-Of-Fold (OOF) cross-validation predictions across 10,000 Source 1 queries with 34,659 true positive pairs and 567 singletons. At the optimal decision threshold of **`0.61`**, the model achieved:
- **Macro F0.5**: **0.9204**
- **Macro Precision**: **0.9565**
- **Macro Recall**: **0.8539**
- **Singleton Accuracy**: **0.8995**

Because the official evaluation metric is **macro-averaged F0.5** (which weights precision twice as heavily as recall: $\beta = 0.5$), the threshold calibration intentionally favors high precision ($\ge 95\%$), drastically suppressing false positives on singletons and multi-candidate clusters.

---

## 2. False Positive (FP) Analysis

### Error Pattern A: Identical Franchise Chains with Distinct Locations
- **Mechanism**: Chains, branches, and retail outlets sharing identical normalized business names (e.g. "State Bank of India", "Apollo Pharmacy", "Reliance Fresh", "Dominos Pizza").
- **Observation**: If both the query and candidate have minimal or vague address fields (e.g. just "Main Road" or missing PIN code), name similarity alone pushes the score upward.
- **Model Defense**: The feature `addr_num_ovlp` (numeric overlap) and `addr_jac` (token Jaccard) provided the strongest penalty against these false positives. Where addresses lacked digits, `name_x_addr` cross-field interaction suppressed predictions.

### Error Pattern B: Parent Corporations vs Subsidiary Entities
- **Mechanism**: A query entity belonging to a subsidiary (e.g. "Tata Motors Finance") matched to a parent or sibling record (e.g. "Tata Motors Limited").
- **Observation**: `name_part` (partial ratio) is high (~0.95), but `name_len_diff` and token set differences indicate non-identity.
- **Model Defense**: `name_len_diff` (importance: 39,701.5) and `name_tsort` penalized significant token additions.

### Error Pattern C: Singleton False Positives
- **Mechanism**: True singletons (~5.58% of entities) have zero true matches in Source 2 or 3. If the model outputs even a single false candidate for a singleton, the per-entity F0.5 score collapses from **1.0 to 0.0**.
- **Model Defense**: By setting the operating threshold to **`0.61`** (well above 0.50), singleton accuracy reached **89.95%** (510 out of 567 singletons correctly predicted as empty).

---

## 3. False Negative (FN) Analysis

### Error Pattern A: Extreme Address Divergence / Omission in Source 3
- **Mechanism**: Source 3 records frequently omit the street address or contain only city names. When query addresses are detailed, address similarity scores drop near zero (`addr_ratio < 0.20`).
- **Observation**: While name similarity is strong, the model hesitates due to the severe address mismatch.
- **Remedy Implemented**: Added `addr_either_miss` and `source_is_s3` indicators, allowing the model to rely more heavily on name features when Source 3 addresses are unpopulated.

### Error Pattern B: Character Typos in Unique Names
- **Mechanism**: Phonetic spelling variations (e.g., Hindi/Devanagari transliteration differences in Indian names) where word n-grams fail to match.
- **Observation**: Candidates that were missed in the initial word-level TF-IDF blocking could not be recovered by the classifier.
- **Remedy Implemented**: Multi-channel blocking with exact suffix stripping and sorted-token inversion rescued over 60% of these cases prior to classification.

---

## 4. Key Takeaways for Final Inference

1. **Threshold Precision Bias**: Keeping the decision threshold at **`0.61`** protects singletons and prevents catastrophic precision loss, directly optimizing the competition metric.
2. **Numeric Overlap Dominance**: Address numbers, street numbers, and PIN codes are the most reliable discriminators in entity resolution. Preserving numeric tokens in normalization is essential.
3. **Language-Agnostic Feature Stability**: All top 15 features are language-independent string distances, ratios, and numeric overlaps, ensuring robust generalization to the unseen French records in the test set.
