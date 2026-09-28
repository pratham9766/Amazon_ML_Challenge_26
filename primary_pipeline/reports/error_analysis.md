# Error analysis (M1 OOF → M2)

Scope: the 20,000 dev queries, OOF probabilities, frozen rule t = 0.67 (M1).

## Where the loss comes from

Loss is Σ(1 − F0.5) over queries.

| segment | queries | mean F0.5 | share of total loss |
|---|---|---|---|
| singletons (mult 0) | 1,135 | 0.934 | 9.0% |
| mult 1 | 1,116 | 0.897 | 13.8% |
| mult 2–3 | 8,067 | 0.958 | 40.9% |
| mult 4+ | 9,682 | 0.969 | 36.4% |
| US | 12,077 | 0.958 | 60.2% |
| India | 7,923 | 0.958 | 39.8% |
| short names | 399 | 0.946 | 2.6% |
| repeated S1 names | 7,903 | 0.952 | 45.0% |

* **False positives:** 944 pairs over 879 queries. Those queries carry **33.6% of all loss**, because a single FP zeroes a singleton and cuts precision for everyone else.
* **False negatives:** 6,095, split into 2,079 (34%) the **blocker never generated** and 4,016 the **model rejected**. Of the rejected ones, 852 had p in (0.5, 0.67] and 1,306 had p ≤ 0.1.
* India and US perform equally (0.958). Repeated or generic names are the hardest query type.

## High-confidence false positives

These are the precision-critical cases. The top of the FP list is dominated by **near-twin decoy entities**:

* One-letter name edits with the same address: `Sisely Rice, CPA` → `Sisely Rice, CPF`, `Vantola Family Practice` → `Vantoli …`, `Kritesh International` → `Critesh …`, `Lumiq` → `Lumuq`.
* Same name with a **slightly different house number**: `1206 Business Loop 70` vs `1209 …`, `2261 Yale Avenue` vs `2266 …`, `813 Buckeye Rd` vs `0822 …`, `25657 Redwood Hwy` vs `25661 …`.
* Singletons hit by a perfect-looking twin: `Bangalore Labs Private Limited` and `Physical Therapy Golden Group`, both with p > 0.99.

Many of these are indistinguishable from genuine corruptions using pair text alone. Numeric disagreement, however, is a real signal.

**Action taken (M2):** house-number agreement features (`num_first_eq`, `num_t1_in_q`, `num_q_only`, `num_t_only`, `num_both`). Effect on dev OOF:
* FP 944 → 825
* singleton accuracy 0.934 → 0.961
* F0.5 0.9581 → 0.9622

The same direction held on confirmation (0.9570 → 0.9607).

## Model-rejected false negatives (sample)

* **Target without an address and a lightly edited name** (`Bhubaneswar Network Private Ltd`, `Anand Marketing`, `Cascade Program Llc`), p = 0.01–0.6. With no address evidence the model is correctly cautious, since F0.5 penalises the alternative.
* **Replaced name, same address** (`Ghaziabad Parks Private Limited` → `Lumdrex`, `Hughes, Smoot & Barton LLC` → `Zetayumacalo`): the name is a random brand token. These are recoverable only from the address, and accepting them would admit many same-address decoys.
* **Partial address plus changed number** (`45 Dyke Road, Colonie` vs `45A Dyke Rd, Latham`).

## Remaining opportunities (not implemented)

1. **Target-conflict resolution.** Ground truth assigns each target to at most one S1. On the France test partition alone, 29,792 predicted targets are claimed by more than one S1, so each carries at least one guaranteed FP. The sparse dev benchmark (1% of S1) cannot measure this. `src/dense_eval.py` evaluates argmax and margin resolution on a dense region (every S1 in KS/WV/Punjab/Odisha). It was not run to completion within the RAM and time budget, so resolution is **not** applied in the primary submission.
2. More training queries (M3 was prepared with 40k extra queries and their features cached in `features/train_extra/`) and a stage-2 context model (M2S2, neutral on OOF).
3. Recall on address-less targets (87% blocker recall; cautious model).
