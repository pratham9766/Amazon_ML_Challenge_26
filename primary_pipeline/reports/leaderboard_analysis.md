# Leaderboard analysis

## Status

No leaderboard upload was made from this workspace. The agent has no access to the competition portal, and uploading is an outward-facing action that belongs to the team. Every submission is produced as a validated, versioned folder (`submissions/submission_<id>/`, with a `manifest.json`) and logged in `experiments/leaderboard_log.csv`. The `public_lb_score` column is left empty for the team to fill.

## Submissions prepared

| id | model | rule | candidate strategy | offline (dev OOF / confirmation) | notes |
|---|---|---|---|---|---|
| S001 | M2 (LightGBM, 95 features) | t=0.69, singleton gate t_top=0.79, no conflict resolution | frozen comb_t@25 + name_na@3 | 0.9622 / 0.9607 | **primary**: every component validated offline |
| S002 | M2 | same + one-to-one (each target kept only for its max-p S1) | same | not measurable on the sparse benchmark | **experimental**: motivated by the ground-truth structure (0 targets shared between S1s) and 29.8k conflicted targets in France alone |

`submission_P001_M2_FR_USpartial` was produced by a helper that another session added (`src/make_partial_submission.py`) before India was ready. It covers France and part of the US only, so it should not be used as a final submission.

## How to read the LB against offline results

* Offline F0.5 is measured on train-distribution queries (US and India). The test set adds France (15%) and shifts the country mix: India 47% vs 40% in train. France looked behaviourally consistent (95% of S1s matched, 3.56 matches each, versus 93.8% and 3.44 on dev), but its precision is unmeasured. An LB below ~0.96 most likely reflects France or the mix, not overfitting.
* **Conflict resolution is the one decision the offline benchmark cannot settle.** S001 vs S002 on the public LB is therefore the informative comparison. If S002 beats S001 by more than a few thousandths, adopt one-to-one. If the gap is within about ±0.002, keep S001 and run `src/dense_eval.py`, which measures resolution on a dense region with labels, before deciding.
* Do not tune `t` against the LB. The OOF optimum is flat (t ∈ [0.64, 0.72] within 0.0002), so LB movements from threshold tweaks would be noise.
