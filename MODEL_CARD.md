# Model card

## Task

Match each Source 1 business to zero or more records in Sources 2 and 3. Evaluation uses entity-level macro F0.5, so precision is weighted more heavily than recall and correct empty predictions for singleton entities matter substantially.

## Data and validation

- Inputs: competition-provided business name, address, country, entity ID, and training labels.
- External data: none.
- Split discipline: grouped folds by Source 1 entity; the blocker also has an untouched 7.5k-entity confirmation set.
- Primary metric: macro F0.5 computed per Source 1 entity, with singletons included.
- Leakage control: transliteration mappings are learned only from training pairs; country is used for candidate gating but not as a matcher feature.

## Model lineage

| Model | Role | Validation scope | F0.5 | Precision | Recall | Singleton accuracy |
|---|---|---|---:|---:|---:|---:|
| Gemini | Earlier LightGBM baseline, 37 pair features | 10k grouped OOF | 0.9204 | 0.9565 | 0.8539 | 0.8995 |
| M1 | First main LightGBM matcher, 79 features | 20k grouped OOF | 0.9581 | 0.9780 | 0.9143 | 0.9339 |
| M2 | Main matcher, 84 features and singleton gate | 20k grouped OOF | 0.9622 | 0.9802 | 0.9221 | 0.9612 |
| M2S2 | Stage-2 model on M2 probabilities and context | 20k grouped OOF | 0.9620 | 0.9787 | 0.9246 | 0.9595 |
| M5 | Competition stage on M2 | dense-region grouped CV | 0.9707 | 0.9856 | 0.9367 | 0.9759 |
| M6 | Competition stage on M3F | dense-region grouped CV | 0.9717 | 0.9862 | 0.9383 | 0.9785 |
| M8 | Ownership stage on M3F | dense-region grouped CV | **0.9733** | — | — | — |

Metrics come from `codex_results/experiments/experiment_log.csv` and the frozen matcher result JSON files. Dense-region scores are not directly comparable with the 20k OOF scores because their evaluation populations differ.

## Candidate generation

The frozen blocker combines:

- hashed word 1–2 gram TF-IDF over normalized name and address, top 25 per source;
- a training-learned native-script-to-Latin transliteration dictionary; and
- a name-only rescue channel for targets with missing addresses, top 3.

On the untouched confirmation set it obtains 97.14% pair recall, 90.89% complete-entity recall, and 52.6 candidates per query on average (cap 56).

## Decision rules and reported leaderboard performance

M2 uses a pair threshold of 0.69 and predicts no matches unless the query’s best candidate reaches 0.79. Later systems add claimant competition or ownership features and one-to-one target assignment.

| Submission | Offline evidence | Public leaderboard |
|---|---|---:|
| S002 M2 + one-to-one | M2 OOF F0.5 0.9622 | 0.957 |
| S005 M5 competition | dense-region F0.5 0.9707 | 0.959 |
| S008 M8 ownership | dense-region F0.5 about 0.9732 | **0.961** |

Leaderboard scores are user-reported. The one-to-one rule has strong structural motivation because training targets map to at most one Source 1 entity, but sparse benchmark conflicts make its standalone offline effect difficult to estimate.

## Known limitations

- France has no labeled training population, so its precision is inferred from behavior rather than directly measured.
- Offline numbers across the 20k benchmark and dense competition regions have different evaluation populations.
- The unseen test-country mix differs from training.
- The repository does not version raw data, trained weights, predictions, or submissions; reproduce them locally from the tracked code and official inputs.
