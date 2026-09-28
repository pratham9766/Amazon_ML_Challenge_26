# Candidate-generation benchmark

This phase builds and evaluates blocking only. It does not train a matcher, infer test matches, use leaderboard feedback, or create a submission.

Run from the repository root:

```powershell
& 'C:\Users\Milind BOKEFODE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' blocking/run_blocking_benchmark.py
```

The script reuses `analysis/cache/audit.sqlite`, whose contents were validated against the seven official TSV files. Generated channel candidates are cached under `blocking/artifacts/`; reruns resume completed channels. Remove a particular `candidates_<channel>.json.gz` only when that channel's implementation or configuration changes.

The fixed benchmark has 20,000 development queries and 7,500 untouched confirmation queries. All retrieval channels search every training S2/S3 record and apply generic same-country gating. Development selects the channel order and final cap. Confirmation is evaluated once by the script after selection.

The environment has no FAISS, HNSW, `sparse_dot_topn`, DuckDB, PyArrow, or Polars. Consequently, the scalable lexical channels use streaming, query-conditioned sparse inverted retrieval with bounded IDF-weighted features. The report keeps the prior classical TF-IDF probe as a baseline and clearly distinguishes the scalable approximation. Gzip JSON/TSV and SQLite replace Parquet because no Parquet engine is installed offline.

Important outputs:

- `reports/blocking_report.md`: benchmark, channel results, ablation, selected configuration, subgroup recall and efficiency.
- `reports/missed_positive_analysis.md`: failure categories and examples.
- `reports/blocker_ablation.csv`: machine-readable incremental union results.
- `artifacts/benchmark_summary.json`: complete metrics.
- `artifacts/confirmation_candidate_provenance.tsv.gz`: selected confirmation candidates with ranks, scores and channel provenance.

The output is a research benchmark and candidate-generator candidate. It is not a claim about final macro F0.5.
