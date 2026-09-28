# Offline dataset analysis

This folder implements the requested dataset-analysis phase only. It reads the seven official TSVs and writes a Markdown report, JSON metrics, six PNG plots, a log, and rebuildable caches. It creates no classifier, semantic embeddings, final matching system, or submission.

## Reproduce

From the repository root, with Python 3.12 and the dependencies below:

```powershell
python analysis/run_eda.py --data-root dataset-20260925T050739Z-1-001/dataset
python analysis/check_methods.py
```

The exact offline command used in this workspace is:

```powershell
& 'C:\Users\Milind BOKEFODE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' analysis/run_eda.py --extra-site-packages 'D:\RESOURCES\NLP\ResumeAnalyzer\venv\Lib\site-packages'
```

The extra directory supplies locally installed SciPy/scikit-learn code only; no data is read from that project. The bundled runtime supplies pandas, NumPy and Pillow. Nothing is downloaded. On another machine, omit `--extra-site-packages` when these packages are already in the active environment.

Recorded versions: Python 3.12.14, pandas 3.0.1, NumPy 2.3.5, SciPy 1.18.0, scikit-learn 1.5.2, Pillow 12.3.0. RapidFuzz is optional. When absent, independently checked exact Indel/Levenshtein and token formulas provide the corresponding diagnostic signals; the report explicitly identifies that fallback.

## Stages and resources

- `--stage load`: chunked source ingestion and disk indexes.
- `--stage analyze`: resume source caches, validate labels, compute overlap, pair statistics and full-universe blocking probes, then report.
- `--stage report`: regenerate Markdown/JSON derived summaries and plots from saved metrics.
- `--stage all` (default): run every stage, reusing validated caches.

Defaults: seed 20260925; 50,000 rows per source ingestion chunk; 10,000 uniformly sampled rows per source/country; 1,200 country-proportional S1 anchors for pair/simple-block diagnostics; 120 of those anchors for expensive full-universe lexical TF-IDF retrieval. The last probe is deliberately small and carries cluster-bootstrap intervals.

The six source files are never loaded together. SQLite and its indexes use substantial **disk** space (allow approximately 15–25 GB). SQLite cache is capped at 128 MiB; lexical retrieval uses 10,000 target rows per batch and float32 sparse matrices. Sparse TF-IDF vocabulary/IDF fitting uses only the uniform training target sample. The retrieval search itself scans every S2/S3 training record. Runtime is dominated by exact disk grouping and full-universe retrieval.

`analysis/cache/` contains only generated data and may be removed for a clean rebuild. File caches validate input size/mtime and sampling configuration. `manifest.json` additionally tracks all seven inputs and run settings; derived caches invalidate on changes. A completed audit stores SHA-256 for every input. Originals are read-only in this workflow. Never load the pickle cache from an untrusted source.

## Definitions and limits

- Raw empty strings, whitespace-only strings, and pandas-default NA lexemes are distinguished. A nonempty business name such as an NA marker is preserved as text for diagnostics; pandas-default missingness does not establish semantic absence.
- Uniqueness/counts/ground-truth validation/country and field histograms are population statistics. Similarity/normalization/separation statistics are samples, reported with denominators.
- Negatives assume the supplied labels enumerate all true targets for each S1. Hard negatives are selected from a uniform sampled target corpus and are not representative of operational prevalence.
- All full-corpus blocking probes use same-country candidates and count every supplied true pair in the denominator. K is over combined S2+S3, not per source.
- Postal detection is only a standalone 5–6-digit pattern, not validation of a postal address.
- Normalization rules are hypotheses for diagnosis, not finalized production transformations.
- Similarity AUC is descriptive, not a trained classifier or a prediction of precision.

Read `dataset_report.md` for results and caveats; `eda_metrics.json` contains all requested percentiles, exact distributions, country/source strata, normalization collision rates, and candidate counts. The script is organized into loaders, normalization, ground truth, similarity, diagnostics, and reporting modules.
