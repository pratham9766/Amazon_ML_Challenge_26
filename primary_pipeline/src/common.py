"""Shared paths, normalization expressions, logging and experiment tracking.

All outputs of this package stay inside primary_pipeline/.
"""
from __future__ import annotations

import csv
import datetime as dt
import gc
import logging
import os
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "primary_pipeline"
DATA = ROOT / "dataset-20260925T050739Z-1-001" / "dataset"
VALIDATOR = ROOT / "resources-20260925T050740Z-1-001" / "resources" / "utils" / "validate_submission.py"
CACHE = OUT / "cache"
BLOCKER = OUT / "blocker"
REPORTS = OUT / "reports"
EXPER = OUT / "experiments"
LOGS = OUT / "logs"
SEED = 20260925

for _d in (CACHE, BLOCKER, REPORTS, EXPER, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return float("nan")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0


# ---------------------------------------------------------------- normalization
# Latin accents are folded (Ássets -> assets); marks on other scripts are kept
# because Indic vowel signs are combining marks and carry meaning.

def norm_expr(col: str) -> pl.Expr:
    return (
        pl.col(col).fill_null("")
        .str.normalize("NFKC").str.to_lowercase()
        .str.normalize("NFKD").str.replace_all(r"([a-z])\p{Mn}+", "$1").str.normalize("NFC")
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w]|_", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


# Alphanumeric address identifiers such as 8-2-351/N/3, 1325/1/K/4, H.No 10/A,
# G-93, ##45 -> compact codes (82351n3, 13251k4, 10a, g93, 45).
CODE_RE = r"[\p{L}\p{N}]*\p{N}[\p{L}\p{N}]*(?:[-/.][\p{L}\p{N}]+)*"


def codes_expr(col: str) -> pl.Expr:
    return (
        pl.col(col).fill_null("")
        .str.normalize("NFKC").str.to_lowercase()
        .str.extract_all(CODE_RE)
        .list.eval(pl.element().str.replace_all(r"[-/.]", ""))
        .list.unique(maintain_order=True)
        .list.join(" ")
    )


def source_path(split: str, src: int) -> Path:
    return CACHE / f"{split}_s{src}.parquet"


def load_source(split: str, src: int, columns=None, country: str | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet(source_path(split, src))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if columns is not None:
        lf = lf.select(columns)
    return lf.collect()


# ---------------------------------------------------------------- experiment log
EXP_FIELDS = [
    "experiment_id", "timestamp", "stage", "model", "features", "blocking_strategy", "candidate_cap",
    "fold_strategy", "threshold", "precision", "recall", "f0_5", "singleton_accuracy", "blocking_recall",
    "mean_candidates", "runtime", "RAM", "notes",
]


def log_experiment(**kw):
    path = EXPER / "experiment_log.csv"
    new = not path.exists()
    row = {k: kw.get(k, "") for k in EXP_FIELDS}
    row["timestamp"] = row["timestamp"] or dt.datetime.now().isoformat(timespec="seconds")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXP_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def free():
    gc.collect()
