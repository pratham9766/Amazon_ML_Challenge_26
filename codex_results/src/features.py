"""Pairwise features for (S1, candidate) pairs inside the bounded candidate pool.

Input pairs frame: s1, tgt, src + retrieval columns (score/rank per channel).
All string similarities use RapidFuzz cpdist (vectorized C++, multi-threaded).
Country is NOT a feature (France is unseen); only generic text signals are used.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

from bench import SUFFIX

import os
W = int(os.environ.get("RF_WORKERS", "-1"))  # rapidfuzz workers (-1 = all cores)


def _cp(a, b, scorer, scale=100.0):
    return (process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32) / scale).astype(np.float32)


def _set_feats(qs: pl.Series, ts: pl.Series, prefix: str) -> dict:
    """Token-set overlap features from space-joined token strings."""
    ql = qs.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    tl = ts.str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    inter = pl.DataFrame({"q": ql, "t": tl}).select(
        pl.col("q").list.set_intersection("t").list.len().alias("i"),
        pl.col("q").list.len().alias("nq"), pl.col("t").list.len().alias("nt"))
    i, nq, nt = (inter[c].to_numpy().astype(np.float32) for c in ("i", "nq", "nt"))
    union = nq + nt - i
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            f"{prefix}_jacc": np.where(union > 0, i / union, np.nan),
            f"{prefix}_cov_q": np.where(nq > 0, i / nq, np.nan),
            f"{prefix}_cov_t": np.where(nt > 0, i / nt, np.nan),
            f"{prefix}_inter": i, f"{prefix}_nq": nq, f"{prefix}_nt": nt,
        }


def text_features(df: pl.DataFrame) -> pl.DataFrame:
    """df has q_name, t_name, q_addr, t_addr, q_codes, t_codes (normalized strings)."""
    qn, tn = df["q_name"].to_list(), df["t_name"].to_list()
    qa, ta = df["q_addr"].to_list(), df["t_addr"].to_list()
    core = lambda s: s.str.replace_all(SUFFIX, "").str.replace_all(r"\s+", " ").str.strip_chars()
    qc, tc = core(df["q_name"]), core(df["t_name"])
    qcl, tcl = qc.to_list(), tc.to_list()
    f = {
        "n_ratio": _cp(qn, tn, fuzz.ratio),
        "n_partial": _cp(qn, tn, fuzz.partial_ratio),
        "n_tset": _cp(qn, tn, fuzz.token_set_ratio),
        "n_tsort": _cp(qn, tn, fuzz.token_sort_ratio),
        "n_jw": _cp(qn, tn, distance.JaroWinkler.normalized_similarity, 1.0),
        "n_lev": _cp(qn, tn, distance.Levenshtein.normalized_similarity, 1.0),
        "nc_ratio": _cp(qcl, tcl, fuzz.ratio),
        "nc_tset": _cp(qcl, tcl, fuzz.token_set_ratio),
        "nc_partial": _cp(qcl, tcl, fuzz.partial_ratio),
        "a_ratio": _cp(qa, ta, fuzz.ratio),
        "a_partial": _cp(qa, ta, fuzz.partial_ratio),
        "a_tset": _cp(qa, ta, fuzz.token_set_ratio),
        "a_tsort": _cp(qa, ta, fuzz.token_sort_ratio),
    }
    f["n_eq"] = (df["q_name"] == df["t_name"]).to_numpy().astype(np.float32)
    f["nc_eq"] = ((qc == tc) & (qc != "")).to_numpy().astype(np.float32)
    f.update(_set_feats(df["q_name"], df["t_name"], "nt"))
    f.update(_set_feats(qc, tc, "nct"))
    f.update(_set_feats(df["q_addr"], df["t_addr"], "at"))
    f.update(_set_feats(df["q_codes"], df["t_codes"], "code"))
    t_missing = (df["t_addr"].str.strip_chars() == "").to_numpy()
    for k in list(f):
        if k.startswith("a_") or k.startswith("at_"):
            f[k] = np.where(t_missing, np.nan, f[k]).astype(np.float32)
    f["t_addr_missing"] = t_missing.astype(np.float32)
    nqc, ntc, ic = f["code_nq"], f["code_nt"], f["code_inter"]
    f["code_conflict"] = ((nqc > 0) & (ntc > 0) & (ic == 0)).astype(np.float32)
    f["code_all_t_in_q"] = np.where(ntc > 0, (ic == ntc).astype(np.float32), np.nan)
    # house/building number agreement (near-twin decoys differ by a few digits)
    qnum = df["q_addr"].str.extract_all(r"\d+")
    tnum = df["t_addr"].str.extract_all(r"\d+")
    nd = pl.DataFrame({"q": qnum, "t": tnum}).select(
        pl.col("q").list.first().str.strip_chars_start("0").alias("q1"),
        pl.col("t").list.first().str.strip_chars_start("0").alias("t1"),
        pl.col("q").list.eval(pl.element().str.strip_chars_start("0")).list.unique().alias("qs"),
        pl.col("t").list.eval(pl.element().str.strip_chars_start("0")).list.unique().alias("ts"))
    nd = nd.with_columns(
        pl.col("qs").list.set_difference("ts").list.len().alias("q_only"),
        pl.col("ts").list.set_difference("qs").list.len().alias("t_only"),
        pl.col("ts").list.set_intersection("qs").list.len().alias("both"),
        pl.col("t1").is_in(pl.col("qs")).alias("t1_in_q"))
    has = (nd["q1"].is_not_null() & nd["t1"].is_not_null()).to_numpy()
    f["num_first_eq"] = np.where(has, (nd["q1"] == nd["t1"]).fill_null(False).to_numpy(), np.nan).astype(np.float32)
    f["num_t1_in_q"] = np.where(has, nd["t1_in_q"].fill_null(False).to_numpy(), np.nan).astype(np.float32)
    f["num_q_only"] = nd["q_only"].to_numpy().astype(np.float32)
    f["num_t_only"] = nd["t_only"].to_numpy().astype(np.float32)
    f["num_both"] = nd["both"].to_numpy().astype(np.float32)
    f["q_name_len"] = df["q_name"].str.len_chars().to_numpy().astype(np.float32)
    f["t_name_len"] = df["t_name"].str.len_chars().to_numpy().astype(np.float32)
    f["name_len_diff"] = np.abs(f["q_name_len"] - f["t_name_len"])
    f["t_name_nonlatin"] = df["t_name"].str.contains(r"[^\x00-\x7f]").to_numpy().astype(np.float32)
    f["q_core_tokens"] = f["nct_nq"]
    # interactions
    f["name_x_addr"] = f["n_tset"] * np.nan_to_num(f["a_tset"], nan=0.5)
    f["strong_name_missing_addr"] = f["n_tset"] * f["t_addr_missing"]
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


def group_features(df: pl.DataFrame, score_cols: list[str]) -> pl.DataFrame:
    """Per-query context of the retrieval scores (singleton / multiplicity signals)."""
    exprs = []
    for c in score_cols:
        exprs += [
            pl.col(c).max().over("s1").alias(f"{c}_qmax"),
            (pl.col(c) - pl.col(c).max().over("s1")).alias(f"{c}_gap"),
            pl.col(c).rank("average", descending=True).over("s1").alias(f"{c}_qrank"),
            (pl.col(c) >= 0.8 * pl.col(c).max().over("s1")).sum().over("s1").alias(f"{c}_n_near"),
        ]
    exprs.append(pl.len().over("s1").alias("q_ncand"))
    return df.select(exprs)


def second_order(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per-query relative position of the model-free similarity signals."""
    exprs = []
    for c in cols:
        exprs += [
            (pl.col(c) - pl.col(c).max().over("s1")).alias(f"{c}_gap"),
            (pl.col(c) >= pl.col(c).max().over("s1") - 1e-6).cast(pl.Float32).alias(f"{c}_is_top"),
            (pl.col(c) >= 0.9).sum().over("s1").cast(pl.Float32).alias(f"{c}_n90"),
        ]
    return df.select(exprs)


def build(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, score_cols: list[str], chunk: int = 1_000_000) -> pl.DataFrame:
    """pairs: s1,tgt,src,<score cols>,<rank cols>. s1/tg: entity_id,name_n,addr_n,codes (tg translated)."""
    d = (pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name_n").alias("q_name"),
                              pl.col("addr_n").alias("q_addr"), pl.col("codes").alias("q_codes"),
                              pl.col("name_rep").alias("q_name_rep")), on="s1", how="left")
         .join(tg.select(pl.col("entity_id").alias("tgt"), pl.col("name_n").alias("t_name"),
                         pl.col("addr_n").alias("t_addr"), pl.col("codes").alias("t_codes")), on="tgt", how="left")
         .with_columns(pl.col(c).fill_null("") for c in ("q_name", "q_addr", "q_codes", "t_name", "t_addr", "t_codes")))
    outs = []
    for s in range(0, len(d), chunk):
        outs.append(text_features(d.slice(s, chunk)))
    tf = pl.concat(outs)
    base = pl.concat([d.drop("q_name", "q_addr", "q_codes", "t_name", "t_addr", "t_codes"), tf], how="horizontal")
    base = pl.concat([base, group_features(base, score_cols),
                      second_order(base, ["n_tset", "a_tset", "nc_ratio", "code_cov_q", "name_x_addr"])], how="horizontal")
    return base
