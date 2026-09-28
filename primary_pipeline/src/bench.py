"""Benchmark truth tables and blocker evaluation utilities."""
from __future__ import annotations

import numpy as np
import polars as pl

from common import CACHE, load_source

SUFFIX = r"\b(private|pvt|limited|ltd|llc|inc|incorporated|corp|corporation|company|co|pc|lp|llp|plc|pllc|sarl|sas|sa|eurl|gmbh)\b"


def build_truth():
    """Per-pair and per-query diagnostic tables for the 27.5k benchmark queries."""
    qp, pp = CACHE / "bench_q.parquet", CACHE / "bench_pairs.parquet"
    if qp.exists() and pp.exists():
        return pl.read_parquet(qp), pl.read_parquet(pp)
    b = pl.read_parquet(CACHE / "bench_queries.parquet")
    s1 = load_source("train", 1, ["entity_id", "name", "name_n", "addr_n", "country"])
    # repeated normalized name within S1 (same country)
    rep = s1.group_by("country", "name_n").agg(pl.len().alias("name_rep"))
    q = b.join(s1, left_on="s1", right_on="entity_id").join(rep, on=["country", "name_n"])
    core = pl.col("name_n").str.replace_all(SUFFIX, "").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    q = q.with_columns(
        core.list.len().alias("core_tokens"),
        pl.col("name").str.contains(r"[^\x00-\x7f]").alias("q_nonascii"),
    ).with_columns(
        (pl.col("core_tokens") <= 1).alias("short_name"),
        (pl.col("name_rep") > 1).alias("repeated_name"),
    )
    gt = pl.read_parquet(CACHE / "train_gt_pairs.parquet").join(b.select("s1"), on="s1", how="semi")
    parts = []
    for src in (2, 3):
        t = load_source("train", src, ["entity_id", "name", "addr", "name_n", "addr_n"])
        parts.append(gt.join(t, left_on="tgt", right_on="entity_id").with_columns(pl.lit(src, pl.Int8).alias("src")))
    p = pl.concat(parts).rename({"name": "t_name", "addr": "t_addr", "name_n": "t_name_n", "addr_n": "t_addr_n"})
    p = p.join(q.select("s1", "split", "country", "mult", "name_n", "addr_n", "short_name", "repeated_name"), on="s1")
    qtok = pl.col("name_n").str.split(" ")
    ttok = pl.col("t_name_n").str.split(" ")
    p = p.with_columns(
        pl.col("t_name").str.contains(r"[^\x00-\x7f]").alias("t_name_nonascii"),
        pl.col("t_addr").str.contains(r"[^\x00-\x7f]").alias("t_addr_nonascii"),
        (pl.col("t_addr").str.strip_chars() == "").alias("t_addr_missing"),
        (qtok.list.set_intersection(ttok).list.len() == 0).alias("no_shared_name_token"),
    )
    q.write_parquet(qp)
    p.drop("t_name", "t_addr").write_parquet(pp)
    return q, p


def subgroup_exprs():
    return {
        "all": pl.lit(True),
        "country=US": pl.col("country") == "US",
        "country=India": pl.col("country") == "India",
        "source=S2": pl.col("src") == 2,
        "source=S3": pl.col("src") == 3,
        "mult=1": pl.col("mult") == 1,
        "mult=2": pl.col("mult") == 2,
        "mult=3": pl.col("mult") == 3,
        "mult=4+": pl.col("mult") >= 4,
        "mult=6+": pl.col("mult") >= 6,
        "t_name_nonascii": pl.col("t_name_nonascii"),
        "t_addr_nonascii": pl.col("t_addr_nonascii"),
        "t_addr_missing": pl.col("t_addr_missing"),
        "no_shared_name_token": pl.col("no_shared_name_token"),
        "short_name": pl.col("short_name"),
        "repeated_name": pl.col("repeated_name"),
    }


def evaluate(cands: pl.DataFrame, q: pl.DataFrame, p: pl.DataFrame, split="dev", groups=True) -> dict:
    """cands: (s1, tgt) unique candidate pairs. Returns metric dict."""
    qs = q.filter(pl.col("split") == split)
    ps = p.filter(pl.col("split") == split)
    c = cands.join(qs.select("s1"), on="s1", how="semi").unique(["s1", "tgt"])
    hit = ps.join(c.with_columns(pl.lit(True).alias("hit")), on=["s1", "tgt"], how="left").with_columns(pl.col("hit").fill_null(False))
    per_q = hit.group_by("s1").agg(pl.col("hit").all().alias("complete"), pl.col("hit").any().alias("any"))
    cnt = qs.select("s1", "mult").join(c.group_by("s1").agg(pl.len().alias("n")), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
    n = cnt["n"].to_numpy()
    res = dict(
        queries=len(qs), true_pairs=len(ps), pair_recall=float(hit["hit"].mean()),
        complete_entity_recall=float(per_q["complete"].mean()), at_least_one_recall=float(per_q["any"].mean()),
        mean=float(n.mean()), median=float(np.median(n)), p90=float(np.percentile(n, 90)), p95=float(np.percentile(n, 95)),
        p99=float(np.percentile(n, 99)), max=int(n.max()), total_candidates=int(n.sum()),
        singleton_mean=float(cnt.filter(pl.col("mult") == 0)["n"].mean() or 0),
    )
    if groups:
        g = {}
        for name, e in subgroup_exprs().items():
            h = hit.filter(e)
            if len(h):
                pq = h.group_by("s1").agg(pl.col("hit").all())
                g[name] = dict(pairs=len(h), pair_recall=float(h["hit"].mean()), complete=float(pq["hit"].mean()))
        res["groups"] = g
    res["_hit"] = hit
    return res


RANK_BUCKETS = [(0, 20, "<=20"), (20, 50, "21-50"), (50, 100, "51-100"), (100, 150, "101-150"), (150, 200, "151-200")]


def rank_bucket_expr(col="rank"):
    e = pl.when(pl.col(col).is_null()).then(pl.lit("not_in_top200"))
    for lo, hi, name in RANK_BUCKETS:
        e = e.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(pl.lit(name))
    return e.otherwise(pl.lit(">200"))
