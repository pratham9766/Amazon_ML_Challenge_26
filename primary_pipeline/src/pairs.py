"""Candidate pairs (frozen blocker) -> feature parquet parts, chunked by queries.

usage: python pairs.py --split train --tag diag   (benchmark queries, labelled)
       python pairs.py --split test  --tag test   (all test queries)
"""
from __future__ import annotations

import argparse
import json

import polars as pl

from common import BLOCKER, CACHE, OUT, Timer, free, get_logger, rss_mb, source_path
from features import build
from translit import load_dict, translate_expr

log = get_logger("pairs")
FEAT = OUT / "features"
FEAT.mkdir(exist_ok=True)
FROZEN = json.load(open(BLOCKER / "frozen_blocker.json"))["channels"]  # {"comb_t": 25, "name_na": 3}


def candidates(tag: str, country: str = "") -> pl.DataFrame:
    src = BLOCKER / tag / "comb_t.parquet"
    if country:  # per-country mode from resumable retrieval parts
        src = [BLOCKER / tag / f"_part_comb_t_{country}_S{s}.parquet" for s in (2, 3)]
        src = [p for p in src if p.exists()] or [BLOCKER / tag / "comb_t.parquet"]
    cb = pl.scan_parquet(src).filter(pl.col("rank") < FROZEN["comb_t"]) \
        .select("s1", "tgt", "src", pl.col("score").alias("comb_score"), pl.col("rank").alias("comb_rank"))
    na = pl.scan_parquet(BLOCKER / tag / "name_na.parquet").filter(pl.col("rank") < FROZEN["name_na"]) \
        .select("s1", "tgt", "src", pl.col("score").alias("na_score"), pl.col("rank").alias("na_rank"))
    if country:
        cq = pl.scan_parquet(source_path("test" if tag == "test" else "train", 1)).filter(pl.col("country") == country).select(pl.col("entity_id").alias("s1"))
        cb = cb.join(cq, on="s1", how="semi")
        na = na.join(cq, on="s1", how="semi")
    c = cb.join(na, on=["s1", "tgt", "src"], how="full", coalesce=True)
    return c.with_columns(
        pl.col("comb_score").fill_null(0.0), pl.col("na_score").fill_null(0.0),
        pl.col("comb_rank").fill_null(99).cast(pl.Float32), pl.col("na_rank").fill_null(99).cast(pl.Float32),
        pl.col("src").cast(pl.Float32).alias("src_f"),
    )


def s1_table(split: str) -> pl.DataFrame:
    s1 = pl.scan_parquet(source_path(split, 1)).select("entity_id", "name_n", "addr_n", "codes", "country").collect()
    rep = s1.group_by("country", "name_n").agg(pl.len().alias("name_rep"))
    return s1.join(rep, on=["country", "name_n"]).with_columns(pl.col("name_rep").cast(pl.Float32))


def target_table(split: str, ids: pl.DataFrame) -> pl.DataFrame:
    nm, am = load_dict()
    parts = []
    for src in (2, 3):
        t = pl.scan_parquet(source_path(split, src)).select("entity_id", "name_n", "addr_n", "codes") \
            .join(ids.lazy(), left_on="entity_id", right_on="tgt", how="semi").collect()
        parts.append(t.with_columns(translate_expr("name_n", nm).alias("name_n"),
                                    translate_expr("addr_n", {**nm, **am}).alias("addr_n")))
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--qchunk", type=int, default=100_000)
    ap.add_argument("--country", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: process only chunks with index %% n == i")
    ap.add_argument("--no_cand", action="store_true", help="skip writing the candidates file (shard workers)")
    a = ap.parse_args()
    t = Timer()
    outdir = FEAT / (a.split if a.tag in ("diag", "test") else f"{a.split}_{a.tag}")
    outdir.mkdir(exist_ok=True)
    cand = candidates(a.tag, a.country)
    if a.country:  # materialize once; per-chunk reads are then cheap predicate scans
        mat = outdir / f"_cand_{a.country}.parquet"
        if not mat.exists():
            tmp = mat.with_suffix(".tmp")
            cand.collect().sort("s1").write_parquet(tmp, row_group_size=200_000)
            tmp.rename(mat)
        cand = pl.scan_parquet(mat)
    pre = f"part_{a.country}_" if a.country else "part_"
    s1 = s1_table(a.split)
    qids = cand.select(pl.col("s1").alias("entity_id")).unique().collect()
    log.info("candidate queries %d", len(qids))
    if a.split == "train":
        gt = pl.scan_parquet(CACHE / "train_gt_pairs.parquet")
        cand = cand.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1", "tgt"], how="left") \
            .with_columns(pl.col("label").fill_null(0))
    ids = qids["entity_id"].sort()
    for i, s in enumerate(range(0, len(ids), a.qchunk)):
        out = outdir / f"{pre}{i:03d}.parquet"
        si, sn = map(int, a.shard.split("/"))
        if out.exists() or i % sn != si:
            continue
        chunk_ids = ids.slice(s, a.qchunk).to_frame("s1")
        c = cand.filter(pl.col("s1").is_in(chunk_ids["s1"].implode())).collect()
        tg = target_table(a.split, c.select("tgt").unique())
        f = build(c, s1.join(chunk_ids.rename({"s1": "entity_id"}), on="entity_id", how="semi"), tg,
                  score_cols=["comb_score", "na_score"])
        f.write_parquet(out)
        log.info("part %d: %d pairs, %.0fs, rss=%.0fMB", i, len(f), t(), rss_mb())
        del c, tg, f
        free()
    if a.no_cand:
        return
    # candidate_pairs for the whole split (needed for test submission)
    cand.select("s1", "tgt").sink_parquet(outdir / (f"candidates_{a.country}.parquet" if a.country else "candidates.parquet"))


if __name__ == "__main__":
    main()
