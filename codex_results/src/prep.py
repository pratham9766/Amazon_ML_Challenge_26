"""Convert the seven official TSVs into compact Parquet caches (streaming, low RAM).

Per source: entity_id, name, addr, country, name_n, addr_n, codes, src.
Ground truth: long table (s1, tgt) plus s1 list with multiplicity.
"""
from __future__ import annotations

import polars as pl

from common import CACHE, DATA, codes_expr, get_logger, norm_expr, source_path, Timer

log = get_logger("prep")


def scan_tsv(path) -> pl.LazyFrame:
    return pl.scan_csv(
        path, separator="\t", quote_char=None, has_header=True, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def prep_source(split: str, src: int):
    out = source_path(split, src)
    if out.exists():
        log.info("exists %s", out.name)
        return
    t = Timer()
    lf = scan_tsv(DATA / split / f"{split}_source{src}.tsv").rename(
        {"business_name": "name", "business_address": "addr"}
    )
    lf = lf.with_columns(
        norm_expr("name").alias("name_n"),
        norm_expr("addr").alias("addr_n"),
        codes_expr("addr").alias("codes"),
        pl.lit(src, dtype=pl.Int8).alias("src"),
    )
    lf.sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    log.info("wrote %s rows=%d in %.0fs", out.name, n, t())


def prep_gt():
    out = CACHE / "train_gt_pairs.parquet"
    if out.exists():
        return
    gt = scan_tsv(DATA / "train" / "train_ground_truth.tsv").collect()
    gt = gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"))
    gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("m").list.len().alias("mult")).write_parquet(CACHE / "train_gt_s1.parquet")
    gt.select(pl.col("source1_entity_id").alias("s1"), pl.col("m").alias("tgt")).explode("tgt").drop_nulls().write_parquet(out)
    log.info("gt pairs written")


if __name__ == "__main__":
    for split in ("train", "test"):
        for src in (1, 2, 3):
            prep_source(split, src)
    prep_gt()
