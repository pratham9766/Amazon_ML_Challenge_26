"""Learn a native-script -> Latin token dictionary from TRAINING match pairs only.

Names: positional alignment of equal-length token sequences (S1 name vs target name).
Addresses: co-occurrence (a native token mapped to the S1 address token that
accompanies it in >= 60% of pairs with support >= 20); mainly state/city names.
Applied to targets as extra columns name_t / addr_t (originals are kept).
"""
from __future__ import annotations

import json

import polars as pl

from common import CACHE, get_logger, source_path

log = get_logger("translit")
NONLATIN = r"[^\x00-\x7f]"
DICT_PATH = CACHE / "translit_dict.json"


def pairs_frame() -> pl.LazyFrame:
    gt = pl.scan_parquet(CACHE / "train_gt_pairs.parquet")
    s1 = pl.scan_parquet(source_path("train", 1)).select(pl.col("entity_id").alias("s1"), pl.col("name_n").alias("q_name"), pl.col("addr_n").alias("q_addr"))
    tg = pl.concat([
        pl.scan_parquet(source_path("train", s)).select(pl.col("entity_id").alias("tgt"), "name_n", "addr_n")
        .filter(pl.col("name_n").str.contains(NONLATIN) | pl.col("addr_n").str.contains(NONLATIN))
        for s in (2, 3)
    ])
    return gt.join(tg, on="tgt").join(s1, on="s1")


def learn():
    df = pairs_frame().collect()
    log.info("pairs with non-ascii target text: %d", len(df))
    # names: positional alignment
    nm = df.filter(pl.col("name_n").str.contains(NONLATIN)).select(
        pl.col("name_n").str.split(" ").alias("t"), pl.col("q_name").str.split(" ").alias("q"))
    nm = nm.filter(pl.col("t").list.len() == pl.col("q").list.len())
    al = nm.with_row_index("i").explode(["t", "q"]).filter(pl.col("t").str.contains(NONLATIN) & ~pl.col("q").str.contains(NONLATIN))
    cnt = al.group_by("t", "q").len()
    tot = cnt.group_by("t").agg(pl.col("len").sum().alias("tot"))
    best = cnt.sort("len", descending=True).group_by("t").first().join(tot, on="t")
    best = best.filter((pl.col("len") >= 2) & (pl.col("len") / pl.col("tot") >= 0.5))
    name_map = dict(zip(best["t"].to_list(), best["q"].to_list()))
    log.info("name dictionary entries: %d (aligned pairs %d)", len(name_map), len(nm))
    # addresses: co-occurrence
    ad = df.filter(pl.col("addr_n").str.contains(NONLATIN)).with_row_index("i").select(
        "i", pl.col("addr_n").str.split(" ").list.unique().alias("t"), pl.col("q_addr").str.split(" ").list.unique().alias("q"))
    tt = ad.select("i", "t").explode("t").filter(pl.col("t").str.contains(NONLATIN))
    qq = ad.select("i", "q").explode("q").filter(~pl.col("q").str.contains(NONLATIN) & ~pl.col("q").str.contains(r"\d"))
    ntok = tt.group_by("t").len().rename({"len": "nt"}).filter(pl.col("nt") >= 20)
    tt = tt.join(ntok.select("t"), on="t", how="semi")
    co = tt.join(qq, on="i").group_by("t", "q").len()
    co = co.join(ntok, on="t").with_columns((pl.col("len") / pl.col("nt")).alias("p"))
    bestA = co.sort("p", descending=True).group_by("t").first().filter(pl.col("p") >= 0.6)
    addr_map = dict(zip(bestA["t"].to_list(), bestA["q"].to_list()))
    log.info("address dictionary entries: %d", len(addr_map))
    json.dump({"name": name_map, "addr": addr_map}, open(DICT_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    return name_map, addr_map


def load_dict():
    d = json.load(open(DICT_PATH, encoding="utf-8"))
    return d["name"], d["addr"]


def translate_expr(col: str, mapping: dict) -> pl.Expr:
    """Replace mapped native tokens; unmapped tokens are kept as-is."""
    keys = list(mapping.keys())
    vals = [mapping[k] for k in keys]
    return (pl.col(col).str.split(" ")
            .list.eval(pl.element().replace(keys, vals))
            .list.join(" "))


if __name__ == "__main__":
    nmap, amap = learn()
    print(list(nmap.items())[:30])
    print(list(amap.items())[:30])
