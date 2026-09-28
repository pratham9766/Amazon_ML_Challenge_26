"""Run retrieval channels for a set of S1 queries against full same-country S2/S3.

Writes one parquet per channel with (s1, tgt, src, score, rank) where rank is
per-source (0-based). Used for the benchmark (queries = dev+conf) and for test.

usage: python run_channels.py --split train --queries bench --channels name_w12,addr_w12 --k 200
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl
from scipy import sparse

from common import BLOCKER, CACHE, free, get_logger, load_source, rss_mb, source_path
from retrieval import HashedTfidf, search

log = get_logger("channels")

# channel -> (field builder, hasher kind, max_df)
CHANNELS = {
    "name_w12": ("name", "word12", 0.02),
    "addr_w12": ("addr", "word12", 0.02),
    "comb_w12": ("comb", "word12", 0.02),
    "name_c34": ("name", "charwb34", 0.05),
    "addr_c34": ("addr", "charwb34", 0.05),
    "name_c23": ("name", "charwb23", 0.05),
    "codes": ("codes", "ws", 0.01),
    "comb_t": ("comb_t", "word12", 0.02),
    "name_t": ("name_t", "word12", 0.02),
    "name_na": ("name_t", "word12", 0.02),  # name index over address-less targets only
}
TARGET_FILTER = {"name_na": pl.col("addr_n") == ""}


NEED = {"name": ["name_n"], "addr": ["addr_n"], "comb": ["name_n", "addr_n"], "codes": ["codes"],
        "comb_t": ["name_n", "addr_n"], "name_t": ["name_n"]}
_TR = None


def translated(df: pl.DataFrame, cols) -> pl.DataFrame:
    """Targets only: replace native-script tokens using the train-learned dictionary."""
    global _TR
    if _TR is None:
        from translit import load_dict
        _TR = load_dict()
    from translit import translate_expr
    nm, am = _TR
    out = df.with_columns(translate_expr("name_n", nm).alias("name_n"))
    if "addr_n" in cols:
        out = out.with_columns(translate_expr("addr_n", {**nm, **am}).alias("addr_n"))
    return out


def field(df: pl.DataFrame, f: str, target: bool = False) -> list[str]:
    if f.endswith("_t"):
        if target:
            df = translated(df, NEED[f])
        f = f[:-2]
    if f == "name":
        return df["name_n"].to_list()
    if f == "addr":
        return df["addr_n"].to_list()
    if f == "comb":
        return (df["name_n"] + " " + df["addr_n"]).to_list()
    if f == "codes":
        return df["codes"].to_list()
    raise ValueError(f)


def chunked_fit(model: HashedTfidf, texts, chunk=500_000):
    """Fit IDF + transform in chunks to limit peak RAM."""
    raws = []
    for i in range(0, len(texts), chunk):
        raws.append(model._raw(texts[i:i + chunk]))
    del texts
    X = sparse.vstack(raws, format="csr")
    del raws
    n = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1]).astype(np.float32)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    idf[df > max(model.max_df * n, 50)] = 0.0
    model.idf = idf.astype(np.float32)
    model.df = df
    return model._weight(X)


def get_queries(split: str, which: str) -> pl.DataFrame:
    cols = ["entity_id", "name_n", "addr_n", "codes", "country"]
    lf = pl.scan_parquet(source_path(split, 1)).select(cols)
    if which in ("bench", "extra", "dense", "more", "dense2"):
        b = pl.scan_parquet(CACHE / f"{which}_queries.parquet").select("s1")
        lf = lf.join(b, left_on="entity_id", right_on="s1", how="semi")
    return lf.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--queries", default="bench")
    ap.add_argument("--channels", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--srcs", default="2,3")
    ap.add_argument("--qshard", default="", help="i/n: only queries with row index %% n == i (parallel workers)")
    ap.add_argument("--countries", default="", help="restrict to these countries; parts only (no final concat)")
    a = ap.parse_args()
    outdir = BLOCKER / a.tag
    outdir.mkdir(parents=True, exist_ok=True)
    Q = get_queries(a.split, a.queries)
    countries = sorted(Q["country"].unique().to_list())
    only = [x for x in a.countries.split(",") if x]
    if only:
        countries = [c for c in countries if c in only]
    log.info("queries=%d countries=%s", len(Q), countries)
    for ch in a.channels.split(","):
        out = outdir / f"{ch}.parquet"
        if out.exists():
            log.info("skip %s (exists)", ch)
            continue
        f, kind, max_df = CHANNELS[ch]
        parts, stats = [], {}
        t0 = time.time()
        for c in countries:
            q = Q.filter(pl.col("country") == c)
            if a.qshard:
                qi_, qn_ = map(int, a.qshard.split("/"))
                q = q.with_row_index("_r").filter(pl.col("_r") % qn_ == qi_).drop("_r")
            qtexts = field(q, f)
            for src in [int(x) for x in a.srcs.split(",")]:
                t1 = time.time()
                part = outdir / (f"_part_{ch}_{c}_S{src}.parquet" if not a.qshard
                                 else f"_shard_{ch}_{c}_S{src}_{a.qshard.replace('/', 'of')}.parquet")
                if part.exists():
                    parts.append(part)
                    log.info("resume %s %s S%d", ch, c, src)
                    continue
                T = load_source(a.split, src, ["entity_id", "addr_n"] + [x for x in NEED[f] if x != "addr_n"], country=c)
                if ch in TARGET_FILTER:
                    T = T.filter(TARGET_FILTER[ch])
                model = HashedTfidf(kind, max_df=max_df)
                TM = chunked_fit(model, field(T, f, target=True))
                T = T.select("entity_id")  # drop text once indexed
                free()
                QM = model.transform(qtexts)
                nnz = TM.nnz
                TT = TM.T.tocsr()
                del TM
                qi, ti, sc, rk = search(QM, TT, a.k, batch=a.batch)
                res = pl.DataFrame({
                    "s1": q["entity_id"].gather(qi), "tgt": T["entity_id"].gather(ti),
                    "src": np.full(len(qi), src, np.int8), "score": sc.astype(np.float32), "rank": rk.astype(np.int16),
                })
                res.write_parquet(part)
                parts.append(part)
                del res, qi, ti, sc, rk
                stats[f"{c}/S{src}"] = dict(n_targets=len(T), nnz=int(nnz), secs=round(time.time() - t1, 1), rss_mb=round(rss_mb()))
                log.info("%s %s S%d targets=%d nnz=%d %.0fs rss=%.0fMB", ch, c, src, len(T), nnz, time.time() - t1, rss_mb())
                del T, TT, QM, model
                free()
        if only:
            continue  # worker mode: leave parts for the main process to concatenate
        pl.scan_parquet(parts).sink_parquet(out)
        for pf in outdir.glob(f"_part_{ch}_*.parquet"):
            pf.unlink()
        json.dump(dict(channel=ch, secs=round(time.time() - t0, 1), per=stats), open(outdir / f"{ch}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
