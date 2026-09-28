"""Score-fusion experiments over the cached channel pool (dev split only)."""
from __future__ import annotations

import argparse

import polars as pl

from bench import build_truth, evaluate
from common import BLOCKER, get_logger, log_experiment

log = get_logger("fusion")


def load_pool(tag: str, channels: list[str], max_rank: int = 60, keep=None) -> pl.DataFrame:
    parts = []
    for c in channels:
        d = pl.scan_parquet(BLOCKER / tag / f"{c}.parquet").filter(pl.col("rank") < max_rank)
        if keep is not None:
            d = d.join(keep.lazy(), on="s1", how="semi")
        parts.append(d.with_columns(pl.lit(c).alias("ch")).collect())
    long = pl.concat(parts)
    sc = long.pivot(on="ch", index=["s1", "tgt", "src"], values="score", aggregate_function="max").with_columns([pl.col(c).fill_null(0.0) for c in channels])
    rk = long.pivot(on="ch", index=["s1", "tgt", "src"], values="rank", aggregate_function="min")
    rk = rk.rename({c: f"r_{c}" for c in channels})
    return sc.join(rk, on=["s1", "tgt", "src"])


def fused(pool: pl.DataFrame, kind: str, channels) -> pl.Expr:
    if kind == "rrf":
        return pl.sum_horizontal([(1.0 / (60.0 + pl.col(f"r_{c}").cast(pl.Float64))).fill_null(0.0) for c in channels])
    if kind == "minrank":
        return -pl.min_horizontal([pl.col(f"r_{c}") for c in channels]).cast(pl.Float64)
    if kind.startswith("w:"):  # w:comb=1,addr=.5,...
        ws = dict(x.split("=") for x in kind[2:].split(","))
        return pl.sum_horizontal([pl.col(c).cast(pl.Float64) * float(w) for c, w in ws.items()])
    raise ValueError(kind)


def topn(pool: pl.DataFrame, expr: pl.Expr, n: int) -> pl.DataFrame:
    return (pool.with_columns(expr.alias("f"))
            .with_columns(pl.col("f").rank("ordinal", descending=True).over("s1", "src").alias("fr"))
            .filter(pl.col("fr") <= n).select("s1", "tgt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--channels", default="comb_w12,addr_w12,name_w12,codes")
    ap.add_argument("--kinds", default="rrf;minrank;w:comb_w12=1,addr_w12=0.5,name_w12=0.5,codes_=0")
    ap.add_argument("--ns", default="10,15,20,30")
    a = ap.parse_args()
    ch = a.channels.split(",")
    q, p = build_truth()
    dev = q.filter(pl.col("split") == "dev").select("s1")
    pool = load_pool(a.tag, ch, keep=dev)
    log.info("pool rows %d", len(pool))
    for kind in a.kinds.split(";"):
        kind = kind.replace("codes_", "codes")
        for n in map(int, a.ns.split(",")):
            r = evaluate(topn(pool, fused(pool, kind, ch), n), q, p, groups=False)
            msg = f"{kind} top{n}/src: pair={r['pair_recall']*100:.2f} complete={r['complete_entity_recall']*100:.2f} mean={r['mean']:.1f} p95={r['p95']:.0f}"
            print(msg, flush=True)
            log_experiment(experiment_id=f"B002_{kind}_{n}", stage="blocker-dev20k", blocking_strategy=f"fusion {kind} over {a.channels} (top200 pool)",
                           candidate_cap=2 * n, blocking_recall=round(r["pair_recall"], 5), mean_candidates=round(r["mean"], 1),
                           notes=f"complete={r['complete_entity_recall']:.4f}")


if __name__ == "__main__":
    main()
