"""Official entity-level macro F0.5 (per S1, singletons included)."""
from __future__ import annotations

import polars as pl


def macro_f05(pred: pl.DataFrame, truth_q: pl.DataFrame, truth_pairs: pl.DataFrame) -> dict:
    """pred: (s1, tgt) predicted matches; truth_q: s1 + mult for ALL evaluated queries;
    truth_pairs: (s1, tgt) all true pairs of those queries (incl. ones the blocker missed)."""
    tp = pred.join(truth_pairs, on=["s1", "tgt"], how="semi").group_by("s1").agg(pl.len().alias("tp"))
    npred = pred.group_by("s1").agg(pl.len().alias("np"))
    d = (truth_q.select("s1", pl.col("mult").cast(pl.Float64))
         .join(npred, on="s1", how="left").join(tp, on="s1", how="left")
         .with_columns(pl.col("np").fill_null(0).cast(pl.Float64), pl.col("tp").fill_null(0).cast(pl.Float64)))
    P = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    R = pl.when(pl.col("mult") > 0).then(pl.col("tp") / pl.col("mult")).otherwise(0.0)
    d = d.with_columns(P.alias("P"), R.alias("R"))
    f = (pl.when((pl.col("mult") == 0) & (pl.col("np") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))))
    d = d.with_columns(f.alias("f"))
    single = d.filter(pl.col("mult") == 0)
    return dict(
        f05=float(d["f"].mean()),
        precision=float(d.with_columns(pl.when((pl.col("mult") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(pl.col("P")).alias("pp"))["pp"].mean()),
        recall=float(d.with_columns(pl.when(pl.col("mult") == 0).then(pl.when(pl.col("np") == 0).then(1.0).otherwise(0.0)).otherwise(pl.col("R")).alias("rr"))["rr"].mean()),
        singleton_acc=float((single["np"] == 0).mean()) if len(single) else float("nan"),
        fp=int((d["np"] - d["tp"]).sum()), fn=int((d["mult"] - d["tp"]).sum()),
        matched_entities=int((d["np"] > 0).sum()), per_query=d,
    )
