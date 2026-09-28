"""Competition-aware stage 2 (M5), trained on the DENSE labelled region.

83% of candidate negatives are the true matches of ANOTHER S1. With every S1 of a region queried (as on
test), each candidate record's claimants are visible. Features per (s1, tgt):
  p (M2 stage-1), best p of any other S1 for the same tgt, margin, rank among claimants, #claimants,
  #strong claimants, + per-query context of p, + all M2 pair features.
GroupKFold(5) by S1 on dense (dev S1s excluded from training/scoring, kept as competitors).
Compares against M2 (frozen rule) with and without one-to-one on the same queries.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import CACHE, OUT, get_logger, log_experiment
from metric import macro_f05
from stage2 import ctx_features
from train_matcher import PARAMS, decide

log = get_logger("compete")
PARAMS["learning_rate"] = 0.1


def tgt_stats(pr: pl.DataFrame) -> pl.DataFrame:
    """Per-target claimant summary over the whole population (small: one row per target)."""
    top = pr.group_by("tgt").agg(
        pl.col("p").top_k(2).alias("t2"), pl.len().alias("c_n"), (pl.col("p") >= 0.5).sum().alias("c_n50"))
    return top.with_columns(pl.col("t2").list.get(0).alias("t1p"),
                            pl.col("t2").list.get(1, null_on_oob=True).fill_null(0.0).alias("t2p")).drop("t2")


def comp_from_stats(pr: pl.DataFrame, top: pl.DataFrame) -> pl.DataFrame:
    """pr: s1, tgt, p rows (any subset); top: tgt_stats over the full population. Aligned to pr rows."""
    d = pr.join(top, on="tgt", how="left", maintain_order="left")
    other = pl.when(pl.col("p") >= pl.col("t1p")).then(pl.col("t2p")).otherwise(pl.col("t1p"))
    return d.select(
        other.alias("c_other"),
        (pl.col("p") - other).alias("c_margin"),
        (pl.col("p") >= pl.col("t1p")).cast(pl.Float32).alias("c_is_top"),
        (pl.col("c_n") - 1).cast(pl.Float32).alias("c_n_other"),
        (pl.col("c_n50") - (pl.col("p") >= 0.5).cast(pl.UInt32)).cast(pl.Float32).alias("c_n50_other"),
    )


def comp_features(pr: pl.DataFrame) -> pl.DataFrame:
    """pr: s1, tgt, p over ALL claimants in the population. Returns comp columns aligned to pr rows."""
    return comp_from_stats(pr, tgt_stats(pr))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="M2", help="booster name, or with --probs a saved-probability source")
    ap.add_argument("--probs", action="store_true", help="stage-1 p from oof/dense_probs_<stage1>.parquet (iteration rounds)")
    ap.add_argument("--name", default="M5")
    a = ap.parse_args()
    feats = json.load(open(OUT / "matcher" / "M2_decision.json"))["features"]
    d = pl.read_parquet(OUT / "features" / "train_dense" / "part_*.parquet", columns=["s1", "tgt", "src", "label"] + feats)
    if a.probs:
        d = d.join(pl.read_parquet(OUT / "oof" / f"dense_probs_{a.stage1}.parquet"), on=["s1", "tgt"], how="left")              .with_columns(pl.col("p").fill_null(0.0))
    else:
        m1 = lgb.Booster(model_file=str(OUT / "models" / f"{a.stage1}_full.txt"))
        d = d.with_columns(pl.Series("p", m1.predict(d.select(feats).to_numpy().astype(np.float32), num_threads=4).astype(np.float32)))
    comp = comp_features(d.select("s1", "tgt", "p"))
    ctx = ctx_features(d.select("s1", "tgt", "src", "p"))
    cfeats = feats + ctx.columns + comp.columns
    full = pl.concat([d.select("s1", "tgt", "label", "p"), d.select(feats), ctx, comp], how="horizontal")
    del d, ctx, comp
    # one common evaluation subset for every model: exclude all S1s any stage-1 model was trained on
    excl = pl.concat([pl.read_parquet(CACHE / "bench_queries.parquet").select("s1"),
                      pl.read_parquet(CACHE / "extra_queries.parquet").select("s1")])
    q = pl.read_parquet(CACHE / "dense_queries.parquet").select("s1", "mult").join(excl, on="s1", how="anti")
    gt = pl.read_parquet(CACHE / "train_gt_pairs.parquet").join(q.select("s1"), on="s1", how="semi")
    tr_df = full.join(q.select("s1"), on="s1", how="semi")
    rest = full.join(q.select("s1"), on="s1", how="anti")
    X = tr_df.select(cfeats).to_numpy().astype(np.float32); y = tr_df["label"].to_numpy(); g = tr_df["s1"].to_numpy()
    oof = np.zeros(len(y), np.float32); iters = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, g)):
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=cfeats), 2000,
                      valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va]); iters.append(m.best_iteration)
        log.info("%s fold %d iter %d", a.name, k, m.best_iteration)
    base = tr_df.select("s1", "tgt", "p")
    pr = tr_df.select("s1", "tgt").with_columns(pl.Series("p", oof))

    def score(pp, t, tt, o2o):
        pred = decide(pp, t, 0.0, tt)
        if o2o:
            pred = pred.join(pp, on=["s1", "tgt"]).sort("p", descending=True).unique("tgt", keep="first").select("s1", "tgt")
        return macro_f05(pred, q, gt)

    def sweep(pp):
        rows = []
        for t in np.arange(0.40, 0.86, 0.04):
            for tt in (None, 0.7, 0.8, 0.9):
                for o in (False, True):
                    r = score(pp, float(t), tt, o)
                    rows.append(dict(t=round(float(t), 2), t_top=tt, o2o=o, f05=r["f05"], P=r["precision"], R=r["recall"], single=r["singleton_acc"]))
        return sorted(rows, key=lambda r: -r["f05"])

    ref = sweep(base)[0]
    rows = sweep(pr)
    best = rows[0]
    log.info("DENSE[B] %s: stage1 %s best %.5f | %s best %s", a.name, a.stage1, ref["f05"], a.name, best)
    m5 = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cfeats), int(np.mean(iters) * 1.1))
    m5.save_model(str(OUT / "models" / f"{a.name}_full.txt"))
    # population probabilities for the next round: OOF on evaluated S1s, full model on excluded (never trained on)
    pr_rest = rest.select("s1", "tgt").with_columns(
        pl.Series("p", m5.predict(rest.select(cfeats).to_numpy().astype(np.float32)).astype(np.float32)))
    pl.concat([pr, pr_rest]).write_parquet(OUT / "oof" / f"dense_probs_{a.name}.parquet")
    json.dump(dict(t=best["t"], t_top=best["t_top"], rel=0.0, one_to_one=best["o2o"], features=cfeats, stage1=a.stage1,
                   stage1_features=feats, dense_f05=best["f05"], dense_ref_stage1=ref["f05"], subset="B"),
              open(OUT / "matcher" / f"{a.name}_decision.json", "w"), indent=1)
    json.dump(dict(ref=ref, top=rows[:15],
                   importance=sorted(zip(cfeats, m5.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1])[:25]),
              open(OUT / "matcher" / f"{a.name}_result.json", "w"), indent=1, default=str)
    log_experiment(experiment_id=f"{a.name}_compete", stage="matcher-oof-denseB", model=f"LightGBM competition stage on {a.stage1}",
                   features=f"{len(cfeats)} (+claimant competition +query ctx)", fold_strategy="dense region (excl. bench+extra) GroupKFold(5) by S1",
                   threshold=f"t={best['t']} t_top={best['t_top']} 1to1={best['o2o']}", precision=round(best["P"], 5), recall=round(best["R"], 5),
                   f0_5=round(best["f05"], 5), singleton_accuracy=round(best["single"], 5),
                   notes=f"same queries, stage1 {a.stage1} best rule={ref['f05']:.5f}")


if __name__ == "__main__":
    main()
