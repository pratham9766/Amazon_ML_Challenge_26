"""M8: target-ownership model with rich claimant tables, trained on dense regions 1+2,
+ cross-fitted isotonic calibration + expected-F0.5 list selection + global one-to-one.

Fair comparison: M8 OOF vs the refine winner's OOF (oof/dense_probs_<W>.parquet) on the SAME region-1 queries.
usage: python m8.py --stage1 M3F --ref M6B
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from common import CACHE, OUT, get_logger, log_experiment
from metric import macro_f05
from stage2 import ctx_features
from train_matcher import PARAMS, decide

log = get_logger("m8")
PARAMS["learning_rate"] = 0.1
MISS = 0.971  # blocker pair recall: expected true matches = sum(p) / MISS


def rich_stats(pr: pl.DataFrame) -> pl.DataFrame:
    """Per-target claimant table over the full population of claimants."""
    t = pr.group_by("tgt").agg(
        pl.col("p").top_k(3).alias("t3"), pl.len().alias("n"),
        (pl.col("p") >= 0.5).sum().alias("n50"), (pl.col("p") >= 0.7).sum().alias("n70"),
        (pl.col("p") >= 0.9).sum().alias("n90"), pl.col("p").mean().alias("pmean"), pl.col("p").std().fill_null(0).alias("pstd"))
    return t.with_columns(pl.col("t3").list.get(0).alias("t1p"),
                          pl.col("t3").list.get(1, null_on_oob=True).fill_null(0.0).alias("t2p"),
                          pl.col("t3").list.get(2, null_on_oob=True).fill_null(0.0).alias("t3p")).drop("t3")


def rich_comp(pr: pl.DataFrame, st: pl.DataFrame) -> pl.DataFrame:
    d = pr.join(st, on="tgt", how="left", maintain_order="left")
    top = pl.col("p") >= pl.col("t1p")
    other = pl.when(top).then(pl.col("t2p")).otherwise(pl.col("t1p"))
    sub = lambda c, thr: (pl.col(c) - (pl.col("p") >= thr).cast(pl.UInt32)).cast(pl.Float32)
    return d.select(
        other.alias("c_other"), (pl.col("p") - other).alias("c_margin"), top.cast(pl.Float32).alias("c_is_top"),
        (pl.col("n") - 1).cast(pl.Float32).alias("c_n_other"), sub("n50", 0.5).alias("c_n50_other"),
        sub("n70", 0.7).alias("c_n70_other"), sub("n90", 0.9).alias("c_n90_other"),
        (pl.col("t1p") - pl.col("t2p")).alias("c_best_second_gap"), pl.col("t3p").alias("c_t3p"),
        pl.col("pmean").alias("c_pmean"), pl.col("pstd").alias("c_pstd"), (pl.col("n") == 1).cast(pl.Float32).alias("c_unique"),
        # population rank from the top-3 table (identical at train and test, independent of part boundaries)
        pl.when(pl.col("p") >= pl.col("t1p")).then(1.0).when(pl.col("p") >= pl.col("t2p")).then(2.0)
        .when(pl.col("p") >= pl.col("t3p")).then(3.0).otherwise(4.0).cast(pl.Float32).alias("c_rank"),
    )


def ef05_select(pr: pl.DataFrame, kmax: int = 12) -> pl.DataFrame:
    """Per S1: choose the prefix of candidates (by p) maximizing expected F0.5 (empty allowed)."""
    d = (pr.sort(["s1", "p"], descending=[False, True])
         .with_columns(pl.col("p").cum_sum().over("s1").alias("S"), pl.int_range(1, pl.len() + 1).over("s1").alias("k"),
                       (pl.col("p").sum().over("s1") / MISS).alias("M"),
                       (1 - pl.col("p")).log().sum().over("s1").exp().alias("p_empty")))
    d = d.filter(pl.col("k") <= kmax).with_columns((1.25 * pl.col("S") / (0.25 * pl.col("M") + pl.col("k"))).alias("ef"))
    best = d.group_by("s1").agg(pl.col("ef").max().alias("efmax"), pl.col("p_empty").first(),
                                pl.col("k").sort_by("ef", descending=True).first().alias("kbest"))
    keep = best.filter(pl.col("efmax") > pl.col("p_empty")).select("s1", "kbest")
    return d.join(keep, on="s1").filter(pl.col("k") <= pl.col("kbest")).select("s1", "tgt")


def o2o(pred: pl.DataFrame, pr: pl.DataFrame) -> pl.DataFrame:
    return pred.join(pr, on=["s1", "tgt"]).sort("p", descending=True).unique("tgt", keep="first").select("s1", "tgt")


def crossfit_iso(pr: pl.DataFrame, y: np.ndarray) -> tuple[np.ndarray, IsotonicRegression]:
    h = (pr["s1"].hash(7) % 2).to_numpy()
    out = np.zeros(len(y), np.float32)
    for side in (0, 1):
        iso = IsotonicRegression(out_of_bounds="clip").fit(pr["p"].to_numpy()[h != side], y[h != side])
        out[h == side] = iso.predict(pr["p"].to_numpy()[h == side])
    return out, IsotonicRegression(out_of_bounds="clip").fit(pr["p"].to_numpy(), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="M3F")
    ap.add_argument("--ref", default="M6B")
    ap.add_argument("--regions", default="train_dense,train_dense2")
    a = ap.parse_args()
    feats = json.load(open(OUT / "matcher" / "M2_decision.json"))["features"]
    m1 = lgb.Booster(model_file=str(OUT / "models" / f"{a.stage1}_full.txt"))
    parts = []
    for r in a.regions.split(","):
        if (OUT / "features" / r).exists() and list((OUT / "features" / r).glob("part_*.parquet")):
            d = pl.read_parquet(OUT / "features" / r / "part_*.parquet", columns=["s1", "tgt", "src", "label"] + feats)
            parts.append(d.with_columns(pl.lit(r).alias("region")))
    d = pl.concat(parts)
    d = d.with_columns(pl.Series("p", m1.predict(d.select(feats).to_numpy().astype(np.float32), num_threads=4).astype(np.float32)))
    comp = rich_comp(d.select("s1", "tgt", "p"), rich_stats(d.select("tgt", "p")))
    ctx = ctx_features(d.select("s1", "tgt", "src", "p"))
    cfeats = feats + ctx.columns + comp.columns
    full = pl.concat([d.select("s1", "tgt", "label", "p", "region"), d.select(feats), ctx, comp], how="horizontal")
    del d, ctx, comp
    excl = pl.concat([pl.read_parquet(CACHE / f).select("s1") for f in ("bench_queries.parquet", "extra_queries.parquet")])
    qs = pl.concat([pl.read_parquet(CACHE / f).select("s1", "mult") for f in ("dense_queries.parquet", "dense2_queries.parquet")
                    if (CACHE / f).exists()]).unique("s1").join(excl, on="s1", how="anti")
    tr = full.join(qs.select("s1"), on="s1", how="semi")
    X = tr.select(cfeats).to_numpy().astype(np.float32); y = tr["label"].to_numpy(); g = tr["s1"].to_numpy()
    oof = np.zeros(len(y), np.float32); iters = []
    for k, (a_, b_) in enumerate(GroupKFold(5).split(X, y, g)):
        m = lgb.train(PARAMS, lgb.Dataset(X[a_], y[a_], feature_name=cfeats), 2000,
                      valid_sets=[lgb.Dataset(X[b_], y[b_])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b_] = m.predict(X[b_]); iters.append(m.best_iteration)
        log.info("M8 fold %d iter %d", k, m.best_iteration)
    pr = tr.select("s1", "tgt", "region").with_columns(pl.Series("p", oof))
    cal, iso = crossfit_iso(pr, y)
    prc = pr.with_columns(pl.Series("p", cal))
    gt_all = pl.read_parquet(CACHE / "train_gt_pairs.parquet").join(qs.select("s1"), on="s1", how="semi")

    def evaluate(pp, region=None):
        qq = qs if region is None else qs.join(pp.filter(pl.col("region") == region).select("s1").unique(), on="s1", how="semi")
        pp = pp.join(qq.select("s1"), on="s1", how="semi")
        gg = gt_all.join(qq.select("s1"), on="s1", how="semi")
        res = {}
        for cname, P in (("raw", pp), ("iso", None)):
            if P is None:
                P = prc.join(qq.select("s1"), on="s1", how="semi")
            best = None
            for t in np.arange(0.40, 0.90, 0.04):
                for tt in (None, 0.8, 0.9):
                    r = macro_f05(o2o(decide(P, float(t), 0.0, tt), P), qq, gg)
                    if best is None or r["f05"] > best[1]["f05"]:
                        best = ((round(float(t), 2), tt), r)
            ef = macro_f05(o2o(ef05_select(P), P), qq, gg)
            res[cname] = dict(threshold=dict(rule=best[0], f05=best[1]["f05"], P=best[1]["precision"], R=best[1]["recall"],
                                             single=best[1]["singleton_acc"]),
                              ef05=dict(f05=ef["f05"], P=ef["precision"], R=ef["recall"], single=ef["singleton_acc"]))
        return res

    results = {"all": evaluate(pr)}
    for r in pr["region"].unique().to_list():
        results[r] = evaluate(pr, r)
    # fair head-to-head with the refine winner on the SAME region-1 queries
    ref = pl.read_parquet(OUT / "oof" / f"dense_probs_{a.ref}.parquet")
    r1q = pr.filter(pl.col("region") == "train_dense").select("s1").unique()
    q1 = qs.join(r1q, on="s1", how="semi"); g1 = gt_all.join(r1q, on="s1", how="semi")
    refp = ref.join(r1q, on="s1", how="semi")
    refdec = json.load(open(OUT / "matcher" / f"{a.ref}_decision.json"))
    rp = decide(refp, refdec["t"], 0.0, refdec["t_top"])
    ref_f = macro_f05(o2o(rp, refp) if refdec["one_to_one"] else rp, q1, g1)["f05"]
    results["ref_region1"] = {a.ref: ref_f}
    log.info("M8 results %s", json.dumps(results, default=str))
    # choose final decision on ALL dense queries
    cands = [("raw", "threshold"), ("raw", "ef05"), ("iso", "threshold"), ("iso", "ef05")]
    cname, dname = max(cands, key=lambda c: results["all"][c[0]][c[1]]["f05"])
    m8 = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cfeats), int(np.mean(iters) * 1.1))
    m8.save_model(str(OUT / "models" / "M8_full.txt"))
    dec = dict(stage1=a.stage1, stage1_features=feats, features=cfeats, calibration=cname, decision=dname,
               threshold_rule=results["all"][cname]["threshold"]["rule"], one_to_one=True,
               iso_x=iso.X_thresholds_.tolist(), iso_y=iso.y_thresholds_.tolist(),
               dense_f05=results["all"][cname][dname]["f05"],
               region1_f05=results["train_dense"][cname][dname]["f05"] if "train_dense" in results else None,
               ref_region1=ref_f, ref=a.ref)
    json.dump(dec, open(OUT / "matcher" / "M8_decision.json", "w"), indent=1)
    json.dump(dict(results=results, importance=sorted(zip(cfeats, m8.feature_importance("gain").round(1).tolist()), key=lambda x: -x[1])[:30]),
              open(OUT / "matcher" / "M8_result.json", "w"), indent=1, default=str)
    log_experiment(experiment_id="M8_ownership", stage="matcher-oof-dense-regions", model=f"LightGBM ownership stage on {a.stage1}",
                   features=f"{len(cfeats)} (rich claimant tables + ctx)", fold_strategy="GroupKFold(5) over dense regions 1+2 (excl bench+extra)",
                   threshold=f"{cname}/{dname} rule={dec['threshold_rule']} +1to1", f0_5=round(dec["dense_f05"], 5),
                   notes=f"region1: M8 {dec['region1_f05']} vs {a.ref} {ref_f:.5f}")
    log.info("M8 chosen %s/%s dense %.5f | region1 M8 %s vs %s %.5f", cname, dname, dec["dense_f05"], dec["region1_f05"], a.ref, ref_f)


if __name__ == "__main__":
    main()
