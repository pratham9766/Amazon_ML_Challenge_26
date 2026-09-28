"""Stage-2 matcher: base pair features + query-context features of stage-1 probabilities.

Trained on stage-1 OUT-OF-FOLD probabilities with the same GroupKFold(5) by S1.
usage: python stage2.py --base M2 --name M2S2
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import OUT, Timer, get_logger, log_experiment
from metric import macro_f05
from train_matcher import PARAMS, decide, load_train, sweep, truth

log = get_logger("stage2")


def ctx_features(pr: pl.DataFrame) -> pl.DataFrame:
    """pr: s1, tgt, src, p (stage-1). Returns context columns aligned with pr rows."""
    return pr.select(
        pl.col("p").alias("s1p"),
        pl.col("p").max().over("s1").alias("s1p_qmax"),
        (pl.col("p") - pl.col("p").max().over("s1")).alias("s1p_gap"),
        pl.col("p").rank("average", descending=True).over("s1").alias("s1p_rank"),
        pl.col("p").rank("average", descending=True).over("s1", "src").alias("s1p_src_rank"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("s1").fill_null(0.0).alias("s1p_second"),
        (pl.col("p") >= 0.5).sum().over("s1").cast(pl.Float32).alias("s1p_n50"),
        (pl.col("p") >= 0.2).sum().over("s1").cast(pl.Float32).alias("s1p_n20"),
        pl.col("p").sum().over("s1").alias("s1p_sum"),
        (pl.col("p") >= 0.5).sum().over("s1", "src").cast(pl.Float32).alias("s1p_src_n50"),
        (pl.col("p") / pl.col("p").sum().over("s1")).alias("s1p_frac"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="M2")
    ap.add_argument("--name", default="M2S2")
    ap.add_argument("--lr", type=float, default=0.1)
    a = ap.parse_args()
    PARAMS["learning_rate"] = a.lr
    t = Timer()
    df, b = load_train()
    base_dec = json.load(open(OUT / "matcher" / f"{a.base}_decision.json"))
    bfeats = base_dec["features"]
    oof = pl.read_parquet(OUT / "oof" / f"{a.base}_oof.parquet").select("s1", "tgt", "p")
    dev = df.filter(pl.col("split") == "dev").join(oof, on=["s1", "tgt"])
    ctx = ctx_features(dev.select("s1", "tgt", "src", "p"))
    feats = bfeats + ctx.columns
    X = pl.concat([dev.select(bfeats), ctx], how="horizontal").to_numpy().astype(np.float32)
    y = dev["label"].to_numpy()
    groups = dev["s1"].to_numpy()
    oof2 = np.zeros(len(dev), np.float32)
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feats), 1500,
                      valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof2[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log.info("fold %d iter %d %.0fs", k, m.best_iteration, t())
    full = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), int(np.mean(iters) * 1.1))
    full.save_model(str(OUT / "models" / f"{a.name}_full.txt"))
    pr = dev.select("s1", "tgt").with_columns(pl.Series("p", oof2))
    pr.with_columns(dev["label"]).write_parquet(OUT / "oof" / f"{a.name}_oof.parquet")
    q, gt = truth(b, "dev")
    rows = sweep(pr, q, gt)
    best = rows[0]
    fold_f = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        qs = pl.Series(np.unique(groups[va])).to_frame("s1")
        fm = macro_f05(decide(pr.join(qs, on="s1", how="semi"), best["t"], best["rel"], best["t_top"]),
                       q.join(qs, on="s1", how="semi"), gt.join(qs, on="s1", how="semi"))
        fold_f.append(round(fm["f05"], 5))
    # confirmation: stage1 full model -> ctx -> stage2 full model
    conf = df.filter(pl.col("split") == "conf")
    b1 = lgb.Booster(model_file=str(OUT / "models" / f"{a.base}_full.txt"))
    p1 = b1.predict(conf.select(bfeats).to_numpy().astype(np.float32))
    cctx = ctx_features(conf.select("s1", "tgt", "src").with_columns(pl.Series("p", p1.astype(np.float32))))
    p2 = full.predict(pl.concat([conf.select(bfeats), cctx], how="horizontal").to_numpy().astype(np.float32))
    cq, cgt = truth(b, "conf")
    cm = macro_f05(decide(conf.select("s1", "tgt").with_columns(pl.Series("p", p2)), best["t"], best["rel"], best["t_top"]), cq, cgt)
    cm.pop("per_query")
    json.dump(dict(t=best["t"], rel=best["rel"], t_top=best["t_top"], features=feats, stage1=a.base, stage1_features=bfeats),
              open(OUT / "matcher" / f"{a.name}_decision.json", "w"), indent=1)
    imp = sorted(zip(feats, full.feature_importance("gain")), key=lambda x: -x[1])
    json.dump(dict(name=a.name, oof_best=best, fold_f05=fold_f, top_rules=rows[:10], confirmation=cm,
                   importance=[(f, round(float(g), 1)) for f, g in imp[:40]]), open(OUT / "matcher" / f"{a.name}_result.json", "w"), indent=1, default=str)
    log.info("OOF best %s | folds %s | conf %s", {k: best[k] for k in ("t", "rel", "t_top", "f05", "precision", "recall", "singleton_acc")}, fold_f,
             {k: round(v, 4) if isinstance(v, float) else v for k, v in cm.items()})
    log_experiment(experiment_id=a.name, stage="matcher-oof-dev20k", model=f"LightGBM stage2 on {a.base} OOF", features=f"{len(feats)} feats (+query ctx of stage-1 p)",
                   blocking_strategy="FROZEN comb_t@25+name_na@3", candidate_cap=56, fold_strategy="GroupKFold(5) by S1",
                   threshold=f"t={best['t']} rel={best['rel']} t_top={best['t_top']}", precision=round(best["precision"], 5),
                   recall=round(best["recall"], 5), f0_5=round(best["f05"], 5), singleton_accuracy=round(best["singleton_acc"], 5),
                   blocking_recall=0.97, mean_candidates=52.6, runtime=round(t()),
                   notes=f"folds={fold_f} CONF f05={cm['f05']:.5f} P={cm['precision']:.4f} R={cm['recall']:.4f} singleton={cm['singleton_acc']:.4f}")


if __name__ == "__main__":
    main()
