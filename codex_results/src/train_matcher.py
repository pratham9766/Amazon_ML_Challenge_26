"""Train the pair matcher with grouped OOF on dev queries; select entity-level decision rule.

usage: python train_matcher.py --name M1 [--hard_neg_weight 1.0] [--preset fast|full]
Outputs: oof/<name>_oof.parquet, models/<name>_fold*.txt, matcher/<name>_decision.json
Confirmation is scored once per call with the frozen (OOF-selected) rule and reported separately.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import CACHE, OUT, Timer, get_logger, log_experiment
from metric import macro_f05

log = get_logger("matcher")
NONFEAT = {"s1", "tgt", "label", "src"}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=3, seed=20260925)


def load_train(extra: bool = False):
    df = pl.read_parquet(OUT / "features" / "train" / "part_*.parquet")
    b = pl.read_parquet(CACHE / "bench_queries.parquet")
    df = df.join(b.select("s1", "split"), on="s1")
    if extra:
        ex = pl.read_parquet(OUT / "features" / "train_extra" / "part_*.parquet").with_columns(pl.lit("extra").alias("split"))
        df = pl.concat([df, ex.select(df.columns)])
    return df, b


def truth(b: pl.DataFrame, split: str):
    q = b.filter(pl.col("split") == split).select("s1", "mult")
    gt = pl.read_parquet(CACHE / "train_gt_pairs.parquet").join(q.select("s1"), on="s1", how="semi")
    return q, gt


def decide(pr: pl.DataFrame, t: float, rel: float = 0.0, t_top: float | None = None) -> pl.DataFrame:
    """Keep p>=t and p>=rel*max_p(query); query must have max_p>=t_top (singleton gate)."""
    d = pr.with_columns(pl.col("p").max().over("s1").alias("pmax"))
    cond = (pl.col("p") >= t) & (pl.col("p") >= rel * pl.col("pmax"))
    if t_top is not None:
        cond = cond & (pl.col("pmax") >= t_top)
    return d.filter(cond).select("s1", "tgt")


def sweep(pr, q, gt):
    rows = []
    for t in np.round(np.arange(0.2, 0.91, 0.05), 3):
        m = macro_f05(decide(pr, t), q, gt)
        rows.append(dict(t=float(t), rel=0.0, t_top=None, **{k: v for k, v in m.items() if k != "per_query"}))
    best = max(rows, key=lambda r: r["f05"])
    for t in np.round(np.arange(best["t"] - 0.05, best["t"] + 0.051, 0.01), 3):
        for rel in (0.0, 0.2, 0.4):
            for t_top in (None, float(t) + 0.1, float(t) + 0.2):
                if t_top is not None and t_top >= 1:
                    continue
                m = macro_f05(decide(pr, float(t), rel, t_top), q, gt)
                rows.append(dict(t=float(t), rel=rel, t_top=t_top, **{k: v for k, v in m.items() if k != "per_query"}))
    return sorted(rows, key=lambda r: -r["f05"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="M1")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--drop", default="", help="comma-separated feature prefixes to drop (ablation)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--extra", action="store_true", help="add 40k extra train queries to the training folds")
    a = ap.parse_args()
    PARAMS["learning_rate"] = a.lr
    t = Timer()
    df, b = load_train(a.extra)
    feats = [c for c in df.columns if c not in NONFEAT | {"split"}]
    if a.drop:
        feats = [c for c in feats if not any(c.startswith(x) for x in a.drop.split(","))]
    dev = df.filter(pl.col("split").is_in(["dev", "extra"]))
    conf = df.filter(pl.col("split") == "conf")
    X = dev.select(feats).to_numpy().astype(np.float32)
    y = dev["label"].to_numpy()
    groups = dev["s1"].to_numpy()
    dev = dev.select("s1", "tgt", "split", "label")  # free the wide frame
    df = None
    import gc; gc.collect()
    oof = np.zeros(len(dev), np.float32)
    best_iters = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        m.save_model(str(OUT / "models" / f"{a.name}_fold{k}.txt"), num_iteration=m.best_iteration)
        log.info("fold %d best_iter=%d %.0fs", k, m.best_iteration, t())
    # final model on all dev for confirmation/test
    n_final = int(np.mean(best_iters) * 1.1)
    full = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), n_final)
    full.save_model(str(OUT / "models" / f"{a.name}_full.txt"))
    imp = sorted(zip(feats, full.feature_importance("gain")), key=lambda x: -x[1])
    is_dev = (dev["split"] == "dev").to_numpy()
    pr = dev.select("s1", "tgt").with_columns(pl.Series("p", oof)).filter(pl.Series(is_dev))
    pr.with_columns(dev["label"].filter(pl.Series(is_dev))).write_parquet(OUT / "oof" / f"{a.name}_oof.parquet")
    q, gt = truth(b, "dev")
    rows = sweep(pr, q, gt)
    best = rows[0]
    # fold stability of the chosen rule
    fold_f = []
    for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        qs = pl.Series(np.unique(groups[va][is_dev[va]])).to_frame("s1")
        fm = macro_f05(decide(pr.join(qs, on="s1", how="semi"), best["t"], best["rel"], best["t_top"]),
                       q.join(qs, on="s1", how="semi"), gt.join(qs, on="s1", how="semi"))
        fold_f.append(round(fm["f05"], 5))
    # confirmation (holdout) with frozen rule and the all-dev model
    pc = conf.select("s1", "tgt").with_columns(pl.Series("p", full.predict(conf.select(feats).to_numpy().astype(np.float32))))
    cq, cgt = truth(b, "conf")
    cm = macro_f05(decide(pc, best["t"], best["rel"], best["t_top"]), cq, cgt)
    cm.pop("per_query")
    decision = dict(t=best["t"], rel=best["rel"], t_top=best["t_top"], features=feats, n_final=n_final)
    json.dump(decision, open(OUT / "matcher" / f"{a.name}_decision.json", "w"), indent=1)
    res = dict(name=a.name, oof_best=best, fold_f05=fold_f, top_rules=rows[:10], confirmation=cm,
               oof_auc_note="pair-level", importance=[(f, round(float(g), 1)) for f, g in imp[:40]], secs=round(t()))
    json.dump(res, open(OUT / "matcher" / f"{a.name}_result.json", "w"), indent=1, default=str)
    log.info("OOF best %s | folds %s | conf %s", {k: best[k] for k in ("t", "rel", "t_top", "f05", "precision", "recall", "singleton_acc")}, fold_f,
             {k: round(v, 4) if isinstance(v, float) else v for k, v in cm.items()})
    log_experiment(experiment_id=a.name, stage="matcher-oof-dev20k", model="LightGBM", features=f"{len(feats)} feats" + (f" drop={a.drop}" if a.drop else ""),
                   blocking_strategy="FROZEN comb_t@25+name_na@3", candidate_cap=56, fold_strategy="GroupKFold(5) by S1",
                   threshold=f"t={best['t']} rel={best['rel']} t_top={best['t_top']}", precision=round(best["precision"], 5),
                   recall=round(best["recall"], 5), f0_5=round(best["f05"], 5), singleton_accuracy=round(best["singleton_acc"], 5),
                   blocking_recall=0.97, mean_candidates=52.6, runtime=round(t()),
                   notes=f"folds={fold_f} CONF f05={cm['f05']:.5f} P={cm['precision']:.4f} R={cm['recall']:.4f} singleton={cm['singleton_acc']:.4f}")


if __name__ == "__main__":
    main()
