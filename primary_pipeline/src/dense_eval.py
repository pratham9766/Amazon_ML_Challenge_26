"""Evaluate target-conflict (one-to-one) resolution on a DENSE region.

Every train S1 in KS/WV (US) and Punjab/Odisha (India) is queried, so competing
S1s that claim the same target are present, as on test. Ground truth says each
target belongs to at most one S1. Dev queries (the model's training set) stay in
as competitors but are excluded from scoring.

usage: python dense_eval.py --model M2
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from common import CACHE, OUT, get_logger, log_experiment
from metric import macro_f05
from stage2 import ctx_features
from train_matcher import decide

log = get_logger("dense")


def resolve(pred: pl.DataFrame, pr: pl.DataFrame, margin: float | None) -> pl.DataFrame:
    """margin None = no resolution; 0 = argmax only; m>0 = keep if p >= max_p(target) - m."""
    if margin is None:
        return pred
    d = pred.join(pr, on=["s1", "tgt"]).with_columns(pl.col("p").max().over("tgt").alias("tmax"))
    return d.filter(pl.col("p") >= pl.col("tmax") - margin - 1e-9).select("s1", "tgt")


def predict(model: str) -> pl.DataFrame:
    dec = json.load(open(OUT / "matcher" / f"{model}_decision.json"))
    b = lgb.Booster(model_file=str(OUT / "models" / f"{model}_full.txt"))
    d = pl.read_parquet(OUT / "features" / "train_dense" / "part_*.parquet")
    if dec.get("stage1"):
        b1 = lgb.Booster(model_file=str(OUT / "models" / f"{dec['stage1']}_full.txt"))
        p1 = b1.predict(d.select(dec["stage1_features"]).to_numpy().astype(np.float32), num_threads=2).astype(np.float32)
        ctx = ctx_features(d.select("s1", "tgt", "src").with_columns(pl.Series("p", p1)))
        X = pl.concat([d.select(dec["stage1_features"]), ctx], how="horizontal").select(dec["features"]).to_numpy().astype(np.float32)
    else:
        X = d.select(dec["features"]).to_numpy().astype(np.float32)
    return d.select("s1", "tgt").with_columns(pl.Series("p", b.predict(X, num_threads=2).astype(np.float32))), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M2")
    a = ap.parse_args()
    pr, dec = predict(a.model)
    q = pl.read_parquet(CACHE / "dense_queries.parquet").select("s1", "mult")
    dev = pl.read_parquet(CACHE / "bench_queries.parquet").filter(pl.col("split") == "dev").select("s1")
    qe = q.join(dev, on="s1", how="anti")
    gt = pl.read_parquet(CACHE / "train_gt_pairs.parquet").join(qe.select("s1"), on="s1", how="semi")
    rows = []
    for t in np.round(np.arange(dec["t"] - 0.2, dec["t"] + 0.11, 0.05), 3):
        for tt in (None, dec["t_top"]):
            base = decide(pr, float(t), dec["rel"], tt)
            for m in (None, 0.0, 0.02, 0.05, 0.1, 0.2):
                r = macro_f05(resolve(base, pr, m).join(qe.select("s1"), on="s1", how="semi"), qe, gt)
                r.pop("per_query")
                rows.append(dict(t=float(t), t_top=tt, margin=m, **r))
    rows.sort(key=lambda r: -r["f05"])
    ref = [r for r in rows if r["margin"] is None and abs(r["t"] - dec["t"]) < 1e-6 and r["t_top"] == dec["t_top"]][0]
    conflicts = decide(pr, dec["t"], dec["rel"], dec["t_top"]).group_by("tgt").len().filter(pl.col("len") > 1).height
    out = dict(model=a.model, queries_scored=len(qe), frozen_rule_no_resolution=ref, best=rows[0], top=rows[:12],
               conflicted_targets_at_frozen_rule=conflicts)
    json.dump(out, open(OUT / "matcher" / f"dense_eval_{a.model}.json", "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("queries_scored", "conflicted_targets_at_frozen_rule")}, indent=0))
    for r in [ref] + rows[:12]:
        print(f"t={r['t']:.2f} t_top={r['t_top']} margin={r['margin']} f05={r['f05']:.5f} P={r['precision']:.4f} R={r['recall']:.4f} single={r['singleton_acc']:.4f} fp={r['fp']} fn={r['fn']}")
    b = rows[0]
    log_experiment(experiment_id=f"D_{a.model}_dense", stage="dense-region-eval", model=a.model, fold_strategy="dense KS/WV/Punjab/Odisha, dev excluded",
                   threshold=f"t={b['t']} t_top={b['t_top']} margin={b['margin']}", precision=round(b["precision"], 5), recall=round(b["recall"], 5),
                   f0_5=round(b["f05"], 5), singleton_accuracy=round(b["singleton_acc"], 5),
                   notes=f"no-resolution frozen rule f05={ref['f05']:.5f}; conflicts={conflicts}")


if __name__ == "__main__":
    main()
