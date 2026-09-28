"""A/B: does adding the 40k extra training queries improve the matcher?

For folds 0 and 1 of the same GroupKFold(5) over dev: train on (dev minus fold) + extra, score the fold
with M2's frozen rule, and compare to M2's OOF on the same fold. If it wins on both folds, fit the final
model (M3F) on dev + conf + extra with M2's parameters and write M3F_decision.json (same rule).
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import CACHE, OUT, get_logger, log_experiment
from metric import macro_f05
from train_matcher import PARAMS, decide, truth

log = get_logger("ab_extra")
PARAMS["learning_rate"] = 0.1
dec = json.load(open(OUT / "matcher" / "M2_decision.json"))
feats = dec["features"]
b = pl.read_parquet(CACHE / "bench_queries.parquet")
cols = ["s1", "tgt", "label"] + feats
dev = pl.read_parquet(OUT / "features" / "train" / "part_*.parquet", columns=cols).join(b.select("s1", "split"), on="s1")
ex = pl.read_parquet(OUT / "features" / "train_extra" / "part_*.parquet", columns=cols)
conf = dev.filter(pl.col("split") == "conf")
dev = dev.filter(pl.col("split") == "dev")
Xd = dev.select(feats).to_numpy().astype(np.float32); yd = dev["label"].to_numpy(); g = dev["s1"].to_numpy()
Xe = ex.select(feats).to_numpy().astype(np.float32); ye = ex["label"].to_numpy()
del ex
q, gt = truth(b, "dev")
m2oof = pl.read_parquet(OUT / "oof" / "M2_oof.parquet").select("s1", "tgt", "p")
wins, rows = 0, []
for k, (tr, va) in enumerate(GroupKFold(5).split(Xd, yd, g)):
    if k > 1:
        break
    X = np.vstack([Xd[tr], Xe]); y = np.concatenate([yd[tr], ye])
    m = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), 1500, valid_sets=[lgb.Dataset(Xd[va], yd[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    del X, y
    qs = pl.Series(np.unique(g[va])).to_frame("s1")
    qk, gk = q.join(qs, on="s1", how="semi"), gt.join(qs, on="s1", how="semi")
    pr = dev[va].select("s1", "tgt").with_columns(pl.Series("p", m.predict(Xd[va]).astype(np.float32)))
    new = macro_f05(decide(pr, dec["t"], dec["rel"], dec["t_top"]), qk, gk)["f05"]
    old = macro_f05(decide(m2oof.join(qs, on="s1", how="semi"), dec["t"], dec["rel"], dec["t_top"]), qk, gk)["f05"]
    rows.append(dict(fold=k, m2=round(old, 5), m2_plus_extra=round(new, 5), iters=m.best_iteration))
    wins += new > old
    log.info("fold %d: M2 %.5f  M2+extra %.5f (iter %d)", k, old, new, m.best_iteration)
json.dump(rows, open(OUT / "matcher" / "ab_extra.json", "w"), indent=1)
log_experiment(experiment_id="AB_extra", stage="matcher-ab-dev-folds01", model="LightGBM M2 params", features=f"{len(feats)} (M2)",
               fold_strategy="dev GroupKFold(5) folds 0,1; train=dev\\fold + 40k extra", threshold=f"M2 frozen t={dec['t']} t_top={dec['t_top']}",
               f0_5=np.mean([r["m2_plus_extra"] for r in rows]), notes=json.dumps(rows))
if wins == 2:
    X = np.vstack([Xd, conf.select(feats).to_numpy().astype(np.float32), Xe])
    y = np.concatenate([yd, conf["label"].to_numpy(), ye])
    n = int(np.mean([r["iters"] for r in rows]) * 1.1)
    full = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feats), n)
    full.save_model(str(OUT / "models" / "M3F_full.txt"))
    json.dump(dict(dec, trained_on="dev+conf+extra", n_final=n), open(OUT / "matcher" / "M3F_decision.json", "w"), indent=1)
    log.info("PROMOTED: M3F trained on %d rows, %d iters", len(y), n)
else:
    log.info("NOT PROMOTED (wins=%d/2)", wins)
