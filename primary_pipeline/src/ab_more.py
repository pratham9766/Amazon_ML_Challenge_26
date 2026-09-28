"""A/B (dev fold 0): M3-style data (dev\\fold + extra) vs + 150k 'more' queries (hard-negative subsampled,
weighted). If better, fit M4F on dev + conf + extra + more with the M2/M3F rule."""
from __future__ import annotations

import glob
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from common import CACHE, OUT, get_logger, log_experiment
from metric import macro_f05
from train_matcher import PARAMS, decide, truth

log = get_logger("ab_more")
PARAMS["learning_rate"] = 0.1
dec = json.load(open(OUT / "matcher" / "M2_decision.json"))
feats = dec["features"]
m2 = lgb.Booster(model_file=str(OUT / "models" / "M2_full.txt"))
cols = ["s1", "tgt", "label"] + feats
rng = np.random.default_rng(7)

# 'more' queries: all positives, hard negatives (M2 p>=0.005), 15% random easy negatives weighted 1/0.15
Xm, ym, wm = [], [], []
for f in sorted(glob.glob(str(OUT / "features" / "train_more" / "part_*.parquet"))):
    d = pl.read_parquet(f, columns=cols)
    X = d.select(feats).to_numpy().astype(np.float32)
    y = d["label"].to_numpy()
    p = m2.predict(X, num_threads=4)
    easy = (y == 0) & (p < 0.005)
    keep_easy = easy & (rng.random(len(y)) < 0.15)
    keep = ~easy | keep_easy
    Xm.append(X[keep]); ym.append(y[keep]); wm.append(np.where(keep_easy[keep], 1 / 0.15, 1.0))
    del d, X
Xm, ym, wm = np.vstack(Xm), np.concatenate(ym), np.concatenate(wm).astype(np.float32)
log.info("more kept rows %d (pos %d)", len(ym), int(ym.sum()))

b = pl.read_parquet(CACHE / "bench_queries.parquet")
dev = pl.read_parquet(OUT / "features" / "train" / "part_*.parquet", columns=cols).join(b.select("s1", "split"), on="s1")
conf = dev.filter(pl.col("split") == "conf")
dev = dev.filter(pl.col("split") == "dev")
ex = pl.read_parquet(OUT / "features" / "train_extra" / "part_*.parquet", columns=cols)
Xd = dev.select(feats).to_numpy().astype(np.float32); yd = dev["label"].to_numpy(); g = dev["s1"].to_numpy()
Xe = ex.select(feats).to_numpy().astype(np.float32); ye = ex["label"].to_numpy()
del ex
q, gt = truth(b, "dev")
tr, va = next(iter(GroupKFold(5).split(Xd, yd, g)))
X = np.vstack([Xd[tr], Xe, Xm]); y = np.concatenate([yd[tr], ye, ym])
w = np.concatenate([np.ones(len(tr) + len(ye), np.float32), wm])
m = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=feats), 2000,
              valid_sets=[lgb.Dataset(Xd[va], yd[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
del X, y, w
qs = pl.Series(np.unique(g[va])).to_frame("s1")
pr = dev[va].select("s1", "tgt").with_columns(pl.Series("p", m.predict(Xd[va]).astype(np.float32)))
qk, gk = q.join(qs, on="s1", how="semi"), gt.join(qs, on="s1", how="semi")
new = macro_f05(decide(pr, dec["t"], dec["rel"], dec["t_top"]), qk, gk)["f05"]
# also check whether the threshold should move with more data
alt = max(((t, tt, macro_f05(decide(pr, t, 0.0, tt), qk, gk)["f05"]) for t in np.arange(0.60, 0.80, 0.02)
           for tt in (None, 0.75, 0.79, 0.85)), key=lambda x: x[2])
base = json.load(open(OUT / "matcher" / "ab_extra.json"))[0]["m2_plus_extra"]
log.info("fold0: dev+extra %.5f | +more %.5f (iter %d) | best rule on +more t=%.2f t_top=%s f05=%.5f",
         base, new, m.best_iteration, alt[0], alt[1], alt[2])
log_experiment(experiment_id="AB_more", stage="matcher-ab-dev-fold0", model="LightGBM M2 params",
               fold_strategy="dev fold0; train=dev\\fold + extra40k + more150k(hard-neg subsample, weighted)",
               threshold=f"frozen t={dec['t']} t_top={dec['t_top']}", f0_5=round(new, 5),
               notes=f"baseline dev+extra fold0={base}; best alt rule {alt}")
if new > base:
    X = np.vstack([Xd, conf.select(feats).to_numpy().astype(np.float32), Xe, Xm])
    y = np.concatenate([yd, conf["label"].to_numpy(), ye, ym])
    w = np.concatenate([np.ones(len(yd) + len(conf) + len(ye), np.float32), wm])
    n = int(m.best_iteration * 1.1)
    full = lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=feats), n)
    full.save_model(str(OUT / "models" / "M4F_full.txt"))
    json.dump(dict(dec, trained_on="dev+conf+extra+more", n_final=n), open(OUT / "matcher" / "M4F_decision.json", "w"), indent=1)
    log.info("PROMOTED M4F: %d rows, %d iters", len(y), n)
else:
    log.info("NOT PROMOTED")
