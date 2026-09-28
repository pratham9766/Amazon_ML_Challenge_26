"""Test submission for M8 (rich ownership features + calibration + EF0.5/threshold + global one-to-one)."""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import DATA, OUT, VALIDATOR, get_logger, source_path
from m8 import ef05_select, o2o, rich_comp, rich_stats
from make_submission import write_lists
from stage2 import ctx_features
from train_matcher import decide

log = get_logger("submit_m8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub_id", default="S008_M8")
    a = ap.parse_args()
    dec = json.load(open(OUT / "matcher" / "M8_decision.json"))
    m8 = lgb.Booster(model_file=str(OUT / "models" / "M8_full.txt"))
    f1 = dec["stage1_features"]
    ix, iy = np.array(dec["iso_x"]), np.array(dec["iso_y"])
    pdir = OUT / "predictions" / "final_M8"; pdir.mkdir(parents=True, exist_ok=True)
    preds, allp = [], []
    for c in ("France", "India", "US"):
        cache = pdir / f"{c}.parquet"; pcache = pdir / f"{c}_all.parquet"
        if cache.exists() and pcache.exists():
            preds.append(pl.read_parquet(cache)); allp.append(pl.read_parquet(pcache)); continue
        pparts = sorted(glob.glob(str(OUT / "predictions" / f"parts_{dec['stage1']}" / f"part_{c}_*.parquet")))
        st = rich_stats(pl.concat([pl.read_parquet(p).select("tgt", "p") for p in pparts]))
        cp, ca = [], []
        for pp in pparts:
            name = pp.replace("\\", "/").split("/")[-1]
            d = pl.read_parquet(OUT / "features" / "test" / name, columns=["s1", "tgt", "src"] + f1).join(pl.read_parquet(pp), on=["s1", "tgt"])
            X = pl.concat([d.select(f1), ctx_features(d.select("s1", "tgt", "src", "p")), rich_comp(d.select("s1", "tgt", "p"), st)],
                          how="horizontal").select(dec["features"]).to_numpy().astype(np.float32)
            p = m8.predict(X, num_threads=4)
            if dec["calibration"] == "iso":
                p = np.interp(p, ix, iy)
            pr = d.select("s1", "tgt").with_columns(pl.Series("p", p.astype(np.float32)))
            if dec["decision"] == "ef05":
                sel = ef05_select(pr)
            else:
                t, tt = dec["threshold_rule"]
                sel = decide(pr, t, 0.0, tt)
            cp.append(sel.join(pr, on=["s1", "tgt"])); ca.append(pr.filter(pl.col("p") >= 0.05))
        c_pred, c_all = pl.concat(cp), pl.concat(ca)
        c_pred.write_parquet(cache); c_all.write_parquet(pcache)
        preds.append(c_pred); allp.append(c_all)
        log.info("%s done", c)
    pred = pl.concat(preds)
    pred = pred.sort("p", descending=True).unique("tgt", keep="first").select("s1", "tgt")  # global one-to-one
    all_s1 = pl.scan_parquet(source_path("test", 1)).select(pl.col("entity_id").alias("s1")).collect()
    out = OUT / "output"
    write_lists(pred, all_s1, "matched_entity_ids", out / "matching_results.tsv")
    shutil.copy(OUT / "submissions" / "submission_S005_M5_compete" / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    stats = dict(model="M8", sub_id=a.sub_id, calibration=dec["calibration"], decision=dec["decision"], rule=dec["threshold_rule"],
                 matched_s1=pred["s1"].n_unique(), predicted_pairs=len(pred), dense_f05=dec["dense_f05"])
    log.info("stats %s", stats)
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"), "--test-dir", str(DATA / "test")],
                       capture_output=True, text=True)
    rc = subprocess.run([sys.executable, "check_candidates.py", str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv")],
                        capture_output=True, text=True)
    log.info("validator exit=%d %s | candidates exit=%d %s", r.returncode, r.stdout[-500:], rc.returncode, rc.stdout[-200:])
    if r.returncode or rc.returncode:
        log.error("VALIDATION FAILED")
        sys.exit(1)
    sd = OUT / "submissions" / f"submission_{a.sub_id}"; sd.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out / f, sd / f)
    json.dump(stats | {"validator": r.stdout[-800:]}, open(sd / "manifest.json", "w"), indent=1, default=str)
    log.info("submission %s ready", a.sub_id)


if __name__ == "__main__":
    main()
