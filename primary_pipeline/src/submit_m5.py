"""Test submission with the competition stage 2 (M5). Candidates are unchanged (reused from S002).

usage: python submit_m5.py --sub_id S005_M5
"""
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
from compete import comp_from_stats, tgt_stats
from make_submission import write_lists
from stage2 import ctx_features
from train_matcher import decide

log = get_logger("submit_m5")


def ikey(col):  # "S2-123" -> 2*10**11 + 123 (compact int keys for big group-bys)
    return (pl.col(col).str.slice(1, 1).cast(pl.Int64) * 10**11 + pl.col(col).str.slice(3).cast(pl.Int64)).alias(f"{col}_k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub_id", default="S005_M5")
    ap.add_argument("--model", default="M5")
    a = ap.parse_args()
    dec = json.load(open(OUT / "matcher" / f"{a.model}_decision.json"))
    m5 = lgb.Booster(model_file=str(OUT / "models" / f"{a.model}_full.txt"))
    f1 = dec["stage1_features"]
    preds = []
    pdir = OUT / "predictions" / f"final_{a.model}"
    pdir.mkdir(parents=True, exist_ok=True)
    for c in ("France", "India", "US"):
        cache = pdir / f"{c}.parquet"
        if cache.exists():
            preds.append(pl.read_parquet(cache))
            log.info("%s cached", c)
            continue
        pparts = sorted(glob.glob(str(OUT / "predictions" / f"parts_{dec['stage1']}" / f"part_{c}_*.parquet")))
        top = tgt_stats(pl.concat([pl.read_parquet(p).select("tgt", "p") for p in pparts]))
        cpreds = []
        for pp in pparts:
            name = pp.split("\\")[-1].split("/")[-1]
            pr = pl.read_parquet(pp)
            fe = pl.read_parquet(OUT / "features" / "test" / name, columns=["s1", "tgt", "src"] + f1)
            d = fe.join(pr, on=["s1", "tgt"])
            comp = comp_from_stats(d.select("s1", "tgt", "p"), top)
            ctx = ctx_features(d.select("s1", "tgt", "src", "p"))
            X = pl.concat([d.select(f1), ctx, comp], how="horizontal").select(dec["features"]).to_numpy().astype(np.float32)
            p5 = d.select("s1", "tgt").with_columns(pl.Series("p", m5.predict(X, num_threads=4).astype(np.float32)))
            sp = OUT / "predictions" / f"parts_{a.model}"
            sp.mkdir(parents=True, exist_ok=True)
            p5.write_parquet(sp / name)  # all-candidate probabilities: stage-1 input for the next round
            cpreds.append(decide(p5, dec["t"], 0.0, dec["t_top"]).join(p5, on=["s1", "tgt"]))
            del pr, fe, d, X
        cp = pl.concat(cpreds)
        cp.write_parquet(cache)
        preds.append(cp)
        log.info("%s done", c)
        del top
    pred = pl.concat(preds)
    if dec["one_to_one"]:
        pred = pred.sort("p", descending=True).unique("tgt", keep="first")
    pred = pred.select("s1", "tgt")
    all_s1 = pl.scan_parquet(source_path("test", 1)).select(pl.col("entity_id").alias("s1")).collect()
    out = OUT / "output"
    write_lists(pred, all_s1, "matched_entity_ids", out / "matching_results.tsv")
    shutil.copy(OUT / "submissions" / "submission_S002_M2_all_1to1" / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    stats = dict(model=a.model, sub_id=a.sub_id, rule={k: dec[k] for k in ("t", "t_top", "one_to_one")},
                 matched_s1=pred["s1"].n_unique(), predicted_pairs=len(pred), dense_f05=dec.get("dense_f05"))
    log.info("stats %s", stats)
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"), "--test-dir", str(DATA / "test")],
                       capture_output=True, text=True)
    rc = subprocess.run([sys.executable, "check_candidates.py", str(out / "matching_results.tsv"), str(out / "candidate_pairs.tsv")],
                        capture_output=True, text=True)
    log.info("validator exit=%d %s | candidates exit=%d %s", r.returncode, r.stdout[-600:], rc.returncode, rc.stdout[-300:])
    if r.returncode or rc.returncode:
        log.error("VALIDATION FAILED")
        sys.exit(1)
    sd = OUT / "submissions" / f"submission_{a.sub_id}"
    sd.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out / f, sd / f)
    json.dump(stats | {"validator": r.stdout[-800:]}, open(sd / "manifest.json", "w"), indent=1, default=str)
    log.info("submission %s ready", a.sub_id)


if __name__ == "__main__":
    main()
