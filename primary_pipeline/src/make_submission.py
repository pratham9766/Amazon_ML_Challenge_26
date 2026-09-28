"""Cached test features -> probabilities -> frozen entity logic -> validated submission.

usage: python make_submission.py --model M1 --sub_id S001 [--one_to_one]
Rejects (exit 1) if the official validator fails.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import OUT, VALIDATOR, DATA, Timer, get_logger, source_path
from train_matcher import decide

log = get_logger("submission")


def write_lists(pairs: pl.DataFrame, all_s1: pl.DataFrame, col: str, path):
    lists = pairs.group_by("s1").agg(pl.col("tgt").unique().sort().str.join(",").alias(col))
    out = all_s1.join(lists, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
    out = out.rename({"s1": "source1_entity_id"})
    # manual TSV writing: no quoting, exact header
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M1")
    ap.add_argument("--sub_id", default="S001")
    ap.add_argument("--one_to_one", action="store_true", help="each target kept only for its highest-probability S1")
    ap.add_argument("--score_only", action="store_true", help="only cache per-part probabilities")
    a = ap.parse_args()
    t = Timer()
    dec = json.load(open(OUT / "matcher" / f"{a.model}_decision.json"))
    booster = lgb.Booster(model_file=str(OUT / "models" / f"{a.model}_full.txt"))
    feats = dec["features"]
    probs_path = OUT / "predictions" / f"test_pair_probabilities_{a.model}.parquet"
    probs_path.parent.mkdir(exist_ok=True)
    stage1 = dec.get("stage1")
    if stage1:
        b1 = lgb.Booster(model_file=str(OUT / "models" / f"{stage1}_full.txt"))
        f1 = dec["stage1_features"]
    pcache = OUT / "predictions" / f"parts_{a.model}"
    pcache.mkdir(parents=True, exist_ok=True)
    from stage2 import ctx_features
    for pf in sorted((OUT / "features" / "test").glob("part_*.parquet")):
        cp = pcache / pf.name
        if cp.exists():
            continue
        if stage1:
            d = pl.read_parquet(pf, columns=["s1", "tgt", "src"] + f1)
            p1 = b1.predict(d.select(f1).to_numpy().astype(np.float32), num_threads=3).astype(np.float32)
            ctx = ctx_features(d.select("s1", "tgt", "src").with_columns(pl.Series("p", p1)))
            X = pl.concat([d.select(f1), ctx], how="horizontal").select(feats).to_numpy().astype(np.float32)
        else:
            d = pl.read_parquet(pf, columns=["s1", "tgt"] + feats)
            X = d.select(feats).to_numpy().astype(np.float32)
        p = booster.predict(X, num_threads=3)
        d.select("s1", "tgt").with_columns(pl.Series("p", p.astype(np.float32))).write_parquet(cp)
        log.info("%s scored %d", pf.name, len(d))
        del d, X
    if a.score_only:
        return
    # stream part by part: each part holds complete queries, so the per-query rule is exact per part
    all_s1 = pl.scan_parquet(source_path("test", 1)).select(pl.col("entity_id").alias("s1")).collect()
    outdir = OUT / "output"
    outdir.mkdir(exist_ok=True)
    preds, seen, n_cand = [], set(), 0
    with open(outdir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for cp in sorted(pcache.glob("part_*.parquet")):
            pr = pl.read_parquet(cp)
            preds.append(decide(pr, dec["t"], dec["rel"], dec["t_top"]).join(pr, on=["s1", "tgt"]))
            g = pr.group_by("s1").agg(pl.col("tgt").unique().sort().str.join(",").alias("c"))
            n_cand += len(pr)
            for s1_, c_ in g.iter_rows():
                fc.write(f"{s1_}\t{c_}\n")
            seen.update(g["s1"].to_list())
            del pr, g
        for s1_ in all_s1["s1"].to_list():
            if s1_ not in seen:
                fc.write(f"{s1_}\t\n")
    pred = pl.concat(preds)
    if a.one_to_one:
        pred = pred.sort("p", descending=True).unique("tgt", keep="first")
    pred = pred.select("s1", "tgt")
    write_lists(pred, all_s1, "matched_entity_ids", outdir / "matching_results.tsv")
    cand = pl.DataFrame({"n": [n_cand]})
    stats = dict(model=a.model, sub_id=a.sub_id, one_to_one=a.one_to_one, rule=dec | {"features": len(feats)},
                 s1=len(all_s1), matched_s1=pred["s1"].n_unique(), predicted_pairs=len(pred),
                 candidate_pairs=n_cand, mean_candidates=n_cand / len(all_s1))
    log.info("stats %s", stats)
    # official validator on the scored file (its candidate cross-check needs ~10 GB for 92.7M ids)
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(outdir / "matching_results.tsv"),
                        "--test-dir", str(DATA / "test")], capture_output=True, text=True)
    log.info("validator exit=%d %s", r.returncode, r.stdout[-3000:])
    rc = subprocess.run([sys.executable, "check_candidates.py", str(outdir / "matching_results.tsv"),
                         str(outdir / "candidate_pairs.tsv")], capture_output=True, text=True)
    log.info("candidate check exit=%d %s", rc.returncode, rc.stdout[-2000:])
    stats["validator"] = r.stdout[-1500:]
    stats["candidate_check"] = rc.stdout[-800:]
    r.returncode = r.returncode or rc.returncode
    if r.returncode != 0:
        log.error("VALIDATION FAILED - submission rejected")
        sys.exit(1)
    sd = OUT / "submissions" / f"submission_{a.sub_id}"
    sd.mkdir(parents=True, exist_ok=True)
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(outdir / f, sd / f)
    json.dump(stats, open(sd / "manifest.json", "w"), indent=1, default=str)
    log.info("submission %s ready in %.0fs", a.sub_id, t())


if __name__ == "__main__":
    main()
