"""Partial submission from whatever test features exist now (does not touch make_submission.py caches).

Scores every features/test/part_{country}_*.parquet with the frozen model (reusing
predictions/{country}_{model}_probs.parquet when it already covers all parts), applies the
frozen entity rule, and writes both TSVs. S1s without features get empty rows.

usage: python make_partial_submission.py --model M2 --sub_id P001 [--one_to_one]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import OUT, VALIDATOR, DATA, get_logger, source_path
from make_submission import write_lists
from train_matcher import decide

log = get_logger("partial_submission")
FEAT = OUT / "features" / "test"


def country_probs(country: str, model: str, feats, booster) -> pl.DataFrame:
    parts = sorted(FEAT.glob(f"part_{country}_*.parquet"))
    n_q = pl.scan_parquet(parts).select(pl.col("s1").n_unique()).collect().item()
    partial = OUT / "predictions" / f"{country}_{model}_probs_partial{len(parts)}.parquet"
    for cached in (OUT / "predictions" / f"{country}_{model}_probs.parquet", partial):
        if cached.exists():
            pr = pl.read_parquet(cached).with_columns(pl.col("p").cast(pl.Float32))
            if pr["s1"].n_unique() == n_q:
                log.info("%s: reusing %s (%d queries)", country, cached.name, n_q)
                return pr
    out = []
    for pf in parts:
        d = pl.read_parquet(pf, columns=["s1", "tgt"] + feats)
        p = booster.predict(d.select(feats).to_numpy().astype(np.float32), num_threads=2)
        out.append(d.select("s1", "tgt").with_columns(pl.Series("p", p.astype(np.float32))))
        log.info("%s scored %d", pf.name, len(d))
    pr = pl.concat(out)
    pr.write_parquet(partial)
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M2")
    ap.add_argument("--sub_id", default="P001")
    ap.add_argument("--one_to_one", action="store_true")
    a = ap.parse_args()
    dec = json.load(open(OUT / "matcher" / f"{a.model}_decision.json"))
    assert not dec.get("stage1"), "stage-2 models not supported here"
    booster = lgb.Booster(model_file=str(OUT / "models" / f"{a.model}_full.txt"))
    countries = sorted({p.name.split("_")[1] for p in FEAT.glob("part_*_*.parquet")})
    pr = pl.concat([country_probs(c, a.model, dec["features"], booster) for c in countries])
    pred = decide(pr, dec["t"], dec["rel"], dec["t_top"])
    if a.one_to_one:
        pred = pred.join(pr, on=["s1", "tgt"]).sort("p", descending=True).unique("tgt", keep="first").select("s1", "tgt")
    # candidates = exactly the pairs the model scored
    cand = pr.select("s1", "tgt")
    all_s1 = pl.scan_parquet(source_path("test", 1)).select(pl.col("entity_id").alias("s1"), "country").collect()
    sd = OUT / "submissions" / f"submission_{a.sub_id}"
    sd.mkdir(parents=True, exist_ok=True)
    write_lists(pred, all_s1.select("s1"), "matched_entity_ids", sd / "matching_results.tsv")
    write_lists(cand, all_s1.select("s1"), "candidate_entity_ids", sd / "candidate_pairs.tsv")
    cov = (all_s1.with_columns(pl.col("s1").is_in(pr["s1"].unique().implode()).alias("scored"),
                               pl.col("s1").is_in(pred["s1"].unique().implode()).alias("matched"))
           .group_by("country").agg(pl.len().alias("s1"), pl.col("scored").sum(), pl.col("matched").sum()).sort("country"))
    stats = dict(model=a.model, sub_id=a.sub_id, one_to_one=a.one_to_one, partial=True,
                 rule={k: dec[k] for k in ("t", "rel", "t_top")}, predicted_pairs=len(pred),
                 candidate_pairs=len(cand), coverage=cov.to_dicts())
    log.info("stats %s", stats)
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(sd / "matching_results.tsv"),
                        "--candidate", str(sd / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")],
                       capture_output=True, text=True)
    log.info("validator exit=%d\n%s", r.returncode, r.stdout[-3000:])
    stats["validator"] = r.stdout[-1500:]
    json.dump(stats, open(sd / "manifest.json", "w"), indent=1, default=str)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
