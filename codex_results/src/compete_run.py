"""dense shards -> merge -> dense features -> M5 (competition stage 2) -> S005 if it beats M2 on dense."""
import glob
import json
import os
import subprocess
import sys
import time

import polars as pl

from common import BLOCKER, LOGS, OUT, get_logger

log = get_logger("compete_run")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4", RF_WORKERS="-1")
d = BLOCKER / "dense"
need = [d / f"_shard_comb_t_{c}_S{s}_{i}of3.parquet" for c in ("India", "US") for s in (2, 3) for i in range(3)]
while not (d / "comb_t.parquet").exists() and not all(p.exists() for p in need):
    time.sleep(30)
time.sleep(10)
if not (d / "comb_t.parquet").exists():
    pl.scan_parquet([str(p) for p in need]).sink_parquet(d / "comb_t.parquet")
    log.info("merged dense shards")
for args in (["pairs.py", "--split", "train", "--tag", "dense", "--qchunk", "10000"], ["compete.py"]):
    log.info("RUN %s", args)
    r = subprocess.run([PY] + args, env=ENV)
    log.info("exit %d", r.returncode)
    if r.returncode:
        sys.exit(1)
dec = json.load(open(OUT / "matcher" / "M5_decision.json"))
gain = dec["dense_f05"] - max(dec["dense_ref_M2"], dec["dense_ref_M2_1to1"])
log.info("M5 dense gain vs best M2 variant: %+.5f", gain)
if gain > 0.002:
    while "OPTIMIZE DONE" not in (LOGS / "finish.log").read_text(encoding="utf-8", errors="ignore"):
        time.sleep(30)
    r = subprocess.run([PY, "submit_m5.py", "--sub_id", "S005_M5_compete"], env=ENV)
    log.info("S005 exit %d", r.returncode)
log.info("COMPETE DONE")
