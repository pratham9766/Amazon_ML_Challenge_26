"""Scale training data: 150k more train queries -> features -> A/B (fold 0) -> M4F -> S004 (final output/ if promoted)."""
import glob
import os
import subprocess
import sys
import time

import polars as pl

from common import BLOCKER, CACHE, LOGS, OUT, get_logger

log = get_logger("more")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4")


def run(args, env=ENV, fatal=True):
    log.info("RUN %s", " ".join(args))
    r = subprocess.run([PY] + args, env=env)
    log.info("exit %d", r.returncode)
    if r.returncode and fatal:
        sys.exit(r.returncode)
    return r.returncode


# wait for S003 stage to finish (RAM)
while "OPTIMIZE DONE" not in (LOGS / "finish.log").read_text(encoding="utf-8", errors="ignore"):
    time.sleep(30)
qp = CACHE / "more_queries.parquet"
if not qp.exists():
    s = pl.read_parquet(CACHE / "train_gt_s1.parquet")
    used = pl.concat([pl.read_parquet(CACHE / f"{n}_queries.parquet").select("s1") for n in ("bench", "extra")])
    s.join(used, on="s1", how="anti").sample(150000, seed=7).with_columns(pl.lit("more").alias("split")).write_parquet(qp)
out = BLOCKER / "more"
if not (out / "name_na.parquet").exists():
    run(["run_channels.py", "--queries", "more", "--tag", "more", "--channels", "name_na", "--k", "3", "--batch", "1000"])
if not (out / "comb_t.parquet").exists():
    base = ["run_channels.py", "--queries", "more", "--tag", "more", "--channels", "comb_t", "--k", "25", "--batch", "500",
            "--countries", "India,US"]
    procs = [subprocess.Popen([PY] + base + ["--qshard", f"{i}/2"], env=ENV) for i in range(2)]
    codes = [p.wait() for p in procs]
    log.info("retrieval shard exits %s", codes)
    if any(codes):
        sys.exit(1)
    pl.scan_parquet(sorted(glob.glob(str(out / "_shard_comb_t_*.parquet")))).sink_parquet(out / "comb_t.parquet")
run(["pairs.py", "--split", "train", "--tag", "more", "--qchunk", "10000"], env=dict(ENV, RF_WORKERS="-1"))
run(["ab_more.py"])
if (OUT / "models" / "M4F_full.txt").exists():
    if run(["make_submission.py", "--model", "M4F", "--sub_id", "S004_M4F_all_1to1", "--one_to_one"], fatal=False):
        log.info("S004 failed validation; restoring S003")
        run(["make_submission.py", "--model", "M3F", "--sub_id", "S003_M3F_all_1to1", "--one_to_one"], fatal=False)
log.info("MORE DONE")
