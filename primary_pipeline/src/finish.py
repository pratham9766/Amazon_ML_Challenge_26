"""Unattended finisher: wait for India S3 retrieval -> India features -> score -> S001 + S002 (final)."""
import os
import subprocess
import sys
import time

from common import BLOCKER, OUT, get_logger

log = get_logger("finish")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4", RF_WORKERS="-1")


def run(args):
    log.info("RUN %s", " ".join(args))
    r = subprocess.run([PY] + args, env=ENV)
    log.info("exit %d", r.returncode)
    if r.returncode != 0:
        sys.exit(r.returncode)


import polars as pl
part = BLOCKER / "test" / "_part_comb_t_India_S3.parquet"
shards = [BLOCKER / "test" / f"_shard_comb_t_India_S3_{i}of3.parquet" for i in range(3)]
while not part.exists() and not all(s.exists() for s in shards):
    time.sleep(30)
time.sleep(15)
if not part.exists():
    tmp = part.with_suffix(".tmp")
    pl.concat([pl.read_parquet(s) for s in shards]).write_parquet(tmp)
    tmp.rename(part)
    log.info("merged India S3 shards")
SH = dict(ENV, POLARS_MAX_THREADS="1", RF_WORKERS="1")
base = ["pairs.py", "--split", "test", "--tag", "test", "--country", "India", "--qchunk", "25000"]
procs = [subprocess.Popen([PY] + base + ["--shard", f"{i}/3", "--no_cand"], env=SH) for i in range(3)]
codes = [p.wait() for p in procs]
log.info("shard exits %s", codes)
if any(codes):
    sys.exit(1)
run(base)  # all parts exist -> only writes candidates_India.parquet
run(["make_submission.py", "--model", "M2", "--sub_id", "S001_M2_all"])
run(["make_submission.py", "--model", "M2", "--sub_id", "S002_M2_all_1to1", "--one_to_one"])  # final output/
log.info("FINISHED")
