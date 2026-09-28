"""Finisher v2: materialize India candidates once -> 3 feature shards -> S001, S002 (final) -> A/B extra data -> S003."""
import os
import subprocess
import sys

from common import OUT, get_logger

log = get_logger("finish")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4", RF_WORKERS="-1")
SH = dict(ENV, POLARS_MAX_THREADS="1", RF_WORKERS="1")


def run(args, env=ENV, fatal=True):
    log.info("RUN %s", " ".join(args))
    r = subprocess.run([PY] + args, env=env)
    log.info("exit %d", r.returncode)
    if r.returncode != 0 and fatal:
        sys.exit(r.returncode)
    return r.returncode


mat = OUT / "features" / "test" / "_cand_India.parquet"
if not mat.exists():
    run(["-c", "from pairs import candidates; import polars as pl; from pathlib import Path; "
               f"m=Path(r'{mat}'); t=m.with_suffix('.tmp'); "
               "candidates('test','India').collect().sort('s1').write_parquet(t, row_group_size=200_000); t.rename(m)"])
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
# optimization: A/B extra training data; if promoted S003 becomes output/, else S002 stays
if run(["ab_extra.py"], fatal=False) == 0 and (OUT / "models" / "M3F_full.txt").exists():
    if run(["make_submission.py", "--model", "M3F", "--sub_id", "S003_M3F_all_1to1", "--one_to_one"], fatal=False) != 0:
        run(["make_submission.py", "--model", "M2", "--sub_id", "S002_M2_all_1to1", "--one_to_one"])
log.info("OPTIMIZE DONE")
