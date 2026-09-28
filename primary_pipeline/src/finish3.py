"""Final chain: S001 -> S002 (final output/) -> A/B extra data -> S003 if promoted."""
import os
import subprocess
import sys

from common import OUT, get_logger

log = get_logger("finish")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4")


def run(args, fatal=True):
    log.info("RUN %s", " ".join(args))
    r = subprocess.run([PY] + args, env=ENV)
    log.info("exit %d", r.returncode)
    if r.returncode != 0 and fatal:
        sys.exit(r.returncode)
    return r.returncode


run(["make_submission.py", "--model", "M2", "--sub_id", "S001_M2_all"])
run(["make_submission.py", "--model", "M2", "--sub_id", "S002_M2_all_1to1", "--one_to_one"])
log.info("FINISHED")
if run(["ab_extra.py"], fatal=False) == 0 and (OUT / "models" / "M3F_full.txt").exists():
    if run(["make_submission.py", "--model", "M3F", "--sub_id", "S003_M3F_all_1to1", "--one_to_one"], fatal=False) != 0:
        run(["make_submission.py", "--model", "M2", "--sub_id", "S002_M2_all_1to1", "--one_to_one"])
log.info("OPTIMIZE DONE")
