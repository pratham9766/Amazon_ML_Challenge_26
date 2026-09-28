"""After the primary submission is written: A/B extra training data; if promoted, S003 becomes the final output."""
import os
import subprocess
import sys
import time

from common import LOGS, OUT, get_logger

log = get_logger("optimize")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4")

while "FINISHED" not in (LOGS / "finish.log").read_text(encoding="utf-8", errors="ignore") if (LOGS / "finish.log").exists() else True:
    time.sleep(60)
log.info("primary submission done; starting A/B")
r = subprocess.run([PY, "ab_extra.py"], env=ENV)
log.info("ab_extra exit %d", r.returncode)
if r.returncode == 0 and (OUT / "models" / "M3F_full.txt").exists():
    r = subprocess.run([PY, "make_submission.py", "--model", "M3F", "--sub_id", "S003_M3F_all_1to1", "--one_to_one"], env=ENV)
    log.info("S003 exit %d", r.returncode)
    if r.returncode != 0:  # keep S002 as output if S003 failed validation
        subprocess.run([PY, "make_submission.py", "--model", "M2", "--sub_id", "S002_M2_all_1to1", "--one_to_one"], env=ENV)
log.info("OPTIMIZE DONE")
