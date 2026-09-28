"""Second dense labelled region: retrieval (1 low-priority process) -> features."""
import os, subprocess, sys
from common import BLOCKER, get_logger
log = get_logger("dense2")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="2", RF_WORKERS="2")
for args in (["run_channels.py", "--queries", "dense2", "--tag", "dense2", "--channels", "name_na", "--k", "3", "--batch", "1000"],
             ["run_channels.py", "--queries", "dense2", "--tag", "dense2", "--channels", "comb_t", "--k", "25", "--batch", "500"],
             ["pairs.py", "--split", "train", "--tag", "dense2", "--qchunk", "10000"]):
    log.info("RUN %s", args)
    r = subprocess.run([PY] + args, env=ENV)
    log.info("exit %d", r.returncode)
    if r.returncode:
        sys.exit(1)
log.info("DENSE2 DONE")
