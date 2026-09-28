"""Fair M5 vs M6 on one subset -> round-2 competition on the winner -> best validated submission in output/."""
import json
import os
import subprocess
import sys
import time

from common import LOGS, OUT, get_logger

log = get_logger("refine")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4")


def run(args, fatal=True):
    log.info("RUN %s", " ".join(args))
    r = subprocess.run([PY] + args, env=ENV)
    log.info("exit %d", r.returncode)
    if r.returncode and fatal:
        sys.exit(r.returncode)
    return r.returncode


def f05(name):
    return json.load(open(OUT / "matcher" / f"{name}_decision.json"))["dense_f05"]


while len(list((OUT / "predictions" / "parts_M3F").glob("part_*.parquet"))) < 71:
    time.sleep(30)
time.sleep(20)
for st, nm in (("M2", "M5B"), ("M3F", "M6B")):
    if not (OUT / "matcher" / f"{nm}_decision.json").exists():
        run(["compete.py", "--stage1", st, "--name", nm])
w = max(("M5B", "M6B"), key=f05)
log.info("fair subset B: M5B %.5f  M6B %.5f -> winner %s", f05("M5B"), f05("M6B"), w)
run(["compete.py", "--stage1", w, "--probs", "--name", "R2"])
log.info("round 2: R2 %.5f vs %s %.5f", f05("R2"), w, f05(w))
ship_w = w == "M6B" and f05("M6B") > f05("M5B") + 0.001
ship_r2 = f05("R2") > f05(w) + 0.001
if ship_w or ship_r2:  # R2's test stage-1 input is the winner's all-candidate probabilities
    if run(["submit_m5.py", "--model", w, "--sub_id", f"S006_{w}"], fatal=False):
        sys.exit(1)
if ship_r2:
    if run(["submit_m5.py", "--model", "R2", "--sub_id", f"S007_R2_on_{w}"], fatal=False) and ship_w is False:
        run(["-c", "import shutil;from common import OUT;s=OUT/'submissions'/'submission_S005_M5_compete';"
                   "[shutil.copy(s/f, OUT/'output'/f) for f in ('matching_results.tsv','candidate_pairs.tsv')]"], fatal=False)
log.info("REFINE DONE (ship_w=%s ship_r2=%s)", ship_w, ship_r2)
