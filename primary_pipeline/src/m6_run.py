"""After S005: M3F test probabilities + M6 (competition on M3F) in parallel -> S006 if M6 beats M5 on dense."""
import json
import os
import subprocess
import sys
import time

from common import LOGS, OUT, get_logger

log = get_logger("m6_run")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="2")
while True:
    t = (LOGS / "submit_m5.log").read_text(encoding="utf-8", errors="ignore") if (LOGS / "submit_m5.log").exists() else ""
    if "submission S005_M5_compete ready" in t or "VALIDATION FAILED" in t:
        break
    time.sleep(20)
log.info("S005 stage over; starting M3F scoring + M6 training")
a = subprocess.Popen([PY, "make_submission.py", "--model", "M3F", "--score_only"], env=ENV)
b = subprocess.Popen([PY, "compete.py", "--stage1", "M3F", "--name", "M6"], env=ENV)
ra, rb = a.wait(), b.wait()
log.info("M3F scoring exit %d, M6 exit %d", ra, rb)
if ra or rb:
    sys.exit(1)
m5 = json.load(open(OUT / "matcher" / "M5_decision.json"))["dense_f05"]
m6 = json.load(open(OUT / "matcher" / "M6_decision.json"))["dense_f05"]
log.info("dense: M5 %.5f vs M6 %.5f", m5, m6)
if m6 > m5 + 0.001:
    r = subprocess.run([PY, "submit_m5.py", "--model", "M6", "--sub_id", "S006_M6_compete_M3F"], env=dict(ENV, POLARS_MAX_THREADS="4"))
    log.info("S006 exit %d", r.returncode)
    if r.returncode:  # restore S005 in output/
        subprocess.run([PY, "-c", "import shutil;from common import OUT;s=OUT/'submissions'/'submission_S005_M5_compete';"
                        "[shutil.copy(s/f, OUT/'output'/f) for f in ('matching_results.tsv','candidate_pairs.tsv')]"], env=ENV)
log.info("M6 DONE")
