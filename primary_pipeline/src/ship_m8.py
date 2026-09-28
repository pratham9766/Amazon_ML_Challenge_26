"""Wait for M8 final fit, set the singleton-safe decision (iso + threshold), build S008."""
import json, os, subprocess, sys, time
from common import LOGS, OUT, get_logger
log = get_logger("ship_m8")
while "M8 chosen" not in (LOGS / "m8.log").read_text(encoding="utf-8", errors="ignore"):
    time.sleep(15)
p = OUT / "matcher" / "M8_decision.json"
d = json.load(open(p))
d.update(calibration="iso", decision="threshold", threshold_rule=[0.68, None],
         note="EF05 was +0.0001 but singleton acc 0.963 vs 0.979; threshold chosen for singleton safety / France robustness")
json.dump(d, open(p, "w"), indent=1)
log.info("decision set: iso + threshold 0.68 + one-to-one")
r = subprocess.run([sys.executable, "submit_m8.py", "--sub_id", "S008_M8_ownership"],
                   env=dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4"))
log.info("S008 exit %d", r.returncode)
if r.returncode:  # keep S005 as output
    import shutil
    s = OUT / "submissions" / "submission_S005_M5_compete"
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(s / f, OUT / "output" / f)
    log.info("restored S005 in output/")
log.info("SHIP M8 DONE")
