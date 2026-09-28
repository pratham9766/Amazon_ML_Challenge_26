"""Wait for dense region 2 + refine round 2 -> M8 (vs best refine model on same region-1 queries) -> S008 if +0.001."""
import json, os, subprocess, sys, time
from common import LOGS, OUT, get_logger
log = get_logger("m8_run")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_MAX_THREADS="4")
txt = lambda f: (LOGS / f).read_text(encoding="utf-8", errors="ignore") if (LOGS / f).exists() else ""
while "DENSE2 DONE" not in txt("dense2.log") or "round 2:" not in txt("refine.log"):
    if "exit 1" in txt("dense2.log"):
        log.info("dense2 failed; continuing with region 1 only"); break
    time.sleep(30)
f = lambda n: json.load(open(OUT / "matcher" / f"{n}_decision.json"))["dense_f05"]
ref = max(("M5B", "M6B", "R2"), key=lambda n: f(n) if (OUT / "matcher" / f"{n}_decision.json").exists() else -1)
log.info("reference model %s (%.5f)", ref, f(ref))
r = subprocess.run([PY, "m8.py", "--stage1", "M3F", "--ref", ref], env=ENV)
log.info("m8 exit %d", r.returncode)
if r.returncode:
    sys.exit(1)
d = json.load(open(OUT / "matcher" / "M8_decision.json"))
gain = (d["region1_f05"] or 0) - d["ref_region1"]
log.info("M8 region1 %.5f vs %s %.5f (gain %+.5f)", d["region1_f05"] or 0, ref, d["ref_region1"], gain)
if gain > 0.001:
    while "REFINE DONE" not in txt("refine.log"):
        time.sleep(30)
    r = subprocess.run([PY, "submit_m8.py", "--sub_id", "S008_M8_ownership"], env=ENV)
    log.info("S008 exit %d", r.returncode)
log.info("M8 DONE")
