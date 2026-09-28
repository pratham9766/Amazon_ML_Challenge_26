#!/usr/bin/env bash
# Health monitor: emits one line per milestone / error / stall / crash / low-memory event.
L="D:/RESOURCES/AmazonMLChallenge/codex_results/logs"
declare -A OFF
MIL='fold [0-9]|DENSE\[B\]|fair subset|round 2:|REFINE DONE|DENSE2 DONE|M8 results|M8 chosen|M8 region1|M8 DONE|SHIP M8 DONE|decision set|restored S005|submission .* ready|stats \{|exit [0-9]+|RUN |VALIDATION FAILED|merged|reference model|scored|done$'
ERR='Traceback|MemoryError|memory allocation|VALIDATION FAILED|INFO exit [1-9]|Error'
last_cpu=-1; still=0; prev_n=-1
while true; do
  for f in refine.log compete.log dense2.log m8_run.log m8.log ship_m8.log submit_m5.log submit_m8.log channels.log pairs.log \
           refine.err dense2.err m8_run.err ship_m8.err; do
    p="$L/$f"; [ -f "$p" ] || continue
    sz=$(stat -c %s "$p"); o=${OFF[$f]:-$sz}
    if [ "$sz" -gt "$o" ]; then
      tail -c +$((o+1)) "$p" | grep -E "$MIL|$ERR" | grep -v "Warning\|Deprecat" | cut -c1-260 | sed "s/^/[$f] /"
    fi
    OFF[$f]=$sz
  done
  read n cpu free < <(powershell -NoProfile -c "\$p=Get-Process python -ErrorAction SilentlyContinue; \$c=(\$p | Measure-Object CPU -Sum).Sum; \$f=[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB); '{0} {1:F0} {2}' -f @(\$p).Count, \$c, \$f" 2>/dev/null)
  [ -z "$n" ] && { sleep 60; continue; }
  if [ "$prev_n" -ge 0 ] && [ "$n" -lt "$prev_n" ]; then echo "[health] python process count dropped $prev_n -> $n (a step finished or crashed)"; fi
  prev_n=$n
  if [ "$cpu" = "$last_cpu" ] && [ "$n" -gt 0 ]; then still=$((still+1)); else still=0; fi
  last_cpu=$cpu
  [ "$still" -eq 10 ] && echo "[health] STALL: no python CPU progress for 10 min ($n procs)"
  [ "$n" -eq 0 ] && ! grep -q "SHIP M8 DONE" "$L/ship_m8.log" 2>/dev/null && echo "[health] WARNING: no python processes running but pipeline not finished"
  [ "${free:-9999}" -lt 300 ] && echo "[health] LOW MEMORY: ${free} MB free"
  sleep 60
done
