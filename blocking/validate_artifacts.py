"""Fast integrity checks for the blocking benchmark deliverables."""
import csv
import gzip
import json
from pathlib import Path

root = Path(__file__).resolve().parent
with gzip.open(root / "artifacts/benchmark_queries.json.gz", "rt", encoding="utf-8") as f:
    benchmark = json.load(f)
dev = benchmark["development"]
confirm = benchmark["confirmation"]
assert len(dev) == 20_000 and len(confirm) == 7_500
dev_ids = {q["id"] for q in dev}
confirm_ids = {q["id"] for q in confirm}
assert len(dev_ids) == len(dev) and len(confirm_ids) == len(confirm)
assert dev_ids.isdisjoint(confirm_ids)

with (root / "artifacts/benchmark_summary.json").open(encoding="utf-8") as f:
    summary = json.load(f)
assert summary["confirmation"]["queries"] == 7_500
assert summary["selected_cap"] in {50, 75, 100, 150, 200}

rows = 0
seen_queries = set()
with gzip.open(root / "artifacts/confirmation_candidate_provenance.tsv.gz", "rt", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        assert row["source1_entity_id"] in confirm_ids
        assert row["candidate_entity_id"].startswith(("S2-", "S3-"))
        seen_queries.add(row["source1_entity_id"])
        rows += 1
assert rows > 0 and seen_queries
print(json.dumps({"development_queries": len(dev), "confirmation_queries": len(confirm),
                  "provenance_rows": rows, "queries_with_candidates": len(seen_queries),
                  "selected_cap": summary["selected_cap"], "status": "ok"}))
