"""Streaming check of candidate_pairs.tsv with the official validator's rules, in bounded memory.

The official validator's candidate cross-check holds every candidate ID in Python sets (~92.7M here),
which does not fit in 8 GB. This enforces the same rules row by row: exact header, tab separation,
one row per test S1 (all present, no duplicates), only S2-/S3- IDs, no duplicate IDs within a row,
and every matched ID contained in that S1's candidate list (the official check only warns on this).
usage: python check_candidates.py <matching_results.tsv> <candidate_pairs.tsv>
"""
import sys

import polars as pl

from common import source_path


def main(matching, candidates):
    required = set(pl.scan_parquet(source_path("test", 1)).select("entity_id").collect()["entity_id"].to_list())
    matches = {}
    with open(matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if rest:
                matches[s1] = rest.split(",")
    errors, seen, not_subset = [], set(), 0
    with open(candidates, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            errors.append(f"bad header {header}")
        for n, line in enumerate(f, start=2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errors.append(f"line {n}: no tab")
                continue
            if s1 in seen:
                errors.append(f"duplicate row {s1}")
            seen.add(s1)
            ids = rest.split(",") if rest else []
            if len(set(ids)) != len(ids):
                errors.append(f"duplicate ids in row {s1}")
            if any(not (i.startswith("S2-") or i.startswith("S3-")) for i in ids):
                errors.append(f"bad prefix in row {s1}")
            m = matches.get(s1)
            if m and not set(m) <= set(ids):
                not_subset += 1
            if len(errors) > 20:
                break
    missing = required - seen
    extra = seen - required
    if missing:
        errors.append(f"{len(missing)} test S1 missing")
    if extra:
        errors.append(f"{len(extra)} unknown S1 rows")
    if not_subset:
        errors.append(f"{not_subset} S1 have matches outside their candidates")
    print(f"candidate rows {len(seen)}; errors {len(errors)}")
    for e in errors[:20]:
        print("  ", e)
    print("CANDIDATES PASS" if not errors else "CANDIDATES FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
