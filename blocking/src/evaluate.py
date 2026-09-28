"""Blocking metrics with subgroup breakdowns."""
from collections import defaultdict
import numpy as np

def _stats(counts):
    a=np.asarray(counts,dtype=float)
    return {"mean":float(a.mean()),"median":float(np.median(a)),"p90":float(np.percentile(a,90)),
            "p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":int(a.max())}

def evaluate(queries,candidates):
    hit_total=truth_total=0;complete=[];atleast=[];counts=[];single=[];groups=defaultdict(lambda:[0,0,0,0])
    for q in queries:
        got=set(candidates.get(q["id"],()))
        truth=set(q["matches"]);hits=len(got&truth)
        hit_total+=hits;truth_total+=len(truth);counts.append(len(got))
        if truth:complete.append(hits==len(truth));atleast.append(hits>0)
        else:single.append(len(got))
        labels=["country="+q["country"],"profile="+q["source_profile"],"multiplicity="+q["multiplicity_group"]]
        if q["short_name"]:labels.append("short_name")
        if q["missing_target_address"]:labels.append("missing_target_address")
        if q["nonascii_target_name"]:labels.append("nonascii_target_name")
        if q["weak_lexical"]:labels.append("weak_lexical")
        if q.get("high_repeated_name"):labels.append("high_repeated_name")
        for label in labels:
            g=groups[label];g[0]+=hits;g[1]+=len(truth);g[2]+=bool(truth) and hits==len(truth);g[3]+=bool(truth)
        for source in (2,3):
            st={x for x in truth if x.startswith(f"S{source}-")};h=len(got&st);g=groups[f"source=S{source}"]
            g[0]+=h;g[1]+=len(st);g[2]+=bool(st) and h==len(st);g[3]+=bool(st)
    return {"queries":len(queries),"true_pairs":truth_total,"pair_recall":hit_total/truth_total if truth_total else None,
            "complete_entity_recall":float(np.mean(complete)) if complete else None,"at_least_one_recall":float(np.mean(atleast)) if atleast else None,
            "candidates":_stats(counts),"singleton_mean":float(np.mean(single)) if single else None,
            "singleton_p95":float(np.percentile(single,95)) if single else None,
            "breakdowns":{k:{"pair_recall":v[0]/v[1] if v[1] else None,"complete_entity_recall":v[2]/v[3] if v[3] else None,"pairs":v[1],"matched_entities":v[3]} for k,v in groups.items()}}
