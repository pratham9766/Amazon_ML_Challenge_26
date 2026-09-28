"""Channel unions, provenance, source budgets, and lightweight retrieval ranking."""
import csv
import gzip
from collections import defaultdict

WEIGHTS={"address_char":1.30,"address_word":1.20,"name_char":1.00,"name_word":0.90,
         "suffix_char":0.85,"exact_name":1.10,"numeric":0.70}

def channel_sets(result,queries,k,source_budget=False):
    out={}
    for q in queries:
        rows=result.get(q["id"],[])
        if source_budget:
            selected=[]
            for source in ("S2-","S3-"):selected.extend([r for r in rows if r["id"].startswith(source)][:k])
        else:selected=rows[:k]
        out[q["id"]]={r["id"] for r in selected}
    return out

def union_channels(results,queries,names):
    out={}
    for q in queries:
        merged=set()
        for name in names:merged.update(r["id"] for r in results[name].get(q["id"],[]))
        out[q["id"]]=merged
    return out

def rerank(results,queries,names,cap):
    out={};provenance={}
    for q in queries:
        merged=defaultdict(dict)
        for name in names:
            for row in results[name].get(q["id"],[]):merged[row["id"]][name]=row
        ranked=[]
        for target,channels in merged.items():
            rrf=sum(WEIGHTS[name]/(10+row["rank"]) for name,row in channels.items())
            rrf+=0.025*(len(channels)-1)
            ranked.append((rrf,target,channels))
        ranked.sort(key=lambda x:(-x[0],x[1]));selected=ranked[:cap]
        out[q["id"]]={x[1] for x in selected};provenance[q["id"]]=selected
    return out,provenance

def write_provenance(path,queries,provenance):
    fields=["source1_entity_id","candidate_entity_id","retrieval_score","retrieval_channel_count"]
    for name in WEIGHTS:fields += [f"from_{name}",f"{name}_rank",f"{name}_score"]
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t");w.writeheader()
        for q in queries:
            for score,target,channels in provenance[q["id"]]:
                row={"source1_entity_id":q["id"],"candidate_entity_id":target,"retrieval_score":score,"retrieval_channel_count":len(channels)}
                for name in WEIGHTS:
                    x=channels.get(name);row[f"from_{name}"]=int(x is not None)
                    row[f"{name}_rank"]=x["rank"] if x else "";row[f"{name}_score"]=x["score"] if x else ""
                w.writerow(row)
