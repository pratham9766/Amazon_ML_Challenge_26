"""Exact normalized-name channels with address-aware collision ranking."""
import gzip,heapq,json,logging,time
from collections import defaultdict
from .normalize import basic,suffix_stripped,tokens,numeric_tokens

def retrieve_exact(db,queries,out,config):
    cache=out/"artifacts"/"candidates_exact_name.json.gz"
    if cache.exists():
        with gzip.open(cache,"rt",encoding="utf-8") as f:return json.load(f)
    started=time.monotonic();maps={"basic":defaultdict(list),"suffix":defaultdict(list)}
    qaddress=[]
    for i,q in enumerate(queries):
        maps["basic"][(q["country"],basic(q["name"]))].append(i)
        maps["suffix"][(q["country"],suffix_stripped(q["name"]))].append(i)
        qaddress.append(set(tokens(q["address"])))
    heaps=[[] for _ in queries];topk=min(100,config["channel_top_k"]);processed=0
    for target,name,address,country,nn,sn in db.execute("SELECT id,name,address,country,nn,sn FROM records WHERE file IN ('train_source2','train_source3')"):
        taddr=set(tokens(address))
        found=set(maps["basic"].get((country,nn),()))|set(maps["suffix"].get((country,sn),()))
        for qi in found:
            inter=len(qaddress[qi]&taddr);union=len(qaddress[qi]|taddr)
            score=2.0+(inter/union if union else 0)+(nn==basic(queries[qi]["name"]))
            item=(score,target);heap=heaps[qi]
            if len(heap)<topk:heapq.heappush(heap,item)
            elif item>heap[0]:heapq.heapreplace(heap,item)
        processed+=1
    result={q["id"]:[{"id":target,"score":score,"rank":r+1,"channel":"exact_name"} for r,(score,target) in enumerate(sorted(h,reverse=True))] for q,h in zip(queries,heaps)}
    result["__metadata__"]={"runtime_seconds":time.monotonic()-started,"targets_scanned":processed,"cap":topk,"ranking":"exact basic/suffix name, then address-token Jaccard"}
    with gzip.open(cache,"wt",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False)
    return result
