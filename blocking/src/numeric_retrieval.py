"""Selective numeric-address retrieval with explicit candidate cap."""
import gzip,heapq,json,time
from collections import Counter,defaultdict
from .normalize import numeric_features,word_features

def retrieve_numeric(db,queries,out,config):
    cache=out/"artifacts"/"candidates_numeric.json.gz"
    if cache.exists():
        with gzip.open(cache,"rt",encoding="utf-8") as f:return json.load(f)
    started=time.monotonic();postings=defaultdict(list);qaddr=[]
    for i,q in enumerate(queries):
        qaddr.append(word_features(q["address"]))
        for f in numeric_features(q["address"]):postings[f].append(i)
    postings={f:v for f,v in postings.items() if len(v)<=config["max_queries_per_feature"]}
    heaps=[[] for _ in queries];topk=min(75,config["channel_top_k"]);processed=0
    for target,name,address,country in db.execute("SELECT id,name,address,country FROM records WHERE file IN ('train_source2','train_source3')"):
        acc=Counter()
        for f in numeric_features(address):
            for qi in postings.get(f,()):
                if queries[qi]["country"]==country:acc[qi]+=2 if f.startswith("np:") else 1
        taddr=word_features(address)
        for qi,nscore in acc.items():
            # Numeric evidence is reranked by lexical address overlap; never unrestricted.
            lexical=len(qaddr[qi]&taddr)/max(1,len(qaddr[qi]|taddr));item=(nscore+lexical,target)
            h=heaps[qi]
            if len(h)<topk:heapq.heappush(h,item)
            elif item>h[0]:heapq.heapreplace(h,item)
        processed+=1
    result={q["id"]:[{"id":target,"score":score,"rank":r+1,"channel":"numeric"} for r,(score,target) in enumerate(sorted(h,reverse=True))] for q,h in zip(queries,heaps)}
    result["__metadata__"]={"runtime_seconds":time.monotonic()-started,"targets_scanned":processed,"cap":topk,"ranking":"numeric token/pair agreement plus address lexical overlap"}
    with gzip.open(cache,"wt",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False)
    return result
