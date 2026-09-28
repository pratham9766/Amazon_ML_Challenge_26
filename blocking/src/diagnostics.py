"""Fixed benchmark construction and missed-positive categorization."""
import gzip
import hashlib
import heapq
import json
import logging
from collections import Counter,defaultdict
from .normalize import basic,tokens

def stable_int(seed,text):
    return int.from_bytes(hashlib.blake2b(f"{seed}|{text}".encode(),digest_size=8).digest(),"big")

def _bucket(n):
    return "0" if n==0 else "1" if n==1 else "2" if n==2 else "3" if n==3 else "4+"

def create_benchmark(db,out,config):
    path=out/"artifacts"/"benchmark_queries.json.gz"
    if path.exists():
        with gzip.open(path,"rt",encoding="utf-8") as f:return json.load(f)
    seed=config["seed"];pool_n=config["candidate_pool_queries"]
    heap=[]
    sql="""SELECT t.id,t.n,t.s2,t.s3,r.name,r.address,r.country
             FROM truth t JOIN records r ON r.file='train_source1' AND r.id=t.id"""
    for row in db.execute(sql):
        priority=stable_int(seed,row[0])
        item=(-priority,row)
        if len(heap)<pool_n:heapq.heappush(heap,item)
        elif priority < -heap[0][0]:heapq.heapreplace(heap,item)
    pool=[row for _,row in sorted(heap,reverse=True)]
    enriched=[]
    for pos,(qid,n,s2,s3,name,address,country) in enumerate(pool):
        targets=[]
        for target,source,tname,taddress,tcountry in db.execute(
            """SELECT e.target,e.source,r.name,r.address,r.country FROM edges e
               JOIN records r ON r.file=('train_source'||e.source) AND r.id=e.target
               WHERE e.s1=?""",(qid,)):
            targets.append({"id":target,"source":source,"name":tname,"address":taddress,"country":tcountry})
        ratios=[]
        qtokens=set(tokens(name))
        for t in targets:
            tt=set(tokens(t["name"]));ratios.append(len(qtokens&tt)/len(qtokens|tt) if qtokens|tt else 0)
        source_profile="singleton" if n==0 else "S2_only" if s2 and not s3 else "S3_only" if s3 and not s2 else "both"
        item={"id":qid,"name":name,"address":address,"country":country,"matches":[t["id"] for t in targets],
              "targets":targets,"multiplicity":n,"multiplicity_group":_bucket(n),"source_profile":source_profile,
              "short_name":len(tokens(name))<=2,
              "missing_target_address":any(not t["address"].strip() for t in targets),
              "nonascii_target_name":any(not t["name"].isascii() for t in targets),
              "weak_lexical":bool(ratios) and min(ratios)<0.25,
              "sample_priority":stable_int(seed,qid),"split_priority":stable_int(seed+1,qid)}
        enriched.append(item)
    groups=defaultdict(list)
    for x in enriched:
        key=(x["country"],x["multiplicity_group"],x["source_profile"])
        groups[key].append(x)
    total=config["development_queries"]+config["confirmation_queries"]
    allocations={k:max(1,round(total*len(v)/len(enriched))) for k,v in groups.items()}
    while sum(allocations.values())!=total:
        delta=total-sum(allocations.values())
        key=max(groups,key=lambda k:(len(groups[k])-allocations[k],str(k))) if delta>0 else max(allocations,key=lambda k:(allocations[k],str(k)))
        allocations[key]+=1 if delta>0 else -1
    chosen=[]
    for key,rows in groups.items():chosen.extend(sorted(rows,key=lambda x:x["split_priority"])[:allocations[key]])
    chosen.sort(key=lambda x:x["split_priority"])
    for x in chosen:
        x["repeated_name_frequency"]=db.execute("SELECT COUNT(*) FROM records INDEXED BY ix_name WHERE file='train_source1' AND country=? AND nn=?",(x["country"],basic(x["name"]))).fetchone()[0]
        x["high_repeated_name"]=x["repeated_name_frequency"]>=20
    dev_n=config["development_queries"]
    benchmark={"seed":seed,"construction":"lowest stable hashes form a 65k pool; proportional stratification by country, multiplicity and source profile; deterministic disjoint split; diagnostic flags overlap", 
               "development":chosen[:dev_n],"confirmation":chosen[dev_n:]}
    with gzip.open(path,"wt",encoding="utf-8") as f:json.dump(benchmark,f,ensure_ascii=False)
    logging.info("Benchmark written: %d development, %d confirmation",len(benchmark["development"]),len(benchmark["confirmation"]))
    return benchmark

def categorize_miss(query,target):
    categories=[]
    qn,tn=basic(query["name"]),basic(target["name"]);qa,ta=basic(query["address"]),basic(target["address"])
    if not target["address"].strip():categories.append("missing target address")
    if len(tokens(query["name"]))<=2:categories.append("short/generic query name")
    if query["name"].isascii()!=target["name"].isascii():categories.append("different script/non-ASCII transition")
    if not (set(tokens(qn))&set(tokens(tn))):categories.append("no shared normalized name token")
    if qa and ta and not (set(tokens(qa))&set(tokens(ta))):categories.append("no shared normalized address token")
    if qn and tn and qn!=tn and sorted(tokens(qn))==sorted(tokens(tn)):categories.append("name word order")
    if query["multiplicity"]>=4:categories.append("high multiplicity")
    if not categories:categories.append("partial lexical corruption/other")
    return categories
