"""Streaming query-conditioned sparse lexical retrieval.

This module implements bounded IDF-weighted word and character-fragment channels.
It never constructs an S1-by-target matrix. Classical sklearn char-TFIDF from EDA is
retained as a separate baseline because an exact 27.5k x 10.3m sparse top-N backend is
not installed in the offline environment.
"""
import gzip
import heapq
import json
import logging
import math
import time
import sqlite3
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
from pathlib import Path
from .normalize import char_fragments,word_features,suffix_stripped

FEATURE_FN={
    "name_char":lambda n,a:char_fragments(n),
    "address_char":lambda n,a:char_fragments(a),
    "suffix_char":lambda n,a:char_fragments(suffix_stripped(n)),
    "name_word":lambda n,a:word_features(n),
    "address_word":lambda n,a:word_features(a),
}

def _partition(tsv_path,file,channel,postings_by_country,denom,topk):
    heaps={};processed=0
    with open(tsv_path,encoding='utf-8',newline='') as f:
        reader=csv.reader(f,delimiter='\t');next(reader)
        for target,name,address,country in reader:
            acc=defaultdict(float)
            for feature in FEATURE_FN[channel](name,address):
                for qi,weight in postings_by_country.get(country,{}).get(feature,()):acc[qi]+=weight
            for qi,score in acc.items():
                item=(score/denom[qi],target);heap=heaps.setdefault(qi,[])
                if len(heap)<topk:heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
            processed+=1
    return heaps,processed,file

def _sample_df(out,channel):
    cache=out/"artifacts"/f"{channel}_sample_df.json.gz"
    if cache.exists():
        with gzip.open(cache,"rt",encoding="utf-8") as f:return json.load(f)
    counts=Counter();documents=0
    for source in (2,3):
        path=out.parent/"analysis"/"cache"/f"train_source{source}_sample.json"
        sample=json.loads(path.read_text(encoding="utf-8"))
        for rows in sample.values():
            for r in rows:
                documents+=1
                counts.update(FEATURE_FN[channel](r["business_name"],r["business_address"]))
    result={"documents":documents,"df":dict(counts)}
    with gzip.open(cache,"wt",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False)
    return result

def retrieve_channel(db,queries,out,config,channel):
    cache=out/"artifacts"/f"candidates_{channel}.json.gz"
    if cache.exists():
        with gzip.open(cache,"rt",encoding="utf-8") as f:return json.load(f)
    started=time.monotonic();df_info=_sample_df(out,channel);df=df_info["df"];docs=df_info["documents"]
    limit=max(config["sample_df_limits"]);per_query=config["selected_features_per_query"]
    postings_by_country=defaultdict(lambda:defaultdict(list));denom=[];selected_meta=[]
    for i,q in enumerate(queries):
        feats=FEATURE_FN[channel](q["name"],q["address"])
        ranked=[]
        for feature in feats:
            freq=df.get(feature,0)
            if 0<freq<=limit:
                weight=math.log((docs+1)/(freq+1))+1
                ranked.append((weight,feature,freq))
        ranked=sorted(ranked,key=lambda x:(-x[0],x[1]))[:per_query]
        total=sum(x[0] for x in ranked) or 1.0;denom.append(total);selected_meta.append(ranked)
        for weight,feature,freq in ranked:postings_by_country[q["country"]][feature].append((i,weight))
    # Prevent a feature selected by many benchmark queries from dominating runtime.
    maxq=config["max_queries_per_feature"]
    postings_by_country={c:{f:v for f,v in posting.items() if len(v)<=maxq} for c,posting in postings_by_country.items()}
    heaps=[[] for _ in queries];topk=config["channel_top_k"];processed=0
    project=Path(__file__).resolve().parents[2]
    data_roots=list(project.glob('dataset*/dataset/train'))
    if len(data_roots)!=1:raise ValueError('Expected one official train directory')
    partitions=['train_source2','train_source3']
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(_partition,str(data_roots[0]/f'{file}.tsv'),file,channel,postings_by_country,denom,topk) for file in partitions]
        for job in as_completed(jobs):
            local,n,file=job.result();processed+=n
            for qi,items in local.items():
                combined=heaps[qi]+items
                heaps[qi]=heapq.nlargest(topk,combined)
                heapq.heapify(heaps[qi])
            logging.info("%s partition complete: %s (%s rows)",channel,file,format(n,","))
    result={q["id"]:[{"id":target,"score":score,"rank":rank+1,"channel":channel} for rank,(score,target) in enumerate(sorted(heap,reverse=True))]
            for q,heap in zip(queries,heaps)}
    result["__metadata__"]={"channel":channel,"runtime_seconds":time.monotonic()-started,"targets_scanned":processed,
        "sample_documents":docs,"sample_df_max":limit,"selected_features_per_query":per_query,
        "max_queries_per_feature":maxq,"retrieval":"IDF-weighted bounded sparse word/character-fragment retrieval"}
    with gzip.open(cache,"wt",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False)
    return result
