"""Sampled pair distributions and full-universe candidate feasibility probes."""
from collections import Counter, defaultdict
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from .normalization import norm,suffix,numbers,postal,forms
from .similarity import features,distribution,BACKEND
from .ground_truth import record

_WORKER_MODELS=None

def _init_transform_workers(model_path,extra_paths):
    global _WORKER_MODELS
    import sys
    for path in extra_paths:
        if path not in sys.path:sys.path.append(path)
    with open(model_path,'rb') as f:_WORKER_MODELS=pickle.load(f)

def _transform(field,texts):
    return _WORKER_MODELS[field].transform(texts)

def sampled_records(out):
    rows=[]
    for source in [2,3]:
        sample=json.loads((out/'cache'/f'train_source{source}_sample.json').read_text(encoding='utf-8'))
        for country, values in sample.items():
            rows.extend({'id':r['entity_id'],'name':r['business_name'],'address':r['business_address'],'country':country} for r in values)
    return rows

def tfidf_models(out,anchors,corpus):
    path=out/'cache'/'tfidf.pkl'
    if path.exists():
        with path.open('rb') as f: return pickle.load(f)
    result={}
    # Uniform training target sample only. No test text and no label-dependent corpus enrichment.
    for field in ['name','address']:
        model=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=120000,
                              lowercase=False,dtype=np.float32,norm='l2',smooth_idf=True)
        model.fit([norm(r[field]) for r in corpus])
        result[field]=model
    with path.open('wb') as f: pickle.dump(result,f)
    return result

def summarize_pairs(df):
    return {col:distribution(df[col].values) for col in df.columns if col not in ['kind','country','source','anchor_id','target_id']}

def analyze_pairs(db,anchors,out,seed,corpus_per_stratum):
    cache=out/'cache'/'pair_results.json'
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    rng=np.random.default_rng(seed)
    corpus=sampled_records(out)
    models=tfidf_models(out,anchors,corpus)
    by_country=defaultdict(list)
    for i,r in enumerate(corpus): by_country[r['country']].append(i)
    matrices={f:m.transform([norm(r[f]) for r in corpus]) for f,m in models.items()}
    anchor_matrices={f:m.transform([norm(r[f]) for r in anchors]) for f,m in models.items()}
    token_index,number_index=defaultdict(list),defaultdict(list)
    for j,r in enumerate(corpus):
        for token in set(norm(r['name']).split()): token_index[(r['country'],token)].append(j)
        for token in set(numbers(r['address'])): number_index[(r['country'],token)].append(j)
    pairs=[]
    for i,a in enumerate(anchors):
        positives=set(a['matches'])
        for target in sorted(positives):
            pairs.append(('positive',a,record(db,'train_source'+target[1],target)))
        choices=by_country[a['country']]
        # Four random negatives per reference, rejection of every labeled positive.
        selected=set()
        for j in rng.permutation(choices):
            b=corpus[int(j)]
            if b['id'] not in positives:
                selected.add(int(j)); pairs.append(('random_same_country',a,b))
                if len(selected)==4: break
        for field,kind in [('name','hard_name'),('address','hard_address')]:
            scores=(anchor_matrices[field][i] @ matrices[field][choices].T).toarray().ravel()
            # Stable ID-order-independent secondary order is candidate sample position.
            order=np.lexsort((np.array(choices),-scores))
            found=0
            for k in order:
                b=corpus[choices[int(k)]]
                if b['id'] not in positives and scores[k]>0:
                    pairs.append((kind,a,b)); found+=1
                    if found==3: break
        shared=set()
        for token in set(norm(a['name']).split()):
            idx=token_index[(a['country'],token)]
            if 1<=len(idx)<=max(20,len(choices)*.001): shared.update(idx)
        for token in set(numbers(a['address'])): shared.update(number_index[(a['country'],token)])
        found=0
        for j in rng.permutation(sorted(shared)):
            b=corpus[int(j)]
            if b['id'] not in positives:
                pairs.append(('shared_token_or_number',a,b));found+=1
                if found==4:break
        if i%200==0:logging.info('Pair diagnostic anchors: %d/%d',i,len(anchors))
    logging.info('Computing %d pair feature rows',len(pairs))
    frame=[]
    for kind,a,b in pairs:
        frame.append({'kind':kind,'country':a['country'],'source':b['id'][:2],
                      'anchor_id':a['id'],'target_id':b['id'],**features(a,b)})
    df=pd.DataFrame(frame)
    for field,model in models.items():
        left=model.transform([norm(a[field]) for _,a,b in pairs])
        right=model.transform([norm(b[field]) for _,a,b in pairs])
        cosine=np.asarray(left.multiply(right).sum(axis=1)).ravel()
        valid=np.array([bool(norm(a[field]) and norm(b[field])) for _,a,b in pairs])
        cosine[~valid]=np.nan
        df[field+'_tfidf_cosine']=cosine
    result={'method':{'anchors':len(anchors),'positive_sampling':'all labeled pairs of proportional country-stratified uniform S1 sample; entity-clustered',
                     'negative_corpus_rows':len(corpus),'negative_corpus_per_country_source':corpus_per_stratum,
                     'hard_negative_method':'top 3 nonmatches in uniform sampled same-country S2+S3 corpus, separately by name/address char TF-IDF',
                     'negative_truth_assumption':'Ground truth is exhaustive for each S1; all listed matches excluded.',
                     'similarity_backend':BACKEND,
                     'tfidf':'char_wb 3-5 grams; min_df=2; max_features=120000; float32; smoothed IDF; L2; fit uniform training target sample only',
                     'feature_scale':'All similarities in [0,1]; numeric/postal features NaN when either side lacks tokens; equality excludes blank strings'},
            'by_kind':{},'by_country_source':{},'separation':{},'normalization':{},'risks':{},'examples':{}}
    for kind,g in df.groupby('kind'): result['by_kind'][kind]=summarize_pairs(g)
    for (kind,country,source),g in df.groupby(['kind','country','source']):
        result['by_country_source'][f'{kind}/{country}/{source}']=summarize_pairs(g)
    pos=df[df.kind=='positive']
    for kind in ['random_same_country','hard_name','hard_address','shared_token_or_number']:
        neg=df[df.kind==kind]
        ranks=[]
        for field in pos.columns:
            if field in ['kind','country','source','anchor_id','target_id']:continue
            p=pos[field].dropna().values;n=neg[field].dropna().values
            auc=float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])) if len(p) and len(n) else None
            ranks.append({'feature':field,'auc':auc,'positive_valid':len(p),'negative_valid':len(n),
                          'positive_ge_09':float(np.mean(p>=.9)) if len(p) else None,
                          'negative_ge_09':float(np.mean(n>=.9)) if len(n) else None})
        result['separation'][kind]=sorted(ranks,key=lambda x:x['auc'] if x['auc'] is not None else -1,reverse=True)
    for field in ['name','address']:
        totals=defaultdict(lambda:Counter())
        gains=defaultdict(lambda:Counter())
        for kind,a,b in pairs:
            fa,fb=forms(a[field]),forms(b[field])
            for rule in fa:
                equal=bool(fa[rule]) and fa[rule]==fb[rule]
                totals[(kind,rule)]['n']+=1
                totals[(kind,rule)]['equal']+=equal
                totals[(kind,rule)]['new_equal_vs_raw']+=equal and a[field]!=b[field]
        result['normalization'][field]={f'{kind}/{rule}':dict(c,rate=c['equal']/c['n']) for (kind,rule),c in totals.items()}
    # Risks are descriptive strata, not business/chain classifications inferred from names.
    flags={
        'identical_normalized_name':lambda a,b:bool(norm(a['name'])) and norm(a['name'])==norm(b['name']),
        'same_normalized_address':lambda a,b:bool(norm(a['address'])) and norm(a['address'])==norm(b['address']),
        'short_name_either_le_2_tokens':lambda a,b:min(len(norm(a['name']).split()),len(norm(b['name']).split()))<=2,
        'missing_address_either':lambda a,b:not a['address'].strip() or not b['address'].strip(),
        'suffix_equal_but_normalized_different':lambda a,b:bool(suffix(a['name'])) and suffix(a['name'])==suffix(b['name']) and norm(a['name'])!=norm(b['name']),
        'shared_number_but_different_address':lambda a,b:bool(set(numbers(a['address']))&set(numbers(b['address']))) and norm(a['address'])!=norm(b['address']),
        'ascii_name_reference_nonascii_target':lambda a,b:a['name'].isascii() and not b['name'].isascii(),
        'ascii_address_reference_nonascii_target':lambda a,b:a['address'].isascii() and not b['address'].isascii(),
    }
    for label,predicate in flags.items():
        idx=[i for i,(_,a,b) in enumerate(pairs) if predicate(a,b)]
        result['risks'][label]={k:len(g) for k,g in df.iloc[idx].groupby('kind')}
        negidx=[i for i in idx if pairs[i][0]!='positive']
        result['examples'][label]=[{'kind':pairs[i][0],'left':pairs[i][1], 'right':pairs[i][2],
                                  'name_ratio':float(df.iloc[i].name_ratio) if pd.notna(df.iloc[i].name_ratio) else None,
                                  'address_ratio':float(df.iloc[i].address_ratio) if pd.notna(df.iloc[i].address_ratio) else None} for i in negidx[:3]]
    # Analyze digit IDs only as a leakage audit; they are excluded from every retrieval signal.
    result['id_leakage_probe']={}
    for kind in df.kind.unique():
        selected=[(a,b) for k,a,b in pairs if k==kind]
        diffs=[abs(int(a['id'][3:])-int(b['id'][3:])) for a,b in selected if a['id'][3:].isdigit() and b['id'][3:].isdigit()]
        result['id_leakage_probe'][kind]={'same_numeric_suffix':sum(a['id'][3:]==b['id'][3:] for a,b in selected),
                                        'absolute_suffix_difference':distribution(diffs),
                                        'numeric_id_pearson_correlation':float(np.corrcoef([int(a['id'][3:]) for a,b in selected],[int(b['id'][3:]) for a,b in selected])[0,1]) if len(selected)>1 else None}
    df.to_csv(out/'cache'/'pair_features.csv.gz',index=False,compression='gzip')
    cache.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result

def summarize_candidates(counts,hits,anchors,label,scope):
    truth_counts=np.array([len(a['matches']) for a in anchors])
    hit_counts=np.array([len(x) for x in hits])
    matched=truth_counts>0
    # Cluster bootstrap samples S1, retaining dependence between one-to-many pairs.
    rng=np.random.default_rng(20260925)
    boot=[]
    for _ in range(500):
        ix=rng.integers(0,len(anchors),len(anchors))
        denom=truth_counts[ix].sum()
        if denom:boot.append(hit_counts[ix].sum()/denom)
    return {'strategy':label,'scope':scope,'anchors':len(anchors),'true_pairs':int(truth_counts.sum()),
            'hits':int(hit_counts.sum()),'recall':float(hit_counts.sum()/truth_counts.sum()),
            'recall_cluster_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
            'mean_candidates':float(np.mean(counts)),'median_candidates':float(np.median(counts)),
            'p95_candidates':float(np.percentile(counts,95)),'max_candidates':int(max(counts)),
            'all_matches_recovered_non_singletons':float(np.mean(hit_counts[matched]==truth_counts[matched])),
            'matched_entity_macro_recall':float(np.mean(hit_counts[matched]/truth_counts[matched])),
            'singleton_mean_candidates':float(np.mean(np.array(counts)[~matched])) if (~matched).any() else None,
            'by_country':{c:{'anchors':sum(a['country']==c for a in anchors),
                              'pairs':sum(len(a['matches']) for a in anchors if a['country']==c),
                              'hits':sum(len(hits[i]) for i,a in enumerate(anchors) if a['country']==c)} for c in sorted({a['country'] for a in anchors})}}

def blocking(db,anchors,out,seed,pair_results):
    cache=out/'cache'/'blocking.json'
    if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))
    rng=np.random.default_rng(seed)
    corpus=sampled_records(out)
    models=tfidf_models(out,anchors,corpus)
    # Full-universe top-K costs scale with query count. Use 120 randomly selected S1s,
    # preserving country proportions; all 1200 anchors are used for simple strategies.
    tf_idx=[]
    for country in sorted({a['country'] for a in anchors}):
        eligible=[i for i,a in enumerate(anchors) if a['country']==country]
        tf_idx.extend(sorted(rng.choice(eligible,min(len(eligible),max(1,round(120*len(eligible)/len(anchors)))),replace=False).tolist()))
    tf_anchors=[anchors[i] for i in tf_idx]
    country_queries={c:[j for j,a in enumerate(tf_anchors) if a['country']==c] for c in sorted({a['country'] for a in anchors})}
    qmat={f:m.transform([norm(a[f]) for a in tf_anchors]) for f,m in models.items()}
    top={f:[[] for _ in tf_anchors] for f in models}
    lookup={s:defaultdict(set) for s in ['exact_normalized_name','suffix_stripped_name','rare_name_token','numeric_address_token']}
    relevant_tokens=defaultdict(set)
    for i,a in enumerate(anchors):
        c=a['country']
        for label,key in [('exact_normalized_name',norm(a['name'])),('suffix_stripped_name',suffix(a['name']))]:
            if key:lookup[label][(c,key)].add(i)
        relevant_tokens[c].update(norm(a['name']).split())
        for token in set(numbers(a['address'])):lookup['numeric_address_token'][(c,token)].add(i)
    frequencies=Counter();country_n=Counter()
    logging.info('Full-corpus query-token document frequencies')
    cursor=db.execute("SELECT country,nn FROM records WHERE file IN ('train_source2','train_source3')")
    for country,name in cursor:
        country_n[country]+=1
        for token in set(name.split())&relevant_tokens[country]:frequencies[(country,token)]+=1
    thresholds={c:max(1,int(n*.001)) for c,n in country_n.items()}
    for i,a in enumerate(anchors):
        for token in set(norm(a['name']).split()):
            if 0<frequencies[(a['country'],token)]<=thresholds[a['country']]:
                lookup['rare_name_token'][(a['country'],token)].add(i)
    strategies=list(lookup)+['simple_union']
    counts={s:np.zeros(len(anchors),dtype=np.int64) for s in strategies}
    hits={s:[set() for _ in anchors] for s in strategies}
    truths=[set(a['matches']) for a in anchors]
    # Keep counts, not huge candidate-ID sets. For top-K records retain a simple-rule
    # membership bit, which is sufficient to compute exact union sizes and recall.
    tf_local={i:j for j,i in enumerate(tf_idx)}
    cursor=db.execute("SELECT id,name,address,country,nn,sn,an FROM records WHERE file IN ('train_source2','train_source3')")
    import sys
    pool=ProcessPoolExecutor(max_workers=2,initializer=_init_transform_workers,
                             initargs=(str(out/'cache'/'tfidf.pkl'),list(sys.path)))
    processed=0
    while True:
        batch=cursor.fetchmany(10000)
        if not batch:break
        batch_simple={}
        for target,name,address,country,nn,sn,an in batch:
            matches={
                'exact_normalized_name':lookup['exact_normalized_name'].get((country,nn),set()),
                'suffix_stripped_name':lookup['suffix_stripped_name'].get((country,sn),set()),
                'rare_name_token':set().union(*(lookup['rare_name_token'].get((country,t),set()) for t in set(nn.split()))),
                'numeric_address_token':set().union(*(lookup['numeric_address_token'].get((country,t),set()) for t in set(numbers(address))))}
            matches['simple_union']=set().union(*matches.values())
            for s,ii in matches.items():
                for i in ii:
                    counts[s][i]+=1
                    if target in truths[i]:hits[s][i].add(target)
            membership=matches['simple_union'] & tf_local.keys()
            if membership:batch_simple[target]=membership
        for country,query_idx in country_queries.items():
            selected=[r for r in batch if r[3]==country]
            if not selected:continue
            ids=np.array([r[0] for r in selected])
            jobs={field:pool.submit(_transform,field,[r[col] for r in selected]) for field,col in [('name',4),('address',6)]}
            for field in ['name','address']:
                x=jobs[field].result()
                score=(qmat[field][query_idx] @ x.T).toarray()
                for row,j in enumerate(query_idx):
                    scores=score[row]
                    k=min(50,len(scores))
                    # Include all candidates tied at this batch's kth score before ID tie-break.
                    threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
                    ix=np.flatnonzero((scores>=threshold)&(scores>0))
                    chosen=sorted(((float(scores[z]),str(ids[z]),tf_idx[j] in batch_simple.get(str(ids[z]),())) for z in ix),key=lambda p:(-p[0],p[1]))[:50]
                    top[field][j]=sorted(top[field][j]+chosen,key=lambda p:(-p[0],p[1]))[:50]
        processed+=len(batch)
        if processed%100000==0:logging.info('Full-universe blocking and TF-IDF probes: %s targets',format(processed,','))
    pool.shutdown()
    results=[summarize_candidates(counts[s],hits[s],anchors,s,'full combined S2+S3 target corpus; same-country') for s in strategies]
    for k in [5,10,20,50]:
        for strategy in ['tfidf_name','tfidf_address','tfidf_union','simple_plus_tfidf_union']:
            candidate_counts=[];h=[]
            for j in range(len(tf_anchors)):
                name_set={t for _,t,_ in top['name'][j][:k]}
                address_set={t for _,t,_ in top['address'][j][:k]}
                simple_members={t for _,t,is_simple in top['name'][j][:k]+top['address'][j][:k] if is_simple}
                cs={'tfidf_name':name_set,'tfidf_address':address_set,'tfidf_union':name_set|address_set,
                    'simple_plus_tfidf_union':name_set|address_set}[strategy]
                count=len(cs)
                hit=cs&set(tf_anchors[j]['matches'])
                if strategy=='simple_plus_tfidf_union':
                    count=int(counts['simple_union'][tf_idx[j]])+len(cs-simple_members)
                    hit=hit|hits['simple_union'][tf_idx[j]]
                candidate_counts.append(count);h.append(hit)
            results.append(summarize_candidates(candidate_counts,h,tf_anchors,f'{strategy}@{k}',
                                              f'full combined S2+S3 corpus; same-country; {len(tf_anchors)} sampled queries; sample-fitted lexical TF-IDF'))
    # Same query set comparison is essential: do not compare 120 vs 1200 as an ablation.
    results.append(summarize_candidates([counts['simple_union'][i] for i in tf_idx],
                                      [hits['simple_union'][i] for i in tf_idx],
                                      tf_anchors,'simple_union_on_tfidf_queries','full combined S2+S3 corpus; same-country'))
    result={'rare_token_definition':'Name token document frequency <=0.1% of full same-country combined S2+S3 corpus; no cap; all shared qualifying tokens',
            'rare_token_thresholds':thresholds,'target_rows_by_country':dict(country_n),'retrieval_query_ids':[a['id'] for a in tf_anchors],
            'tfidf_vocabulary_sizes':{f:len(m.vocabulary_) for f,m in models.items()},
            'top_k_definition':'K over combined S2+S3; strictly positive cosine; deterministic lexicographic ID tie-break. Vocabulary and IDF fitted to uniform target sample, retrieval searches every target.',
            'results':results}
    cache.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (out/'cache'/'retrieval_top50.json').write_text(json.dumps(top),encoding='utf-8')
    return result
