"""Ground-truth structure and exact reference checks, never inferred labels."""
from collections import Counter
import json
import logging
import re
import numpy as np
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
from .loaders import audit_structure, hist_stats

def record(db, file, entity_id):
    row=db.execute('SELECT id,name,address,country FROM records WHERE file=? AND id=?',(file,entity_id)).fetchone()
    if row is None:
        raise ValueError(f'Missing record: {file}/{entity_id}')
    return dict(zip(['id','name','address','country'],row))

def analyze_ground_truth(db,path,out,seed,anchor_count):
    cache=out/'cache'/'ground_truth.json'
    if cache.exists():
        saved=json.loads(cache.read_text(encoding='utf-8'))
        stats=saved['statistics']
    else:
        db.execute('DROP TABLE IF EXISTS truth')
        db.execute('DROP TABLE IF EXISTS edges')
        db.execute('CREATE TABLE truth (id TEXT,n INTEGER,s2 INTEGER,s3 INTEGER)')
        db.execute('CREATE TABLE edges (s1 TEXT,target TEXT,source INTEGER)')
        stats={'rows':0,'blank_lists':0,'malformed_lists':0,'duplicate_ids_in_lists':0,
               'invalid_source1_syntax':0,'invalid_target_syntax':0,'examples':[]}
        stats['bytes']=path.stat().st_size
        stats['columns']=['source1_entity_id','matched_entity_ids']
        schema=pd.read_csv(path,sep='\t',nrows=1000)
        stats['pandas_inferred_dtypes_first_1000']={c:str(t) for c,t in schema.dtypes.items()}
        stats['loaded_dtypes']={c:'str (raw lexemes preserved)' for c in stats['columns']}
        stats['pandas_null_counts']={c:0 for c in stats['columns']}
        stats['literal_empty_counts']={c:0 for c in stats['columns']}
        stats['whitespace_only_counts']={c:0 for c in stats['columns']}
        hist=Counter()
        for df in pd.read_csv(path,sep='\t',dtype=str,keep_default_na=False,chunksize=50000):
            if list(df.columns)!=['source1_entity_id','matched_entity_ids']:
                raise ValueError('Unexpected ground-truth schema')
            for column in df:
                stats['pandas_null_counts'][column]+=int(df[column].isin(STR_NA_VALUES).sum())
                stats['literal_empty_counts'][column]+=int(df[column].eq('').sum())
                stats['whitespace_only_counts'][column]+=int((df[column].ne('')&df[column].str.strip().eq('')).sum())
            truths,edges=[],[]
            for s1,text in df.itertuples(index=False,name=None):
                stats['rows']+=1
                if not re.fullmatch(r'S1-\d+',s1):
                    stats['invalid_source1_syntax']+=1
                ids=text.split(',') if text else []
                stats['blank_lists']+=not bool(ids)
                if any(x!=x.strip() or not x for x in ids):
                    stats['malformed_lists']+=1
                    if len(stats['examples'])<10: stats['examples'].append([s1,text])
                stats['duplicate_ids_in_lists']+=len(ids)-len(set(ids))
                s2=sum(x.startswith('S2-') for x in ids)
                s3=sum(x.startswith('S3-') for x in ids)
                for x in ids:
                    ok=re.fullmatch(r'S([23])-\d+',x)
                    stats['invalid_target_syntax']+=not bool(ok)
                    edges.append((s1,x,int(ok.group(1)) if ok else 0))
                truths.append((s1,len(ids),s2,s3))
                hist[len(ids)]+=1
            db.executemany('INSERT INTO truth VALUES (?,?,?,?)',truths)
            db.executemany('INSERT INTO edges VALUES (?,?,?)',edges)
            db.commit()
            logging.info('Ground truth: %s rows',format(stats['rows'],','))
        db.execute('CREATE INDEX truth_id ON truth(id)')
        db.execute('CREATE INDEX edge_s1 ON edges(s1)')
        db.execute('CREATE INDEX edge_target ON edges(target,s1)')
        db.commit()
        stats['structure']=audit_structure(path)
        stats['unique_s1']=db.execute('SELECT COUNT(DISTINCT id) FROM truth').fetchone()[0]
        stats['duplicate_s1_rows']=stats['rows']-stats['unique_s1']
        stats['match_count']=hist_stats(hist)
        stats['matches_total']=sum(k*v for k,v in hist.items())
        stats['maximum_matches']=max(hist)
        stats['buckets']={str(k):{'count':hist[k],'percent':100*hist[k]/stats['rows']} for k in [0,1,2]}
        count=sum(v for k,v in hist.items() if k>=3)
        stats['buckets']['3+']={'count':count,'percent':100*count/stats['rows']}
        for label,cond in [('s2_only','s2>0 AND s3=0'),('s3_only','s3>0 AND s2=0'),('both','s2>0 AND s3>0')]:
            stats[label]=db.execute(f'SELECT COUNT(*) FROM truth WHERE {cond}').fetchone()[0]
        stats['invalid_s1_references']=db.execute("SELECT COUNT(*) FROM truth t WHERE NOT EXISTS (SELECT 1 FROM records r WHERE r.file='train_source1' AND r.id=t.id)").fetchone()[0]
        stats['s1_without_truth']=db.execute("SELECT COUNT(*) FROM records r WHERE r.file='train_source1' AND NOT EXISTS (SELECT 1 FROM truth t WHERE t.id=r.id)").fetchone()[0]
        for source in [2,3]:
            stats[f'invalid_s{source}_references']=db.execute('SELECT COUNT(*) FROM edges e WHERE source=? AND NOT EXISTS (SELECT 1 FROM records r WHERE r.file=? AND r.id=e.target)',(source,f'train_source{source}')).fetchone()[0]
        stats['targets_with_multiple_s1']=db.execute('SELECT COUNT(*) FROM (SELECT target FROM edges GROUP BY target HAVING COUNT(DISTINCT s1)>1)').fetchone()[0]
        stats['unique_targets']=db.execute('SELECT COUNT(DISTINCT target) FROM edges').fetchone()[0]
        stats['same_numeric_id_suffix_positive_pairs']=db.execute('SELECT COUNT(*) FROM edges WHERE SUBSTR(s1,4)=SUBSTR(target,4)').fetchone()[0]
        stats['by_country']={c:{'s1':n,'singletons':z,'singleton_percent':100*z/n,'pairs':p}
            for c,n,z,p in db.execute("SELECT r.country,COUNT(*),SUM(t.n=0),SUM(t.n) FROM truth t JOIN records r ON r.file='train_source1' AND r.id=t.id GROUP BY r.country")}
        stats['cross_country_positive_pairs']=0
        for source in [2,3]:
            stats['cross_country_positive_pairs']+=db.execute("SELECT COUNT(*) FROM edges e JOIN records a ON a.file='train_source1' AND a.id=e.s1 JOIN records b ON b.file=? AND b.id=e.target WHERE e.source=? AND a.country<>b.country",(f'train_source{source}',source)).fetchone()[0]
        cache.write_text(json.dumps({'statistics':stats},indent=2),encoding='utf-8')
    # Bottom-priority source sample is uniform within country; choose proportional anchor counts.
    source_samples=json.loads((out/'cache'/'train_source1_sample.json').read_text(encoding='utf-8'))
    country_sizes=stats['by_country']
    rng=np.random.default_rng(seed)
    anchors=[]
    for country,rows in source_samples.items():
        n=round(anchor_count*country_sizes[country]['s1']/stats['unique_s1'])
        chosen=rng.choice(len(rows),min(n,len(rows)),replace=False)
        for i in chosen:
            row=rows[int(i)]
            a=record(db,'train_source1',row['entity_id'])
            a['matches']=[x[0] for x in db.execute('SELECT target FROM edges WHERE s1=?',(a['id'],))]
            anchors.append(a)
    (out/'cache'/'anchors.json').write_text(json.dumps(anchors,ensure_ascii=False),encoding='utf-8')
    return stats,anchors

def leakage(db,out):
    cache=out/'cache'/'leakage.json'
    if cache.exists(): return json.loads(cache.read_text(encoding='utf-8'))
    result={'scope':'All nine source combinations; unique values per comparison; no leakage used for matching.', 'comparisons':{}}
    logging.info('Checking exact train/test overlaps')
    for s in [1,2,3]:
        tr,te=f'train_source{s}',f'test_source{s}'
        result[f's{s}_id_overlap']=db.execute('SELECT COUNT(*) FROM (SELECT id FROM records WHERE file=? INTERSECT SELECT id FROM records WHERE file=?)',(tr,te)).fetchone()[0]
        result[f's{s}_identical_full_rows']=0 if result[f's{s}_id_overlap']==0 else db.execute('SELECT COUNT(*) FROM (SELECT a.id,a.name,a.address,a.country FROM records a JOIN records b ON b.file=? AND b.id=a.id WHERE a.file=? AND a.name=b.name AND a.address=b.address AND a.country=b.country GROUP BY a.id,a.name,a.address,a.country)',(te,tr)).fetchone()[0]
    # DISTINCT intersections avoid double counting repeated text within a comparison.
    comparisons=[(s,t) for s in [1,2,3] for t in [1,2,3]]
    result['comparisons']={f'train_s{s}_test_s{t}':{} for s,t in comparisons}
    for label,cols in [('raw_names','country,name'),('normalized_names','country,nn'),('normalized_name_address','country,nn,an'),('raw_name_address','country,name,address')]:
        # One exact grouping over all sources instead of 9 repeated INTERSECT sorts.
        flags=','.join(f"MAX(file='{split}_source{s}') AS {split}{s}" for split in ['train','test'] for s in [1,2,3])
        sums=','.join(f'COALESCE(SUM(train{s}*test{t}),0)' for s,t in comparisons)
        values=db.execute(f"SELECT {sums} FROM (SELECT {flags} FROM records WHERE name<>'' GROUP BY {cols})").fetchone()
        for (s,t),value in zip(comparisons,values):result['comparisons'][f'train_s{s}_test_s{t}'][label]=value
        logging.info('All-source exact overlaps completed: %s',label)
    result['id_patterns']={}
    for file, in db.execute('SELECT DISTINCT file FROM records'):
        length_counts=dict(db.execute('SELECT LENGTH(SUBSTR(id,4)),COUNT(*) FROM records WHERE file=? GROUP BY LENGTH(SUBSTR(id,4))',(file,)))
        last_digits=dict(db.execute('SELECT SUBSTR(id,-1),COUNT(*) FROM records WHERE file=? GROUP BY SUBSTR(id,-1)',(file,)))
        result['id_patterns'][file]={'numeric_suffix_length_counts':length_counts,'last_digit_counts':last_digits,
            'nondigit_suffix_count':db.execute("SELECT COUNT(*) FROM records WHERE file=? AND (SUBSTR(id,4) GLOB '*[^0-9]*' OR LENGTH(id)<4)",(file,)).fetchone()[0]}
    cache.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
