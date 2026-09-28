"""Cross-check finished evidence and sampled pair labels against the disk audit."""
import json
from pathlib import Path
import sqlite3
import pandas as pd

out=Path(__file__).resolve().parent
m=json.loads((out/'eda_metrics.json').read_text(encoding='utf-8'))
gt=m['ground_truth']
assert sum(v['count'] for v in gt['buckets'].values())==gt['rows']
assert gt['s2_only']+gt['s3_only']+gt['both']+gt['buckets']['0']['count']==gt['rows']
hist={int(k):v for k,v in gt['match_count']['histogram'].items()}
assert sum(k*v for k,v in hist.items())==gt['matches_total']
for file,d in m['files'].items():
    assert d['rows']==sum(g['rows'] for g in d['countries'].values())
    assert d['rows']==d['structure']['csv_record_count']
    assert d['unique_ids']+d['duplicate_id_excess']==d['rows']
    for country,g in d['countries'].items():
        for field in ['entity_id','business_name','business_address','country']:
            assert 0<=g['literal_empty'][field]<=g['null'][field]<=g['rows']
        for field,col in [('name_length','business_name'),('address_length','business_address')]:
            assert sum(g[field]['histogram'].values())+g['null'][col]==g['rows']
for kind,features in m['pairs']['by_kind'].items():
    for feature,d in features.items():
        assert d['valid_n']+d['missing_n']==d['n']
        values=[d[f'p{p}'] for p in [1,5,10,25,50,75,90,95,99]]
        if d['valid_n']:
            assert values==sorted(values)
            assert 0<=values[0]<=values[-1]<=1.000001
blocks={r['strategy']:r for r in m['blocking']['results']}
for strategy in ['tfidf_name','tfidf_address','tfidf_union','simple_plus_tfidf_union']:
    for previous,current in zip([5,10,20],[10,20,50]):
        a,b=blocks[f'{strategy}@{previous}'],blocks[f'{strategy}@{current}']
        assert a['hits']<=b['hits'] and a['mean_candidates']<=b['mean_candidates']
for k in [5,10,20,50]:
    assert blocks[f'tfidf_union@{k}']['hits']>=max(blocks[f'tfidf_name@{k}']['hits'],blocks[f'tfidf_address@{k}']['hits'])
    assert blocks[f'simple_plus_tfidf_union@{k}']['hits']>=blocks['simple_union_on_tfidf_queries']['hits']
db=sqlite3.connect(f'file:{out / "cache" / "audit.sqlite"}?mode=ro',uri=True)
df=pd.read_csv(out/'cache'/'pair_features.csv.gz')
for a,g in df.groupby('anchor_id'):
    truth={r[0] for r in db.execute('SELECT target FROM edges WHERE s1=?',(a,))}
    assert set(g.loc[g.kind=='positive','target_id'])==truth
    assert not set(g.loc[g.kind!='positive','target_id'])&truth
for file,d in m['files'].items():
    assert db.execute('SELECT COUNT(*) FROM records WHERE file=?',(file,)).fetchone()[0]==d['rows']
assert len(list((out/'figures').glob('*.png')))==6
assert (out/'dataset_report.md').stat().st_size>10000
print('PASS: exact counts reconcile; all negative pairs exclude truth; feature percentiles valid; candidate unions and K curves monotonic; six figures and full report exist.')
