"""Deterministic small fixture for end-to-end EDA plumbing; not challenge data."""
from pathlib import Path
import csv
import subprocess
import sys

root=Path(__file__).resolve().parent/'cache'/'smoke_dataset'
for split,countries in [('train',['India','US']),('test',['India','US','France'])]:
    (root/split).mkdir(parents=True,exist_ok=True)
    for source in [1,2,3]:
        with (root/split/f'{split}_source{source}.tsv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,delimiter='\t')
            writer.writerow(['entity_id','business_name','business_address','country'])
            for j,country in enumerate(countries):
                for i in range(40):
                    k=j*100+i+(1000 if split=='test' else 0)
                    name=f'Business {i} Limited' if source==1 else f'Business {i} Ltd'
                    address=f'{i+1} Main Road, Town {j}, 12345' if i%7 else ''
                    writer.writerow([f'S{source}-{k}',name,address,country])
    if split=='train':
        with (root/split/'train_ground_truth.tsv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f,delimiter='\t');writer.writerow(['source1_entity_id','matched_entity_ids'])
            for j in range(2):
                for i in range(40):
                    k=j*100+i
                    writer.writerow([f'S1-{k}',f'S2-{k},S3-{k}' if i%4 else ''])
out=root.parent/'smoke_output'
cmd=[sys.executable,str(Path(__file__).resolve().parent/'run_eda.py'),'--data-root',str(root),
     '--output',str(out),'--anchors','20','--corpus-per-stratum','30']
if len(sys.argv)>1:cmd+=['--extra-site-packages',sys.argv[1]]
subprocess.run(cmd,check=True)
import json
m=json.loads((out/'eda_metrics.json').read_text(encoding='utf-8'))
assert m['ground_truth']['rows']==80
assert m['ground_truth']['buckets']['0']['count']==20
assert m['ground_truth']['matches_total']==120
assert all(m['files'][f'train_source{s}']['rows']==80 for s in [1,2,3])
assert all(m['files'][f'test_source{s}']['rows']==120 for s in [1,2,3])
assert m['ground_truth']['targets_with_multiple_s1']==0
assert len(list((out/'figures').glob('*.png')))==6
for b in m['blocking']['results']:
    assert 0<=b['hits']<=b['true_pairs']
    assert b['median_candidates']<=b['max_candidates']
import numpy as np
anchors=json.loads((out/'cache'/'anchors.json').read_text(encoding='utf-8'))
records=[]
for source in [2,3]:
    with (root/'train'/f'train_source{source}.tsv').open(encoding='utf-8') as f:
        records.extend(csv.DictReader(f,delimiter='\t'))
from src.normalization import norm,suffix,numbers
from collections import Counter
frequency=Counter((b['country'],t) for b in records for t in set(norm(b['business_name']).split()))
threshold=m['blocking']['rare_token_thresholds']
checks={}
for a in anchors:
    sets={s:set() for s in ['exact_normalized_name','suffix_stripped_name','rare_name_token','numeric_address_token']}
    for b in records:
        if a['country']!=b['country']:continue
        if norm(a['name']) and norm(a['name'])==norm(b['business_name']):sets['exact_normalized_name'].add(b['entity_id'])
        if suffix(a['name']) and suffix(a['name'])==suffix(b['business_name']):sets['suffix_stripped_name'].add(b['entity_id'])
        if any(frequency[(a['country'],t)]<=threshold[a['country']] for t in set(norm(a['name']).split())&set(norm(b['business_name']).split())):sets['rare_name_token'].add(b['entity_id'])
        if set(numbers(a['address']))&set(numbers(b['business_address'])):sets['numeric_address_token'].add(b['entity_id'])
    sets['simple_union']=set().union(*sets.values())
    for strategy,s in sets.items():checks.setdefault(strategy,[]).append((len(s),len(s&set(a['matches']))))
for b in m['blocking']['results']:
    if b['strategy'] not in checks:continue
    values=checks[b['strategy']]
    assert abs(b['mean_candidates']-np.mean([x[0] for x in values]))<1e-9
    assert b['hits']==sum(x[1] for x in values)
print('Passed end-to-end fixture counts, labels, blocking bounds, JSON, report, and six charts.')
