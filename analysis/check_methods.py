"""Small independent checks for diagnostic math; no model or dataset mutation."""
import random
import sys
from src.similarity import ratio,lev,token_set,features
from src.normalization import norm,suffix
from src.loaders import hist_stats
import numpy as np

def edit(a,b,substitution):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(0 if x==y else substitution)))
        previous=current
    return previous[-1]

rng=random.Random(17)
for _ in range(500):
    a=''.join(rng.choices('abcde ',k=rng.randrange(20)))
    b=''.join(rng.choices('abcde ',k=rng.randrange(20)))
    expected_ratio=1-edit(a,b,2)/(len(a)+len(b)) if a or b else 1
    expected_lev=1-edit(a,b,1)/max(len(a),len(b)) if a or b else 1
    assert abs(ratio(a,b)-expected_ratio)<1e-12,(a,b,'ratio')
    assert abs(lev(a,b)-expected_lev)<1e-12,(a,b,'lev')
assert token_set('alpha beta','alpha beta gamma')==1
assert norm('Ａ & B, Ltd.')=='a b ltd'
assert suffix('A Private Limited')=='a'
assert hist_stats({1:1,3:1})['median']==2
blank=features({'name':'!!!','address':''},{'name':'???','address':''})
assert blank['name_normalized_equal']==0 and np.isnan(blank['name_ratio'])
assert blank['address_raw_equal']==0 and np.isnan(blank['address_numeric_overlap'])
print('Passed: 1,000 independently checked string distances, token-set subset, normalization, histogram median.')
