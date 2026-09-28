"""Unlabeled lexical coverage diagnostics; no test fitting or predictions."""
import json
import pickle
from collections import Counter
import unicodedata
from .normalization import norm
from .similarity import distribution

def lexical_coverage(metrics,out):
    path=out/'cache'/'lexical_coverage.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    with (out/'cache'/'tfidf.pkl').open('rb') as f:models=pickle.load(f)
    result={'method':'First 2,000 bottom-priority sampled records per source/country (uniform sample). Fraction of normalized char_wb 3-5 gram occurrences present in the training-only diagnostic vocabulary. Empty text excluded. Coverage is not matching accuracy.',
            'strata':{},'exact_name_character_categories':{}}
    for key,file in metrics['files'].items():
        samples=json.loads((out/'cache'/f'{key}_sample.json').read_text(encoding='utf-8'))
        for country,records in samples.items():
            d={}
            for field,column in [('name','business_name'),('address','business_address')]:
                model=models[field];analyze=model.build_analyzer();values=[];empty=0
                for row in records[:2000]:
                    text=norm(row[column])
                    if not text:empty+=1;continue
                    grams=analyze(text)
                    if not grams:empty+=1;continue
                    values.append(sum(g in model.vocabulary_ for g in grams)/len(grams))
                d[field]={'coverage':distribution(values),'empty_or_no_grams':empty,'zero_known_grams':sum(x==0 for x in values)}
            result['strata'][f'{key}/{country}']=d
            categories=Counter()
            for char,count in file['countries'][country]['characters'].items():
                category=unicodedata.category(char)
                if category.startswith('L'):
                    name=unicodedata.name(char,'UNKNOWN')
                    label=next((s for s in ['LATIN','DEVANAGARI','BENGALI','TAMIL','TELUGU','GUJARATI','KANNADA','MALAYALAM','CYRILLIC','GREEK','ARABIC'] if s in name),'OTHER_LETTER')
                else:label=category
                categories[label]+=count
            result['exact_name_character_categories'][f'{key}/{country}']=dict(categories)
    path.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return result
