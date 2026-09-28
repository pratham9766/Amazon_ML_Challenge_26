"""Evidence-backed Markdown, complete percentile tables, and compact PNG figures."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .loaders import hist_stats

def pct(x):
    return 'N/A' if x is None else f'{100*x:.2f}%'

def fmt(x):
    if x is None:return 'N/A'
    if isinstance(x,int):return f'{x:,}'
    if isinstance(x,float):return f'{x:,.3f}'
    return str(x).replace('|','\\|').replace('\n',' ')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])+'\n'

def aggregate(file):
    groups=file['countries'].values()
    out={'rows':file['rows']}
    for field in ['name_length','address_length','name_tokens','address_tokens','numeric_count','postal_count']:
        h=Counter()
        for g in groups:h.update({int(k):v for k,v in g[field]['histogram'].items()})
        out[field]=hist_stats(h)
    for kind in ['null','literal_empty','whitespace_only','characters']:
        h=Counter()
        for g in groups:h.update(g[kind])
        out[kind]=h
    for field in ['non_ascii_names','non_ascii_addresses']:
        out[field]=sum(g[field] for g in groups)
    return out

def tv(a,b):
    sa,sb=sum(a.values()),sum(b.values())
    return .5*sum(abs(a.get(k,0)/sa-b.get(k,0)/sb) for k in set(a)|set(b)) if sa and sb else None

def shifts(metrics):
    result={}
    for source in [1,2,3]:
        tr,te=metrics['files'][f'train_source{source}'],metrics['files'][f'test_source{source}']
        for country in ['ALL','India','US']:
            a,b=(aggregate(tr),aggregate(te)) if country=='ALL' else (tr['countries'].get(country),te['countries'].get(country))
            if not a or not b:continue
            d={}
            for field in ['name_length','address_length','name_tokens','address_tokens','numeric_count','postal_count']:
                d[field]={'train_mean':a[field]['mean'],'test_mean':b[field]['mean'],
                          'mean_delta':b[field]['mean']-a[field]['mean'],
                          'histogram_total_variation':tv({str(k):v for k,v in a[field]['histogram'].items()},{str(k):v for k,v in b[field]['histogram'].items()})}
            d['character_total_variation']=tv(a['characters'],b['characters'])
            for f in ['non_ascii_names','non_ascii_addresses']:
                d[f]={'train_rate':a[f]/a['rows'],'test_rate':b[f]/b['rows']}
            result[f'S{source}/{country}']=d
    metrics['train_test_shift']=result
    return result

def fonts(size):
    for p in [Path('C:/Windows/Fonts/arial.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
        if p.exists():return ImageFont.truetype(str(p),size)
    return ImageFont.load_default(size=size)

def base(title,subtitle=''):
    im=Image.new('RGB',(1500,900),'white');d=ImageDraw.Draw(im)
    d.text((65,28),title,fill='#17212b',font=fonts(32))
    d.text((65,80),subtitle,fill='#445363',font=fonts(20))
    return im,d

COLORS=['#2463a1','#ce6b24','#39815c','#94569a','#666666']

def bars(path,title,labels,series,ylabel,subtitle=''):
    im,d=base(title,subtitle);left,top,right,bottom=115,150,1420,690
    maximum=max([v for _,vals in series for v in vals]+[.01])*1.15
    for k in range(6):
        v=maximum*k/5;y=bottom-(bottom-top)*k/5
        d.line((left,y,right,y),fill='#dce2e8',width=1)
        d.text((12,y-12),f'{v:,.1f}',fill='#445363',font=fonts(18))
    group=(right-left)/max(1,len(labels));width=group*.72/max(1,len(series))
    for s,(label,values) in enumerate(series):
        for i,value in enumerate(values):
            x=left+group*i+group*.14+s*width;y=bottom-(bottom-top)*value/maximum
            d.rectangle((x,y,x+width*.92,bottom),fill=COLORS[s%len(COLORS)])
        d.rectangle((100+s*275,825,122+s*275,847),fill=COLORS[s%len(COLORS)])
        d.text((130+s*275,822),label,fill='#17212b',font=fonts(18))
    for i,label in enumerate(labels):
        box=d.textbbox((0,0),label,font=fonts(17));x=left+group*(i+.5)-(box[2]-box[0])/2
        d.multiline_text((x,705),label,fill='#17212b',font=fonts(17),align='center')
    d.text((left,115),ylabel,fill='#445363',font=fonts(18))
    im.save(path)

def similarity_plot(out,metrics,field):
    kinds=['positive','random_same_country','hard_name','hard_address','shared_token_or_number']
    im,d=base(f'{field.title()} similarity: positive and sampled negative pairs',
              'Dots: median; thick interval: p25-p75; thin interval: p5-p95. All similarities scaled 0-1.')
    left,right=365,1400
    for tick in range(11):
        x=left+(right-left)*tick/10
        d.line((x,155,x,750),fill='#dce2e8')
        d.text((x-10,765),f'{tick/10:.1f}',fill='#445363',font=fonts(18))
    for i,kind in enumerate(kinds):
        y=210+i*115
        d.text((40,y-15),kind.replace('_',' '),fill='#17212b',font=fonts(22))
        for offset,feature,color in [(-18,field+'_ratio',COLORS[0]),(18,field+'_tfidf_cosine',COLORS[1])]:
            stats=metrics['pairs']['by_kind'][kind][feature]
            coord=lambda p:left+(right-left)*stats[p]
            d.line((coord('p5'),y+offset,coord('p95'),y+offset),fill=color,width=3)
            d.line((coord('p25'),y+offset,coord('p75'),y+offset),fill=color,width=12)
            x=coord('p50');d.ellipse((x-7,y+offset-7,x+7,y+offset+7),fill=color,outline='white',width=2)
    d.text((420,835),'Blue: normalized Indel ratio',fill=COLORS[0],font=fonts(22))
    d.text((900,835),'Orange: char TF-IDF cosine',fill=COLORS[1],font=fonts(22))
    im.save(out/'figures'/f'{field}_similarity.png')

def candidate_plot(out,metrics):
    simple_n=metrics['pairs']['method']['anchors']
    tf_n=len(metrics['blocking']['retrieval_query_ids'])
    im,d=base('Candidate recall versus candidate volume',
              f'Full target corpus. Simple strategies: {simple_n:,} sampled S1s; TF-IDF: {tf_n}. See report for cluster-bootstrap intervals.')
    rows=metrics['blocking']['results'];left,right,top,bottom=150,1400,150,720
    xmax=math.ceil(math.log10(max(r['mean_candidates'] for r in rows)+1))
    for tick in range(xmax+1):
        x=left+(right-left)*tick/xmax
        d.line((x,top,x,bottom),fill='#dce2e8');d.text((x-20,bottom+15),f'{10**tick-1:,}',fill='#445363',font=fonts(18))
    for tick in range(6):
        y=bottom-(bottom-top)*tick/5
        d.line((left,y,right,y),fill='#dce2e8');d.text((55,y-10),f'{tick*20}%',fill='#445363',font=fonts(18))
    for i,r in enumerate(rows):
        x=left+(right-left)*math.log10(r['mean_candidates']+1)/xmax;y=bottom-(bottom-top)*r['recall']
        color=COLORS[1] if 'tfidf' in r['strategy'] else COLORS[0]
        d.ellipse((x-6,y-6,x+6,y+6),fill=color)
        # Label only selected points to keep plot legible; complete table is in report.
        labels={'numeric_address_token':'numeric token','rare_name_token':'rare name token','exact_normalized_name':'exact name',
                'tfidf_union@50':'TF-IDF union K=50','tfidf_union@5':'TF-IDF union K=5'}
        if r['strategy'] in labels:d.text((min(x+10,1200),y+10),labels[r['strategy']],fill=color,font=fonts(18))
    d.text((450,800),'Mean candidates per S1 (log10(count + 1) axis)',fill='#17212b',font=fonts(23))
    d.text((80,115),'Pair recall',fill='#445363',font=fonts(20));im.save(out/'figures'/'candidate_recall.png')

def report(metrics,out):
    files=metrics['files'];gt=metrics['ground_truth'];pairs=metrics['pairs'];blocks=metrics['blocking']['results']
    shift=shifts(metrics)
    from .shift import lexical_coverage
    metrics['lexical_coverage']=lexical_coverage(metrics,out)
    (out/'eda_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    hist={int(k):v for k,v in gt['match_count']['histogram'].items()}
    bars(out/'figures'/'matches_per_s1.png','Training match multiplicity',[str(k) for k in sorted(hist)],
         [('S1 entities',[100*hist[k]/gt['rows'] for k in sorted(hist)])],'% of all S1 entities','Exact ground truth; zero means singleton.')
    keys=list(files)
    labels=[k.replace('_source','\nS') for k in keys]
    bars(out/'figures'/'country_distribution.png','Country composition by source',labels,
         [(c,[100*files[k]['countries'].get(c,{'rows':0})['rows']/files[k]['rows'] for k in keys]) for c in ['India','US','France']],
         '% of rows','Exact counts; France appears only in test.')
    bars(out/'figures'/'missingness.png','Pandas-default missing-value rate',labels,
         [(field,[100*aggregate(files[k])['null'][col]/files[k]['rows'] for k in keys]) for field,col in [('Name','business_name'),('Address','business_address'),('Country','country')]],
         '% of rows','Default NA lexemes include empty cells; empty-cell counts are reported separately.')
    similarity_plot(out,metrics,'name');similarity_plot(out,metrics,'address');candidate_plot(out,metrics)
    sections=[]
    def add(title,text):sections.append(f'## {title}\n\n{text}\n')
    positive=pairs['by_kind']['positive']
    top_name=pairs['separation']['hard_name'][:5]
    add('Executive summary',
        f"The six source files contain **{sum(f['rows'] for f in files.values()):,} records**. Training ground truth represents **{gt['unique_s1']:,} Source 1 entities** and **{gt['matches_total']:,} labeled pairs**. "
        f"**{gt['buckets']['0']['percent']:.2f}%** of reference entities are singletons. The observed relation is "
        + ('one-to-many from S1, with uniquely assigned target records.' if gt['targets_with_multiple_s1']==0 and gt['maximum_matches']>1 else 'described by the multiplicity and ownership checks below.')
        +f"\n\nOn the {pairs['method']['anchors']:,}-entity diagnostic sample, name raw equality is {pct(positive['name_raw_equal']['mean'])}; normalized equality is {pct(positive['name_normalized_equal']['mean'])}. "
        f"Address raw equality is {pct(positive['address_raw_equal']['mean'])}; normalized equality is {pct(positive['address_normalized_equal']['mean'])}. "
        "These are sampled pair-weighted statistics, not population claims. No classifier, semantic embedding, final blocker, or submission was built.")
    add('Scope, methods, and reproducibility',
        'Only the seven official TSV files were read as data. The supplied local README was used to interpret labels and scoring. Original files were not edited; SHA-256 fingerprints are stored in JSON. '
        'Source reads use `pd.read_csv(path, sep="\\t", chunksize=50000, dtype=str, keep_default_na=False)` to preserve raw empty cells and NA-like strings. '
        'Default pandas NA membership is counted separately; a default-parser first-1,000-row read records inferred dtypes. TSV has no intrinsic type system: all four source fields are strings. '
        'A strict independent CSV pass checks widths and record counts. Disk-backed SQLite stores raw records and simple diagnostic forms; no full set of source files is resident in RAM. '
        'Per-country length/token histograms, missingness, uniqueness, duplicates and truth validation are exact. Pair signals use a reproducible proportional country-stratified S1 sample. '
        'Every true target of each sampled S1 is retained; standard errors therefore cluster by S1. Hard negatives are sampled-corpus examples, so their error rates do not estimate production precision. '
        'Full-universe blocking probes search all same-country training S2 and S3 records. TF-IDF vocabulary and IDF use a uniform training target sample; this remains an exploratory lexical diagnostic. '
        'Country gating is evaluated against cross-country truth, including excluded matches in the recall denominator.\n\n'
        f"Seed: `{metrics['metadata']['seed']}`. Similarity backend: **{pairs['method']['similarity_backend']}**. The fallback implements exact normalized Indel, token-sort, token-set and unit-cost Levenshtein formulas; it is not a call to the unavailable RapidFuzz package. "
        'The included independent method check compares both distances against dynamic programming on 500 random string pairs.\n\n'
        'Reproduce: `python analysis/run_eda.py --data-root dataset-20260925T050739Z-1-001/dataset`. '
        'Dependencies and the exact locally used command are in `analysis/README.md`. Cache fingerprints include input sizes/mtime and configuration; remove only `analysis/cache/` for a clean run. '
        'No network access or dependency download is required. Similarities run in [0,1]. Undefined numeric/postal comparisons remain missing, never disguised as disagreement or agreement.')
    dims=[]
    for key,m in files.items():
        a=aggregate(m)
        dims.append([key,m['bytes']/1e6,m['rows'],m['unique_ids'],m['duplicate_id_excess'],m['invalid_prefix'],m['duplicate_full_row_excess']])
    dims.append(['train_ground_truth',gt['bytes']/1e6,gt['rows'],gt['unique_s1'],gt['duplicate_s1_rows'],gt['invalid_source1_syntax'],'see GT checks'])
    add('Dataset dimensions and integrity',table(['File','MB (decimal)','Rows','Unique IDs','Duplicate ID excess','Bad prefixes/syntax','Duplicate full-row excess'],dims)
        +'\nAll source schemas: `entity_id, business_name, business_address, country`; ground truth: `source1_entity_id, matched_entity_ids`. '
        'Per-file inferred and loaded dtypes, strict parser results, and hashes are in JSON. '
        f"Unexpected country labels: {json.dumps({k:m['unexpected_countries'] for k,m in files.items()})}.")
    add('Ground-truth structure and singleton statistics',table(['Matches per S1','S1 count','% of S1'],[[k,v['count'],v['percent']] for k,v in gt['buckets'].items()])
        +'\n'+table(['Statistic','Value'],[['Mean matches',gt['match_count']['mean']],['Median matches',gt['match_count']['median']],['Maximum matches',gt['maximum_matches']],['S2 only',gt['s2_only']],['S3 only',gt['s3_only']],['Both S2 and S3',gt['both']],['Unique referenced targets',gt['unique_targets']],['Targets assigned to multiple S1s',gt['targets_with_multiple_s1']],['S1 missing truth rows',gt['s1_without_truth']],['Invalid S1 references',gt['invalid_s1_references']],['Invalid S2 references',gt['invalid_s2_references']],['Invalid S3 references',gt['invalid_s3_references']],['Duplicate IDs inside lists',gt['duplicate_ids_in_lists']],['Malformed lists',gt['malformed_lists']],['Invalid target syntax',gt['invalid_target_syntax']],['Cross-country positive pairs',gt['cross_country_positive_pairs']]])
        +'\n'+table(['Country','S1','Singletons','Singleton %','Pairs'],[[c,v['s1'],v['singletons'],v['singleton_percent'],v['pairs']] for c,v in gt['by_country'].items()])
        +'\n![Exact match multiplicity](figures/matches_per_s1.png)\n\nThe metric gives a correct empty prediction full credit for a singleton and any false match zero credit. Pair prevalence therefore cannot replace an entity-level analysis.')
    rows=[]
    for key,m in files.items():
        a=aggregate(m)
        for col in ['entity_id','business_name','business_address','country']:
            rows.append([key,col,a['null'][col],100*a['null'][col]/m['rows'],a['literal_empty'][col],a['null'][col]-a['literal_empty'][col],a['whitespace_only'][col]])
    add('Missingness and empty cells',table(['File','Column','Pandas-null count','Null %','Literal empty count','Nonempty NA lexemes','Whitespace-only'],rows)
        +'\nPandas-null counts include empty strings by default. Empty strings are also reported separately, so these columns must not be summed. '
        'Only non-null field values contribute to length/token summaries. Whitespace-only strings are counted separately from zero-length fields. '
        f"Ground truth has {gt['blank_lists']:,} literal empty match lists; these are valid singleton labels.\n\n![Missingness](figures/missingness.png)")
    rows=[]
    for key,m in files.items():
        a=aggregate(m)
        rows.append([key,a['name_length']['mean'],a['name_length']['median'],a['name_tokens']['median'],a['name_tokens']['p95'],m['unique_names_global'],a['address_length']['mean'],a['address_tokens']['median'],m['unique_addresses_global']/m['rows'],
                     1-a['numeric_count']['histogram'].get(0,0)/a['numeric_count']['n'],1-a['postal_count']['histogram'].get(0,0)/a['postal_count']['n']])
    add('Field quality by source',table(['File','Name mean chars','Name median chars','Name median tokens','Name p95 tokens','Unique nonempty names','Address mean chars','Address median tokens','Unique address / all rows','Any digits rate','5/6-digit token rate'],rows)
        +'\nToken counts use whitespace on raw strings. Numeric tokens are contiguous Unicode digits; postal-like tokens are standalone 5–6 digits with word boundaries. '
        'These are generic shapes, not validated postal codes; street/building numbers can be mistaken for postal-like numbers. '
        'Full token-count, character, and numeric-count distributions are retained in JSON.')
    rows=[];top=[]
    for key,m in files.items():
        for c,g in m['countries'].items():
            u=g['uniqueness'];rows.append([key,c,g['rows'],100*g['rows']/m['rows'],u['name']['rows_in_repeated_values']/g['rows'],u['nn']['rows_in_repeated_values']/g['rows'],u['address']['unique_nonempty']/g['rows']])
            top.append(f"**{key} / {c}:** "+'; '.join(f"`{x['name']}` ({x['count']:,})" for x in u['top_normalized_names'][:5]))
    add('Country mix and repeated names',table(['File','Country','Rows','Country %','Rows in repeated raw names','Rows in repeated normalized names','Unique address ratio'],rows)
        +'\nRepeated-name rates count every row in a value group of size >1, not just the excess duplicates. '
        'Source 1 entity deduplication does not imply name uniqueness. Repeated names may describe chains or unrelated generic names; names alone cannot establish which.\n\n'
        +'\n\n'.join(top)+'\n\n![Country distribution](figures/country_distribution.png)')
    rows=[]
    for kind,g in pairs['by_kind'].items():
        for field in ['name_raw_equal','name_normalized_equal','name_suffix_equal','name_ratio','name_token_set_ratio','name_tfidf_cosine','address_raw_equal','address_normalized_equal','address_ratio','address_tfidf_cosine','address_numeric_overlap','address_postal_equal']:
            v=g[field];rows.append([kind,field,v['valid_n'],v['mean'],v['p5'],v['p50'],v['p95']])
    add('Positive-match and hard-negative similarity patterns',table(['Pair stratum','Signal','Valid N','Mean','p5','p50','p95'],rows)
        +'\nName and address fuzzy scores use the conservative NFKC/lowercase/alphanumeric-space form. Raw, lowercase and whitespace equality remain distinct diagnostics. '
        'The `name_punctuation_equal` signal combines punctuation spacing with NFKC/lowercase; the normalization section separately reports punctuation deletion. '
        'Empty/empty equality is never evidence of a match. Reported equality percentiles are binary; their means give equality rates.\n\n'
        '![Name similarity](figures/name_similarity.png)\n\n![Address similarity](figures/address_similarity.png)')
    rows=[]
    for kind,ranking in pairs['separation'].items():
        for r in ranking[:6]:rows.append([kind,r['feature'],r['auc'],r['positive_valid'],r['negative_valid'],r['positive_ge_09'],r['negative_ge_09']])
    add('Strongest separating signals',table(['Negative stratum','Feature','Univariate ROC AUC','Positive valid N','Negative valid N','Positive >=0.9','Negative >=0.9'],rows)
        +'\nAUC here is only a descriptive ranking of one signal at a time; no model was trained. '
        'Easy random negatives inflate separation. Hard-name and hard-address comparisons deliberately stress the corresponding field. '
        'Sample construction and missing-feature denominators differ by stratum; do not interpret AUC as precision or choose operational thresholds from it.')
    rows=[]
    for label,counts in pairs['risks'].items():rows.append([label]+[counts.get(k,0) for k in ['positive','random_same_country','hard_name','hard_address','shared_token_or_number']])
    examples=[]
    for label,exs in pairs['examples'].items():
        if exs:
            e=exs[0];a,b=e['left'],e['right']
            examples.append(f"- **{label}** ({e['kind']}): `{a['id']}` {a['name']} — {a['address']} ↔ `{b['id']}` {b['name']} — {b['address']}. This pair is not listed in ground truth.")
    add('False-match risks and inspected examples',table(['Diagnostic flag','Positive','Random','Hard name','Hard address','Shared tokens/numbers'],rows)
        +'\n'+'\n'.join(examples)+'\n\nFlags overlap and are not mutually exclusive. Common-name repetition is observed, but actual chain membership is unverified. '
        'Likewise a shared normalized address indicates a textual collision, not proof that businesses share a physical building.')
    rows=[]
    for field,data in pairs['normalization'].items():
        for rule in ['raw','lowercase','unicode_nfkc','punctuation_removal','ampersand_and','whitespace','normalized','legal_suffix','token_sort','digit_extraction']:
            p=data[f'positive/{rule}'];h=data[f'hard_name/{rule}'];a=data[f'hard_address/{rule}'];n=data[f'random_same_country/{rule}']
            rows.append([field,rule,p['rate'],p['new_equal_vs_raw'],h['rate'],a['rate'],n['rate']])
    add('Normalization diagnostics and collision risk',table(['Field','Transformation','Positive equality rate','Positive new equal vs raw','Hard-name collision rate','Hard-address collision rate','Random collision rate'],rows)
        +'\nEach transformation is tested independently against raw text, except explicitly combined forms: `normalized` = NFKC + lowercase + replace non-alphanumeric with spaces + collapse whitespace; '
        '`ampersand_and` substitutes `&` with `and` before normalized; `legal_suffix` repeatedly removes trailing tokens from the normalized form; `token_sort` sorts normalized tokens. '
        'The suffix vocabulary is a deliberately limited hypothesis: inc, incorporated, corp, corporation, llc, llp, ltd, limited, pvt, private, plc, co, company. '
        'Digit extraction joins every digit run in order and discards all text. Unicode NFKC preserves accents and is not transliteration or accent folding. '
        'Positive new-equality counts are absolute sample counts. Negative equality measures collisions under exhaustive-label assumptions; it is not a population false-positive rate. '
        'Do not finalize any rule from its positive gain alone. No France-specific suffix vocabulary was imported.')
    rows=[]
    for b in blocks:
        rows.append([b['strategy'],b['anchors'],b['true_pairs'],pct(b['recall']),f"{pct(b['recall_cluster_bootstrap_95'][0])}–{pct(b['recall_cluster_bootstrap_95'][1])}",b['mean_candidates'],b['median_candidates'],b['p95_candidates'],b['max_candidates'],pct(b['all_matches_recovered_non_singletons'])])
    add('Candidate-generation feasibility',table(['Strategy','S1 N','True pairs','Pair recall','95% cluster-bootstrap interval','Mean candidates','Median','p95','Max','All matches recovered, matched S1'],rows)
        +'\nAll counts use the **full combined S2+S3 corpus**, with same-country gating. No top-K strategy is measured against a reduced target universe. '
        'Rare means document frequency ≤0.1% of the full same-country target corpus. Numeric blocking accepts any shared address digit run, including common street numbers. '
        'Simple union combines exact normalized name, suffix-stripped name, rare name token and numeric address token without caps. '
        'K is over the combined sources, not K per source. Name/address union uses K from each, so it can yield up to 2K candidates. '
        'TF-IDF returns strictly positive scores only; ties use lexicographic IDs solely for deterministic ordering. TF-IDF uses char_wb 3–5 grams, sample-fitted vocabulary/IDF, min_df=2 and a 120,000-feature cap. '
        'Compare TF-IDF unions with `simple_union_on_tfidf_queries` for a fair same-query ablation. '
        'Bootstrap intervals describe query-sampling uncertainty only, not variation from another TF-IDF corpus. A 120-query probe cannot establish a tight high-recall guarantee. '
        'Maxima are observed sample maxima, not population worst cases.\n\n![Candidate feasibility](figures/candidate_recall.png)')
    rows=[]
    for key,s in shift.items():
        rows.append([key,s['name_length']['mean_delta'],s['address_length']['mean_delta'],s['name_tokens']['mean_delta'],s['address_tokens']['mean_delta'],s['numeric_count']['histogram_total_variation'],s['character_total_variation'],s['non_ascii_names']['train_rate'],s['non_ascii_names']['test_rate']])
    add('Train/test distribution shift',table(['Source/country','Name chars Δ','Address chars Δ','Name tokens Δ','Address tokens Δ','Numeric-count TV','Name character TV','Train non-ASCII names','Test non-ASCII names'],rows)
        +'\nDeltas are test minus train. Total variation (TV) is 0 for equal distributions and 1 for disjoint distributions; histogram bins use exact lengths/counts. '
        'Character TV is character-frequency weighted, not row weighted. Both overall and within-country comparisons are shown to separate country-mix shift from changes within India/US. '
        'Row counts, missingness, country percentages and repeat-name rates are reported in preceding exact tables; complete histograms and source-country metrics are in JSON.')
    density=[]
    for split in ['train','test']:
        for country in ['ALL','India','US','France']:
            counts=[files[f'{split}_source{s}']['rows'] if country=='ALL' else files[f'{split}_source{s}']['countries'].get(country,{'rows':0})['rows'] for s in [1,2,3]]
            if counts[0]:density.append([split,country,*counts,(counts[1]+counts[2])/counts[0]])
    add('Source population balance',table(['Split','Country','S1 rows','S2 rows','S3 rows','Target records per reference record'],density)
        +'\nThis ratio describes file populations, not labeled match multiplicity. Without test labels, a rise cannot distinguish more true matches from more distractors or different reference coverage.')
    rows=[]
    for key,m in files.items():
        if 'France' not in m['countries']:continue
        g=m['countries']['France'];u=g['uniqueness']
        rows.append([key,g['rows'],g['name_length']['mean'],g['address_length']['mean'],g['name_tokens']['mean'],g['address_tokens']['mean'],g['non_ascii_names']/g['rows'],g['non_ascii_addresses']/g['rows'],u['nn']['rows_in_repeated_values']/g['rows'],1-g['postal_count']['histogram'].get('0',g['postal_count']['histogram'].get(0,0))/g['postal_count']['n']])
    add('France observations',table(['File','Rows','Name chars','Address chars','Name tokens','Address tokens','Non-ASCII name rate','Non-ASCII address rate','Repeated norm-name rows','Postal-shape rate'],rows)
        +'\nFrance has no labeled training counterpart. These measurements can support mechanical compatibility of Unicode-preserving character/token/number diagnostics, but **cannot establish matching generalization or precision in France**. '
        'The generic postal detector reports only digit shapes. No geocoding, registry, translation, external language lexicon or business lookup was used. '
        'Country should be treated as an open string label. Any later claims about France quality will need an evaluation design that respects its lack of labels.')
    rows=[]
    for key,fields in metrics['lexical_coverage']['strata'].items():
        rows.append([key,fields['name']['coverage']['valid_n'],fields['name']['coverage']['mean'],fields['name']['zero_known_grams'],fields['address']['coverage']['valid_n'],fields['address']['coverage']['mean'],fields['address']['zero_known_grams']])
    add('Training-vocabulary coverage on held-out text',table(['File/country','Name N','Mean name gram coverage','Names with zero known grams','Address N','Mean address gram coverage','Addresses with zero known grams'],rows)
        +'\n'+metrics['lexical_coverage']['method']+' Training rows provide an in-sample baseline because the vocabulary was fitted on their enclosing target sample. Test text was never used to fit vocabulary or IDF. '
        'Unicode character categories for the full name population are also in JSON. Non-ASCII text can reflect punctuation, accents or a different script; it is not automatically transliteration.')
    leak=metrics['leakage'];rows=[]
    for key,v in leak['comparisons'].items():rows.append([key,v['raw_names'],v['normalized_names'],v['raw_name_address'],v['normalized_name_address']])
    add('Data quality and leakage checks',table(['Train/test comparison','Common raw (country,name)','Common normalized (country,name)','Common raw (country,name,address)','Common normalized (country,name,address)'],rows)
        +'\nCounts are distinct text keys per source comparison; they must not be summed across comparisons because keys can recur in several sources. '
        'Text overlap is a possible validation-contamination risk, not automatically label leakage.\n\n'
        +table(['Source','Overlapping train/test IDs','Identical full rows including ID'],[[f'S{s}',leak[f's{s}_id_overlap'],leak[f's{s}_identical_full_rows']] for s in [1,2,3]])
        +'\nNumeric ID suffix lengths, final-digit counts, and sampled positive/negative ID-distance distributions are stored in JSON. '
        'The audit checks syntax, repeated suffixes and conspicuous numerical alignment; absence of a simple pattern does not prove absence of all possible leakage. '
        'IDs are excluded from similarity and candidate signals; only source prefixes route references. No observed overlap was exploited.')
    insights=[
        f"Singletons are {gt['buckets']['0']['percent']:.2f}% of S1s; every future comparison must preserve entity-level accounting and explicitly measure false matches on zero-match entities.",
        f"Multiplicity reaches {gt['maximum_matches']} matches per S1 (mean {gt['match_count']['mean']:.3f}); one-to-one assumptions would discard labeled relationships.",
        f"Raw name equality covers {pct(positive['name_raw_equal']['mean'])} of sampled positives; normalizing raises/changes this to {pct(positive['name_normalized_equal']['mean'])}, with collision rates shown above.",
        f"Raw address equality covers {pct(positive['address_raw_equal']['mean'])} of sampled positives; address ratio p5 is {positive['address_ratio']['p5']:.3f}. Equality alone cannot cover noisy positives.",
        'Hard negatives must be judged separately by how they were selected; their sample mixture does not reproduce real candidate prevalence.',
        'Numeric/postal overlap needs a distinct missingness state and full-corpus fanout checks; a digit match is not a validated location match.',
        'Candidate recall and all-matches-per-entity recovery answer different questions. Report both, including tail candidate counts and singleton fanout.',
        'France is an unlabeled distribution shift. Unicode-preserving mechanics are testable, while French matching quality remains unverified.',
        'Repeated names and overlapping train/test text require contamination-aware evaluation diagnostics; IDs and overlaps must not become matching shortcuts.',
        'The full-corpus TF-IDF probes are preliminary, with only 120 references and sample-fitted IDF. Their uncertainty is too large to finalize a high-recall search policy.'
    ]
    add('Major risks and recommended next experiments','\n'.join(f'{i+1}. {x}' for i,x in enumerate(insights))
        +'\n\nFor the next analysis round, expand the query sample and repeat the same full-universe blocking probes; audit sampled false positives and missed positives by country, source, name length, address missingness and multiplicity; '
        'compare normalization gains against collisions on identical query sets; and quantify retrieval stability under vocabulary/IDF resampling. '
        'These are evidence-gathering experiments, not a proposed final model or inference pipeline. Stop here until the next phase is authorized.')
    rows=[]
    for kind,g in pairs['by_kind'].items():
        for feature,s in g.items():rows.append([kind,feature,s['valid_n'],s['missing_n']]+[s[f'p{p}'] for p in [1,5,10,25,50,75,90,95,99]])
    add('Appendix: complete similarity percentiles',table(['Pair stratum','Feature','Valid N','Missing N','p1','p5','p10','p25','p50','p75','p90','p95','p99'],rows)
        +'\nCountry × target-source distributions for every pair stratum are also stored in `eda_metrics.json` under `pairs.by_country_source`.')
    (out/'dataset_report.md').write_text('# Amazon ML Challenge 2026 — Dataset analysis\n\n'+'\n'.join(sections),encoding='utf-8')
