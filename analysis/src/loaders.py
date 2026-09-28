"""Chunked TSV audit and resumable, disk-backed exact statistics."""
import csv
import hashlib
import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES
from .normalization import norm, suffix, numbers, postal

COLS = ['entity_id', 'business_name', 'business_address', 'country']

def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('CREATE TABLE IF NOT EXISTS records (file TEXT,id TEXT,name TEXT,address TEXT,country TEXT,nn TEXT,an TEXT,sn TEXT)')
    return db

def hist_stats(h):
    count = sum(h.values())
    if not count:
        return {'n': 0}
    keys = sorted(h)
    cum = np.cumsum([h[k] for k in keys])
    def at(i):
        return keys[int(np.searchsorted(cum, i + 1))]
    def quantile(q):
        x = (count-1)*q
        lo, hi = int(np.floor(x)), int(np.ceil(x))
        return at(lo)+(at(hi)-at(lo))*(x-lo)
    return {'n': count, 'mean': sum(k*v for k,v in h.items())/count,
            'median': quantile(.5), 'p5': quantile(.05), 'p95': quantile(.95),
            'histogram': dict(sorted(h.items()))}

def new_group():
    return {'rows': 0, 'null': Counter(), 'literal_empty': Counter(), 'whitespace_only': Counter(),
            'name_length': Counter(), 'address_length': Counter(), 'name_tokens': Counter(),
            'address_tokens': Counter(), 'numeric_count': Counter(), 'postal_count': Counter(),
            'characters': Counter(), 'non_ascii_names': 0, 'non_ascii_addresses': 0}

def audit_structure(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8*1024*1024):
            digest.update(block)
    bad, rows = [], 0
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t', strict=True)
        header = next(reader)
        for row in reader:
            rows += 1
            if len(row) != len(header):
                if len(bad) < 20:
                    bad.append({'record': rows+1, 'fields': len(row)})
    return {'sha256': digest.hexdigest(), 'csv_record_count': rows, 'bad_width_examples': bad,
            'strict_csv_parse_ok': True}

def load_file(db, path, out, seed, chunk_size=50000, sample_size=10000):
    key = path.stem
    cache = out/'cache'/f'{key}.json'
    sample_path = out/'cache'/f'{key}_sample.json'
    signature = [path.stat().st_size, path.stat().st_mtime_ns]
    if cache.exists() and sample_path.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved['signature'] == signature and saved['seed'] == seed and saved['sample_size'] == sample_size:
            logging.info('Reusing validated file cache: %s', key)
            return saved, json.loads(sample_path.read_text(encoding='utf-8'))
    db.execute('DELETE FROM records WHERE file=?', (key,))
    schema = pd.read_csv(path, sep='\t', nrows=1000)
    if list(schema.columns) != COLS:
        raise ValueError(f'{path}: unexpected columns {list(schema.columns)}')
    groups, samples, duplicate_empty_ids = {}, {}, 0
    rng = np.random.default_rng(seed)
    expected_prefix = 'S'+key[-1]+'-'
    invalid_prefix = 0
    for chunk in pd.read_csv(path, sep='\t', chunksize=chunk_size, dtype=str, keep_default_na=False, on_bad_lines='error'):
        invalid_prefix += int((~chunk.entity_id.str.startswith(expected_prefix)).sum())
        null_masks = {c: chunk[c].isin(STR_NA_VALUES) for c in COLS}
        for country, g in chunk.groupby('country', dropna=False):
            state = groups.setdefault(country, new_group())
            state['rows'] += len(g)
            for col in COLS:
                state['null'][col] += int(null_masks[col].loc[g.index].sum())
                state['literal_empty'][col] += int((g[col] == '').sum())
                state['whitespace_only'][col] += int(((g[col] != '') & g[col].str.strip().eq('')).sum())
            for col, short in [('business_name','name'), ('business_address','address')]:
                valid = g.loc[~null_masks[col].loc[g.index], col]
                state[short+'_length'].update(valid.str.len())
                state[short+'_tokens'].update(valid.str.split().str.len())
                state['non_ascii_'+('names' if short=='name' else 'addresses')] += sum(not s.isascii() for s in valid)
                if short == 'name':
                    state['characters'].update(''.join(valid))
                else:
                    state['numeric_count'].update(len(numbers(s)) for s in valid)
                    state['postal_count'].update(len(postal(s)) for s in valid)
            candidate = g.copy()
            candidate['_priority'] = rng.random(len(g))
            samples[country] = pd.concat([samples.get(country, pd.DataFrame()), candidate], ignore_index=True).nsmallest(sample_size, '_priority')
        rows = []
        for entity_id, name, address, country in chunk.itertuples(index=False, name=None):
            # Preserve raw strings, including empty cells and literal NA lexemes.
            rows.append((key, entity_id, name, address, country, norm(name), norm(address), suffix(name)))
        db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?)', rows)
        db.commit()
        logging.info('%s: %s rows', key, format(sum(s['rows'] for s in groups.values()), ','))
    structure = audit_structure(path)
    result = {'path': str(path), 'bytes': signature[0], 'signature': signature,
              'seed': seed, 'sample_size': sample_size, 'columns': COLS,
              'pandas_inferred_dtypes_first_1000': {c:str(t) for c,t in schema.dtypes.items()},
              'loaded_dtypes': {c:'str (raw lexemes preserved)' for c in COLS},
              'rows': sum(s['rows'] for s in groups.values()), 'invalid_prefix': invalid_prefix,
              'unexpected_countries': [c for c in groups if c not in (['India','US'] if key.startswith('train') else ['India','US','France'])],
              'countries': {}, 'structure': structure}
    if result['rows'] != structure['csv_record_count'] or structure['bad_width_examples']:
        raise ValueError(f'Structural inconsistency: {key}')
    for country, g in groups.items():
        for col in ['name_length','address_length','name_tokens','address_tokens','numeric_count','postal_count']:
            g[col] = hist_stats(g[col])
        g['characters'] = dict(g['characters'].most_common())
        result['countries'][country] = g
    saved_sample = {country: df.drop(columns='_priority').to_dict('records') for country,df in samples.items()}
    cache.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
    sample_path.write_text(json.dumps(saved_sample, ensure_ascii=False), encoding='utf-8')
    return result, saved_sample

def exact_quality(db, metrics):
    cache_root=Path(db.execute('PRAGMA database_list').fetchone()[2]).parent
    for key, m in metrics.items():
        cache=cache_root/f'{key}_quality.json'
        if cache.exists():
            saved=json.loads(cache.read_text(encoding='utf-8'))
            if saved.get('signature')==m['signature']:
                m.update(saved)
                logging.info('Reusing exact quality cache: %s',key)
                continue
        logging.info('Exact uniqueness and repetitions: %s', key)
        n = m['rows']
        unique_ids = db.execute('SELECT COUNT(DISTINCT id) FROM records WHERE file=?',(key,)).fetchone()[0]
        m['unique_ids'], m['duplicate_id_excess'] = unique_ids, n-unique_ids
        m['duplicate_full_row_excess'] = 0 if unique_ids == n else n-db.execute('SELECT COUNT(*) FROM (SELECT id,name,address,country FROM records WHERE file=? GROUP BY id,name,address,country)',(key,)).fetchone()[0]
        for country,g in m['countries'].items():
            args=(key,country)
            stats={}
            for field in ['name','address','nn']:
                index='ix_name' if field=='nn' else 'ix_file'
                values=db.execute(f'SELECT COUNT(*),COALESCE(SUM(c),0),COALESCE(SUM(c-1),0),COALESCE(SUM(CASE WHEN c>1 THEN c ELSE 0 END),0) FROM (SELECT COUNT(*) c FROM records INDEXED BY {index} WHERE file=? AND country=? AND {field}<>\'\' GROUP BY {field})',args).fetchone()
                stats[field]={'unique_nonempty':values[0], 'nonempty_rows':values[1], 'duplicate_excess':values[2]}
                stats[field]['rows_in_repeated_values']=values[3]
            stats['top_normalized_names'] = [{'name':x,'count':c} for x,c in db.execute('SELECT nn,COUNT(*) c FROM records WHERE file=? AND country=? AND nn<>\'\' GROUP BY nn ORDER BY c DESC,nn LIMIT 12',args)]
            g['uniqueness']=stats
        # Sorted grouping avoids the random disk writes of a DISTINCT temp B-tree.
        m['unique_names_global']=db.execute('SELECT COUNT(*) FROM (SELECT name FROM records INDEXED BY ix_file WHERE file=? AND name<>\'\' GROUP BY name)',(key,)).fetchone()[0]
        m['unique_addresses_global']=db.execute('SELECT COUNT(*) FROM (SELECT address FROM records INDEXED BY ix_file WHERE file=? AND address<>\'\' GROUP BY address)',(key,)).fetchone()[0]
        cache.write_text(json.dumps(m,ensure_ascii=False),encoding='utf-8')
