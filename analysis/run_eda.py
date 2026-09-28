"""Offline, reproducible Phase-1 EDA. No model training or submission generation.

Run: python analysis/run_eda.py --data-root <official dataset directory>
Optional --extra-site-packages points to locally installed scipy/scikit-learn.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import hashlib

def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--seed',type=int,default=20260925)
    parser.add_argument('--anchors',type=int,default=1200)
    parser.add_argument('--corpus-per-stratum',type=int,default=10000)
    parser.add_argument('--extra-site-packages',type=Path)
    parser.add_argument('--stage',choices=['all','load','analyze','report'],default='all')
    args=parser.parse_args()
    if args.extra_site_packages:
        sys.path.append(str(args.extra_site_packages))
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
    out=args.output.resolve()
    (out/'cache').mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log',encoding='utf-8')])
    if args.data_root is None:
        roots=list(Path(__file__).resolve().parents[1].glob('dataset*/dataset'))
        if len(roots)!=1:
            raise ValueError('Pass --data-root; expected exactly one official dataset root')
        args.data_root=roots[0]
    official=[args.data_root/s/f'{s}_source{i}.tsv' for s in ['train','test'] for i in [1,2,3]]+[args.data_root/'train'/'train_ground_truth.tsv']
    manifest={'inputs':{str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in official},
              'seed':args.seed,'anchors':args.anchors,'corpus_per_stratum':args.corpus_per_stratum,'schema_version':1}
    manifest_path=out/'cache'/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:
        logging.info('Input/configuration changed; invalidating derived analysis caches')
        for name in ['ground_truth.json','leakage.json','pair_results.json','blocking.json','tfidf.pkl','anchors.json','retrieval_top50.json','pair_features.csv.gz','lexical_coverage.json']:
            (out/'cache'/name).unlink(missing_ok=True)
        if args.stage in ['analyze','report']:
            raise ValueError('Inputs or settings changed; rerun --stage all to refresh source metrics')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from src.loaders import connect,load_file,exact_quality
    db=connect(out/'cache'/'audit.sqlite')
    metrics_path=out/'eda_metrics.json'
    if args.stage in ['analyze','report']:
        metrics=json.loads(metrics_path.read_text(encoding='utf-8'))
    else:
        files={}
        for split in ['train','test']:
            for source in [1,2,3]:
                path=args.data_root/split/f'{split}_source{source}.tsv'
                files[path.stem],_=load_file(db,path,out,args.seed+source, sample_size=args.corpus_per_stratum)
        logging.info('Building disk indexes')
        for name,cols in [('file','file'),('id','file,id'),('name','file,country,nn')]:
            db.execute(f'CREATE INDEX IF NOT EXISTS ix_{name} ON records ({cols})')
        db.commit()
        metrics={'metadata':{'seed':args.seed,'anchors_requested':args.anchors,
                            'corpus_per_stratum':args.corpus_per_stratum,'data_root':str(args.data_root.resolve()),
                            'input_policy':'Official seven TSVs only; no internet, external data, models, or submissions'},'files':files}
        if metrics_path.exists():
            previous=json.loads(metrics_path.read_text(encoding='utf-8'))
            same_inputs=all(previous.get('files',{}).get(k,{}).get('signature')==v['signature'] for k,v in files.items())
            if same_inputs and previous.get('quality_complete'):
                metrics['files']=previous['files']
                metrics['quality_complete']=True
        metrics_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.stage=='load':
        return
    if args.stage in ['all','analyze']:
        from src.ground_truth import analyze_ground_truth,leakage
        from src.diagnostics import analyze_pairs,blocking
        if 'quality_complete' not in metrics:
            exact_quality(db,metrics['files'])
            metrics['quality_complete']=True
            save(metrics_path,metrics)
        gt, anchors=analyze_ground_truth(db,args.data_root/'train'/'train_ground_truth.tsv',out,args.seed,args.anchors)
        metrics['ground_truth']=gt
        save(metrics_path,metrics)
        metrics['leakage']=leakage(db,out)
        save(metrics_path,metrics)
        metrics['pairs']=analyze_pairs(db,anchors,out,args.seed,args.corpus_per_stratum)
        save(metrics_path,metrics)
        metrics['blocking']=blocking(db,anchors,out,args.seed,metrics['pairs'])
        save(metrics_path,metrics)
    from src.reporting import report
    import importlib.metadata
    import sqlite3
    metrics['metadata']['python_version']=sys.version
    metrics['metadata']['sqlite_version']=sqlite3.sqlite_version
    metrics['metadata']['packages']={}
    for package in ['pandas','numpy','scipy','scikit-learn','Pillow']:
        try:metrics['metadata']['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:metrics['metadata']['packages'][package]='not installed in active path'
    metrics['metadata']['elapsed_seconds_this_invocation']=time.monotonic()-started
    if sys.platform=='win32':
        import ctypes
        class ProcessMemory(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('page_faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        pm=ProcessMemory();pm.cb=ctypes.sizeof(pm)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
        ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(pm),pm.cb)
        metrics['metadata']['main_process_peak_working_set_bytes']=pm.peak_working_set if ok else None
        metrics['metadata']['memory_note']='Main-process peak only; excludes two bounded TF-IDF transformation workers.'
    report(metrics,out)
    for p in official:
        if [p.stat().st_size,p.stat().st_mtime_ns] != manifest['inputs'][str(p.resolve())]:
            raise RuntimeError(f'Input changed during analysis: {p}')
    logging.info('Analysis complete: %s',out/'dataset_report.md')

def save(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)

if __name__=='__main__':
    main()
