"""Run the candidate-generation benchmark only; no matcher or test submission."""
import argparse,gzip,json,logging,sqlite3,sys,time
from collections import Counter
from pathlib import Path

def main():
    started=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-cache",type=Path,default=Path("analysis/cache"))
    parser.add_argument("--output",type=Path,default=Path("blocking"))
    parser.add_argument("--channels",nargs="*")
    args=parser.parse_args();out=args.output.resolve();(out/"artifacts").mkdir(parents=True,exist_ok=True);(out/"reports").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(out/"benchmark.log",encoding="utf-8")])
    config=json.loads((out/"config.yaml").read_text(encoding="utf-8"));db=sqlite3.connect(args.analysis_cache/"audit.sqlite");db.execute("PRAGMA cache_size=-65536")
    from src.diagnostics import create_benchmark,categorize_miss
    from src.tfidf_retrieval import retrieve_channel
    from src.exact_retrieval import retrieve_exact
    from src.numeric_retrieval import retrieve_numeric
    from src.combine import channel_sets,union_channels,rerank,write_provenance
    from src.evaluate import evaluate
    from src.reporting import write_reports
    benchmark=create_benchmark(db,out,config);queries=benchmark["development"]+benchmark["confirmation"]
    channels=args.channels or config["channels"];results={}
    for channel in channels:
        logging.info("Retrieval channel: %s",channel)
        if channel in ["name_char","address_char","suffix_char","name_word","address_word"]:results[channel]=retrieve_channel(db,queries,out,config,channel)
        elif channel=="exact_name":results[channel]=retrieve_exact(db,queries,out,config)
        elif channel=="numeric":results[channel]=retrieve_numeric(db,queries,out,config)
        else:raise ValueError(channel)
    dev=benchmark["development"];confirm=benchmark["confirmation"]
    individual={}
    for name,result in results.items():individual[name]=evaluate(dev,channel_sets(result,dev,config["channel_top_k"]))
    # Rank channels by standalone recall, then build the ablation incrementally.
    # This avoids repeatedly rebuilding every possible union on a memory-constrained host.
    order=sorted(results,key=lambda n:(individual[n]["pair_recall"],-individual[n]["candidates"]["mean"]),reverse=True)
    ablation=[];previous_recall=0;previous_mean=0;running={q["id"]:set() for q in dev}
    for name in order:
        for q in dev:running[q["id"]].update(r["id"] for r in results[name].get(q["id"],[]))
        metric=evaluate(dev,running)
        ablation.append({"configuration":" + ".join(order[:len(ablation)+1]),"pair_recall":metric["pair_recall"],"delta_pair_recall":metric["pair_recall"]-previous_recall,
                         "complete_entity_recall":metric["complete_entity_recall"],"mean_candidates":metric["candidates"]["mean"],
                         "delta_candidates":metric["candidates"]["mean"]-previous_mean,"p95_candidates":metric["candidates"]["p95"]})
        previous_recall=metric["pair_recall"];previous_mean=metric["candidates"]["mean"]
    max_cap=max(config["final_candidate_caps"]);_,dev_prov=rerank(results,dev,order,max_cap)
    cap_metrics={}
    for cap in config["final_candidate_caps"]:
        cand={q["id"]:{x[1] for x in dev_prov[q["id"]][:cap]} for q in dev}
        cap_metrics[cap]=evaluate(dev,cand)
    eligible=[c for c,m in cap_metrics.items() if m["pair_recall"]>=.995 and m["complete_entity_recall"]>=.985]
    selected_cap=min(eligible) if eligible else max(config["final_candidate_caps"],key=lambda c:(cap_metrics[c]["pair_recall"],-c))
    selected_dev=cap_metrics[selected_cap]
    _,confirm_full_prov=rerank(results,confirm,order,max_cap)
    confirm_prov={q["id"]:confirm_full_prov[q["id"]][:selected_cap] for q in confirm}
    confirm_candidates={q["id"]:{x[1] for x in confirm_prov[q["id"]]} for q in confirm};confirmation=evaluate(confirm,confirm_candidates)
    write_provenance(out/"artifacts"/"confirmation_candidate_provenance.tsv.gz",confirm,confirm_prov)
    # Independent source budgets from strongest first channel.
    strongest=max(individual,key=lambda n:individual[n]["pair_recall"])
    source_budget={}
    for k in (20,30,50):source_budget[f"{strongest}:{k}+{k}"]=evaluate(dev,channel_sets(results[strongest],dev,k,source_budget=True))
    misses=Counter();examples=[]
    qmap={q["id"]:q for q in confirm}
    for q in confirm:
        missing=set(q["matches"])-confirm_candidates[q["id"]]
        tmap={x["id"]:x for x in q["targets"]}
        for target_id in missing:
            cats=categorize_miss(q,tmap[target_id]);misses.update(cats)
            if len(examples)<30:examples.append({"query_id":q["id"],"target_id":target_id,"categories":cats,"query_name":q["name"],"query_address":q["address"],"target_name":tmap[target_id]["name"],"target_address":tmap[target_id]["address"]})
    target_rows=db.execute("SELECT COUNT(*) FROM records WHERE file IN ('train_source2','train_source3')").fetchone()[0]
    country_counts=dict(db.execute("SELECT country,COUNT(*) FROM records WHERE file IN ('train_source2','train_source3') GROUP BY country"))
    avg_same=sum(country_counts[q["country"]] for q in dev)/len(dev)
    artifact_bytes=sum(p.stat().st_size for p in (out/"artifacts").glob("*"))
    frozen=confirmation["pair_recall"]>=.995 and confirmation["complete_entity_recall"]>=.985 and confirmation["candidates"]["p95"]<=200
    summary={"seed":config["seed"],"target_rows":target_rows,"country_candidate_reduction":1-avg_same/target_rows,
             "individual_channels":individual,"greedy_order":order,"cap_metrics":cap_metrics,"selected_cap":selected_cap,
             "selected_configuration":"reciprocal-rank fusion of "+", ".join(order),"selected_development":selected_dev,
             "confirmation":confirmation,"source_budget_experiment":source_budget,"runtime_seconds":time.monotonic()-started,
             "artifact_bytes":artifact_bytes,"decision":"Blocking meets the stated freeze criteria." if frozen else "Blocking does not meet the stated freeze criteria. Next: inspect confirmation misses and add only a measured rescue channel targeted at the dominant class; do not start the matcher."}
    (out/"artifacts"/"benchmark_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write_reports(out,summary,{"categories":dict(misses),"examples":examples},ablation)
    logging.info("Complete. Confirmation pair recall %.6f, complete recall %.6f",confirmation["pair_recall"],confirmation["complete_entity_recall"])

if __name__=="__main__":main()
