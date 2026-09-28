"""Markdown and CSV reporting for blocker research."""
import csv,json

def p(x):return "N/A" if x is None else f"{100*x:.3f}%"
def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(str(x) for x in row)+" |" for row in rows])

def write_reports(out,summary,misses,ablation):
    best=summary["selected_development"];confirm=summary["confirmation"]
    rows=[]
    for name,r in summary["individual_channels"].items():
        rows.append([name,p(r["pair_recall"]),p(r["complete_entity_recall"]),f"{r['candidates']['mean']:.1f}",f"{r['candidates']['p95']:.1f}"])
    abr=[]
    for r in ablation:
        abr.append([r["configuration"],p(r["pair_recall"]),p(r["delta_pair_recall"]),p(r["complete_entity_recall"]),f"{r['mean_candidates']:.1f}",f"{r['delta_candidates']:.1f}",f"{r['p95_candidates']:.1f}"])
    breakdown=[]
    for key,v in confirm["breakdowns"].items():breakdown.append([key,p(v["pair_recall"]),p(v["complete_entity_recall"]),v["pairs"]])
    report=f"""# Blocking benchmark report

## Benchmark definition

The benchmark contains 20,000 development and 7,500 untouched confirmation S1 queries. It is deterministic (seed {summary['seed']}) and disjoint. A stable-hash pool was proportionally stratified by country, singleton/multiplicity, and S2-only/S3-only/both-source profile; diagnostic flags include short names, missing target addresses, non-ASCII targets, weak lexical overlap, and repeated names. Every evaluation searches all {summary['target_rows']:,} training S2+S3 records. IDs are used only as identifiers/source prefixes. Country gating is generic string equality; training truth loses zero pairs and reduces the search universe by {p(summary['country_candidate_reduction'])} on average.

The offline environment lacks a sparse-top-N or ANN package. Classical `char_wb` TF-IDF results from EDA are retained as a small-query baseline. The scalable channels here use exact IDF-weighted word features and bounded Unicode character fragments in a streaming inverted design. This is an explicit approximation, not mislabeled full-matrix TF-IDF.

For context, the earlier 120-query EDA baseline (name/address character TF-IDF union) reached 92.23% pair recall at k=10, 93.69% at k=20, and 94.66% at k=50, with about 98 union candidates at k=50. These exploratory figures are not directly comparable to the stratified benchmark below.

## Individual channel results on development

{table(['Channel','Pair recall','Complete entity recall','Mean candidates','p95 candidates'],rows)}

## Union ablation on development

{table(['Configuration','Pair recall','Δ recall','Complete recall','Mean','Δ mean','p95'],abr)}

## Selected candidate generator

Development selected **{summary['selected_configuration']}** at cap {summary['selected_cap']}. Development pair recall is {p(best['pair_recall'])}; complete-entity recall is {p(best['complete_entity_recall'])}. The untouched confirmation result is {p(confirm['pair_recall'])} pair recall and {p(confirm['complete_entity_recall'])} complete-entity recall.

Confirmation candidate volume: mean {confirm['candidates']['mean']:.2f}, median {confirm['candidates']['median']:.0f}, p90 {confirm['candidates']['p90']:.0f}, p95 {confirm['candidates']['p95']:.0f}, p99 {confirm['candidates']['p99']:.0f}, max {confirm['candidates']['max']}; singleton mean {confirm['singleton_mean']:.2f}, singleton p95 {confirm['singleton_p95']:.0f}.

## Confirmation breakdowns

{table(['Subgroup','Pair recall','Complete recall','Pairs'],breakdown)}

## Source-budget experiment

{json.dumps(summary['source_budget_experiment'],indent=2)}

## Remaining misses

The companion missed-positive report lists categories and representative examples. Counts overlap because one pair can have several failure properties.

## Efficiency

Total benchmark runtime: {summary['runtime_seconds']/60:.1f} minutes. Input index: SQLite disk cache. Generated artifact bytes: {summary['artifact_bytes']:,}. Peak RAM was not reliably available from the host API. Each retrieval channel scans targets in batches/streaming form and keeps only bounded per-query heaps. Gzip TSV/JSON is used because no local Parquet engine is installed; this limitation is explicit and artifacts remain resumable.

## Decision

{summary['decision']}
"""
    (out/"reports"/"blocking_report.md").write_text(report,encoding="utf-8")
    miss_rows=[[k,v] for k,v in sorted(misses["categories"].items(),key=lambda x:-x[1])]
    examples="\n".join(f"- **{e['query_id']} → {e['target_id']}**: {', '.join(e['categories'])}. Query `{e['query_name']}` / `{e['query_address']}`; target `{e['target_name']}` / `{e['target_address']}`." for e in misses["examples"])
    (out/"reports"/"missed_positive_analysis.md").write_text("# Missed positive analysis\n\n"+table(["Failure category","Missed pairs"],miss_rows)+"\n\n## Representative examples\n\n"+examples+"\n",encoding="utf-8")
    with (out/"reports"/"blocker_ablation.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(ablation[0]));w.writeheader();w.writerows(ablation)
