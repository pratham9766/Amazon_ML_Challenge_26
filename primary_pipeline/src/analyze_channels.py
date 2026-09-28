"""Rank-failure diagnosis + union ablation over cached channel outputs (dev split only)."""
from __future__ import annotations

import argparse
import json

import polars as pl

from bench import build_truth, evaluate, rank_bucket_expr, subgroup_exprs
from common import BLOCKER, REPORTS, get_logger

log = get_logger("analyze")


def load_channel(tag, ch):
    return pl.read_parquet(BLOCKER / tag / f"{ch}.parquet").with_columns(pl.lit(ch).alias("ch"))


def union(chans: dict[str, pl.DataFrame], ks: dict[str, int]) -> pl.DataFrame:
    parts = [chans[c].filter(pl.col("rank") < k).select("s1", "tgt") for c, k in ks.items()]
    return pl.concat(parts).unique()


def rank_table(p_dev: pl.DataFrame, allc: pl.DataFrame, chan_names) -> pl.DataFrame:
    """Best per-source rank of every dev true pair, per channel and overall."""
    best = allc.group_by("s1", "tgt", "ch").agg(pl.col("rank").min())
    wide = best.pivot(on="ch", index=["s1", "tgt"], values="rank")
    h = p_dev.join(wide, on=["s1", "tgt"], how="left")
    h = h.with_columns(pl.min_horizontal([pl.col(c) for c in chan_names if c in h.columns]).alias("best"))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag")
    ap.add_argument("--channels", default="addr_w12,name_w12,comb_w12,codes")
    a = ap.parse_args()
    q, p = build_truth()
    names = a.channels.split(",")
    chans = {c: load_channel(a.tag, c) for c in names}
    p_dev = p.filter(pl.col("split") == "dev")
    allc = pl.concat([chans[c].select("s1", "tgt", "rank", "ch") for c in names])
    h = rank_table(p_dev, allc, names)
    lines = ["# Blocker rank-failure analysis (development, 20,000 queries)\n",
             "Rank = 0-based per-source rank inside each channel (top-200 retrieved per source). "
             "`best` = minimum rank over all channels. Denominator: all dev true pairs.\n"]
    # overall bucket table per channel
    rows = []
    for c in names + ["best"]:
        d = h.with_columns(rank_bucket_expr(c).alias("b")).group_by("b").agg(pl.len()).with_columns(
            (pl.col("len") / len(h) * 100).round(2).alias("pct"))
        rows.append({"channel": c, **{r["b"]: r["pct"] for r in d.iter_rows(named=True)}})
    order = ["<=20", "21-50", "51-100", "101-150", "151-200", "not_in_top200"]
    lines.append("| channel | " + " | ".join(order) + " |\n|" + "---|" * (len(order) + 1))
    for r in rows:
        lines.append(f"| {r['channel']} | " + " | ".join(f"{r.get(o, 0):.2f}" for o in order) + " |")
    # subgroup by best rank
    lines.append("\n## Best-rank buckets by subgroup (% of subgroup pairs)\n")
    lines.append("| subgroup | pairs | " + " | ".join(order) + " |\n|" + "---|" * (len(order) + 2))
    hb = h.with_columns(rank_bucket_expr("best").alias("b"))
    for g, e in subgroup_exprs().items():
        s = hb.filter(e)
        if not len(s):
            continue
        d = dict(s.group_by("b").agg(pl.len()).iter_rows())
        lines.append(f"| {g} | {len(s)} | " + " | ".join(f"{d.get(o, 0) / len(s) * 100:.2f}" for o in order) + " |")
    # union sweep
    lines.append("\n## Union sweep (per-channel per-source k)\n")
    lines.append("| config | pair recall | complete | at-least-one | mean cands | p95 | p99 |\n|---|---|---|---|---|---|---|")
    sweep = {}
    configs = []
    for k in (10, 20, 30, 50, 100, 200):
        for c in names:
            configs.append({c: k})
        configs.append({c: k for c in names})
    for cfg in configs:
        r = evaluate(union(chans, cfg), q, p, groups=False)
        key = "+".join(f"{c}@{k}" for c, k in cfg.items())
        sweep[key] = {k2: v for k2, v in r.items() if not k2.startswith("_")}
        lines.append(f"| {key} | {r['pair_recall']*100:.2f} | {r['complete_entity_recall']*100:.2f} | {r['at_least_one_recall']*100:.2f} | {r['mean']:.1f} | {r['p95']:.0f} | {r['p99']:.0f} |")
    txt = "\n".join(lines)
    (REPORTS / f"rank_analysis_{a.tag}.md").write_text(txt, encoding="utf-8")
    json.dump(sweep, open(BLOCKER / a.tag / "sweep.json", "w"), indent=1)
    h.write_parquet(BLOCKER / a.tag / "dev_true_ranks.parquet")
    print(txt)


if __name__ == "__main__":
    main()
