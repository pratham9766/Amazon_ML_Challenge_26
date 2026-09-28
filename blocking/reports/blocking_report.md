# Blocking benchmark report

## Benchmark definition

The benchmark contains 20,000 development and 7,500 untouched confirmation S1 queries. It is deterministic (seed 20260925) and disjoint. A stable-hash pool was proportionally stratified by country, singleton/multiplicity, and S2-only/S3-only/both-source profile; diagnostic flags include short names, missing target addresses, non-ASCII targets, weak lexical overlap, and repeated names. Every evaluation searches all 10,320,219 training S2+S3 records. IDs are used only as identifiers/source prefixes. Country gating is generic string equality; training truth loses zero pairs and reduces the search universe by 47.934% on average.

The offline environment lacks a sparse-top-N or ANN package. Classical `char_wb` TF-IDF results from EDA are retained as a small-query baseline. The scalable channels here use exact IDF-weighted word features and bounded Unicode character fragments in a streaming inverted design. This is an explicit approximation, not mislabeled full-matrix TF-IDF.

For context, the earlier 120-query EDA baseline (name/address character TF-IDF union) reached 92.23% pair recall at k=10, 93.69% at k=20, and 94.66% at k=50, with about 98 union candidates at k=50. These exploratory figures are not directly comparable to the stratified benchmark below.

## Individual channel results on development

| Channel | Pair recall | Complete entity recall | Mean candidates | p95 candidates |
| --- | --- | --- | --- | --- |
| name_char | 31.543% | 18.028% | 50.1 | 75.0 |
| address_char | 70.692% | 46.986% | 74.6 | 75.0 |

## Union ablation on development

| Configuration | Pair recall | Δ recall | Complete recall | Mean | Δ mean | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| address_char | 70.692% | 70.692% | 46.986% | 74.6 | 74.6 | 75.0 |
| address_char + name_char | 79.742% | 9.050% | 60.975% | 123.7 | 49.2 | 150.0 |

## Selected candidate generator

Development selected **reciprocal-rank fusion of address_char, name_char** at cap 150. Development pair recall is 79.742%; complete-entity recall is 60.975%. The untouched confirmation result is 79.345% pair recall and 61.062% complete-entity recall.

Confirmation candidate volume: mean 123.73, median 147, p90 150, p95 150, p99 150, max 150; singleton mean 124.42, singleton p95 150.

## Confirmation breakdowns

| Subgroup | Pair recall | Complete recall | Pairs |
| --- | --- | --- | --- |
| country=US | 82.145% | 65.587% | 15671 |
| profile=both | 79.304% | 59.108% | 23898 |
| multiplicity=4+ | 78.888% | 53.992% | 17559 |
| nonascii_target_name | 74.585% | 51.525% | 9884 |
| source=S2 | 78.558% | 72.283% | 12611 |
| source=S3 | 80.082% | 72.834% | 13455 |
| weak_lexical | 74.831% | 50.057% | 10771 |
| multiplicity=2 | 80.192% | 70.421% | 2610 |
| country=India | 75.123% | 54.277% | 10395 |
| profile=singleton | N/A | N/A | 0 |
| multiplicity=0 | N/A | N/A | 0 |
| short_name | 79.545% | 62.354% | 3476 |
| multiplicity=3 | 80.556% | 65.206% | 5544 |
| profile=S2_only | 79.797% | 73.457% | 985 |
| multiplicity=1 | 76.771% | 76.771% | 353 |
| missing_target_address | 70.911% | 32.807% | 4414 |
| high_repeated_name | 66.921% | 43.371% | 3927 |
| profile=S3_only | 79.797% | 71.429% | 1183 |

## Source-budget experiment

{
  "address_char:20+20": {
    "queries": 20000,
    "true_pairs": 69268,
    "pair_recall": 0.6824652075994687,
    "complete_entity_recall": 0.4469652796183408,
    "at_least_one_recall": 0.8479194275112643,
    "candidates": {
      "mean": 27.96875,
      "median": 25.0,
      "p90": 40.0,
      "p95": 40.0,
      "p99": 40.0,
      "max": 40
    },
    "singleton_mean": 27.602643171806168,
    "singleton_p95": 40.0,
    "breakdowns": {
      "country=US": {
        "pair_recall": 0.6998947922146239,
        "complete_entity_recall": 0.4814879803474294,
        "pairs": 41822,
        "matched_entities": 11398
      },
      "profile=both": {
        "pair_recall": 0.6825081372058087,
        "complete_entity_recall": 0.4209676417613813,
        "pairs": 63904,
        "matched_entities": 16101
      },
      "multiplicity=4+": {
        "pair_recall": 0.6834909952807229,
        "complete_entity_recall": 0.37182400330510224,
        "pairs": 47253,
        "matched_entities": 9682
      },
      "short_name": {
        "pair_recall": 0.6768681212589303,
        "complete_entity_recall": 0.4513494318181818,
        "pairs": 10358,
        "matched_entities": 2816
      },
      "source=S2": {
        "pair_recall": 0.695489844683393,
        "complete_entity_recall": 0.6160524348875984,
        "pairs": 33480,
        "matched_entities": 17393
      },
      "source=S3": {
        "pair_recall": 0.6702805409634515,
        "complete_entity_recall": 0.5703067205371878,
        "pairs": 35788,
        "matched_entities": 17573
      },
      "country=India": {
        "pair_recall": 0.6559061429716534,
        "complete_entity_recall": 0.3942681130306683,
        "pairs": 27446,
        "matched_entities": 7467
      },
      "weak_lexical": {
        "pair_recall": 0.681910711778745,
        "complete_entity_recall": 0.4150405925081897,
        "pairs": 28492,
        "matched_entities": 7021
      },
      "multiplicity=3": {
        "pair_recall": 0.6783490731024834,
        "complete_entity_recall": 0.472193074501574,
        "pairs": 14295,
        "matched_entities": 4765
      },
      "nonascii_target_name": {
        "pair_recall": 0.6659419716973674,
        "complete_entity_recall": 0.39881713554987214,
        "pairs": 25298,
        "matched_entities": 6256
      },
      "multiplicity=2": {
        "pair_recall": 0.6845850999394306,
        "complete_entity_recall": 0.5523924894003635,
        "pairs": 6604,
        "matched_entities": 3302
      },
      "high_repeated_name": {
        "pair_recall": 0.6611995341614907,
        "complete_entity_recall": 0.4284195605953225,
        "pairs": 10304,
        "matched_entities": 2822
      },
      "profile=S3_only": {
        "pair_recall": 0.6660929432013769,
        "complete_entity_recall": 0.576766304347826,
        "pairs": 2905,
        "matched_entities": 1472
      },
      "missing_target_address": {
        "pair_recall": 0.5295906625020549,
        "complete_entity_recall": 0.0,
        "pairs": 12166,
        "matched_entities": 2845
      },
      "multiplicity=1": {
        "pair_recall": 0.6792114695340502,
        "complete_entity_recall": 0.6792114695340502,
        "pairs": 1116,
        "matched_entities": 1116
      },
      "profile=singleton": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "multiplicity=0": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "profile=S2_only": {
        "pair_recall": 0.7006913379422529,
        "complete_entity_recall": 0.6230650154798761,
        "pairs": 2459,
        "matched_entities": 1292
      }
    }
  },
  "address_char:30+30": {
    "queries": 20000,
    "true_pairs": 69268,
    "pair_recall": 0.6936247617947682,
    "complete_entity_recall": 0.45931619401007157,
    "at_least_one_recall": 0.8569838324940365,
    "candidates": {
      "mean": 39.40845,
      "median": 35.0,
      "p90": 60.0,
      "p95": 60.0,
      "p99": 60.0,
      "max": 60
    },
    "singleton_mean": 39.075770925110135,
    "singleton_p95": 60.0,
    "breakdowns": {
      "country=US": {
        "pair_recall": 0.7076897326765818,
        "complete_entity_recall": 0.4910510615897526,
        "pairs": 41822,
        "matched_entities": 11398
      },
      "profile=both": {
        "pair_recall": 0.6936029043565348,
        "complete_entity_recall": 0.4333892304825787,
        "pairs": 63904,
        "matched_entities": 16101
      },
      "multiplicity=4+": {
        "pair_recall": 0.6945379129367447,
        "complete_entity_recall": 0.38514769675686844,
        "pairs": 47253,
        "matched_entities": 9682
      },
      "short_name": {
        "pair_recall": 0.6882602819077042,
        "complete_entity_recall": 0.4641335227272727,
        "pairs": 10358,
        "matched_entities": 2816
      },
      "source=S2": {
        "pair_recall": 0.7047491039426523,
        "complete_entity_recall": 0.626861381015351,
        "pairs": 33480,
        "matched_entities": 17393
      },
      "source=S3": {
        "pair_recall": 0.6832178383815805,
        "complete_entity_recall": 0.5840778466966369,
        "pairs": 35788,
        "matched_entities": 17573
      },
      "country=India": {
        "pair_recall": 0.6721926692414195,
        "complete_entity_recall": 0.4108745145306013,
        "pairs": 27446,
        "matched_entities": 7467
      },
      "weak_lexical": {
        "pair_recall": 0.6935982030043522,
        "complete_entity_recall": 0.42956843754450935,
        "pairs": 28492,
        "matched_entities": 7021
      },
      "multiplicity=3": {
        "pair_recall": 0.6903112976565232,
        "complete_entity_recall": 0.4837355718782791,
        "pairs": 14295,
        "matched_entities": 4765
      },
      "nonascii_target_name": {
        "pair_recall": 0.6781563759981026,
        "complete_entity_recall": 0.4127237851662404,
        "pairs": 25298,
        "matched_entities": 6256
      },
      "multiplicity=2": {
        "pair_recall": 0.6950333131435493,
        "complete_entity_recall": 0.5639006662628709,
        "pairs": 6604,
        "matched_entities": 3302
      },
      "high_repeated_name": {
        "pair_recall": 0.6745923913043478,
        "complete_entity_recall": 0.443656980864635,
        "pairs": 10304,
        "matched_entities": 2822
      },
      "profile=S3_only": {
        "pair_recall": 0.680895008605852,
        "complete_entity_recall": 0.5903532608695652,
        "pairs": 2905,
        "matched_entities": 1472
      },
      "missing_target_address": {
        "pair_recall": 0.5399473943777742,
        "complete_entity_recall": 0.0,
        "pairs": 12166,
        "matched_entities": 2845
      },
      "multiplicity=1": {
        "pair_recall": 0.6890681003584229,
        "complete_entity_recall": 0.6890681003584229,
        "pairs": 1116,
        "matched_entities": 1116
      },
      "profile=singleton": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "multiplicity=0": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "profile=S2_only": {
        "pair_recall": 0.7092313948759659,
        "complete_entity_recall": 0.6331269349845201,
        "pairs": 2459,
        "matched_entities": 1292
      }
    }
  },
  "address_char:50+50": {
    "queries": 20000,
    "true_pairs": 69268,
    "pair_recall": 0.7019114165271121,
    "complete_entity_recall": 0.467479459316194,
    "at_least_one_recall": 0.8659952292605354,
    "candidates": {
      "mean": 58.6776,
      "median": 55.0,
      "p90": 75.0,
      "p95": 75.0,
      "p99": 75.0,
      "max": 75
    },
    "singleton_mean": 58.418502202643175,
    "singleton_p95": 75.0,
    "breakdowns": {
      "country=US": {
        "pair_recall": 0.7145521495863422,
        "complete_entity_recall": 0.49824530619406915,
        "pairs": 41822,
        "matched_entities": 11398
      },
      "profile=both": {
        "pair_recall": 0.7018809464196294,
        "complete_entity_recall": 0.4412769393205391,
        "pairs": 63904,
        "matched_entities": 16101
      },
      "multiplicity=4+": {
        "pair_recall": 0.7026008930649906,
        "complete_entity_recall": 0.39299731460442056,
        "pairs": 47253,
        "matched_entities": 9682
      },
      "short_name": {
        "pair_recall": 0.6956941494497008,
        "complete_entity_recall": 0.4715909090909091,
        "pairs": 10358,
        "matched_entities": 2816
      },
      "source=S2": {
        "pair_recall": 0.7087514934289127,
        "complete_entity_recall": 0.6315759213476686,
        "pairs": 33480,
        "matched_entities": 17393
      },
      "source=S3": {
        "pair_recall": 0.6955124622778585,
        "complete_entity_recall": 0.5970522961361179,
        "pairs": 35788,
        "matched_entities": 17573
      },
      "country=India": {
        "pair_recall": 0.6826495664213365,
        "complete_entity_recall": 0.4205169412079818,
        "pairs": 27446,
        "matched_entities": 7467
      },
      "weak_lexical": {
        "pair_recall": 0.7013898638214235,
        "complete_entity_recall": 0.4364050705027774,
        "pairs": 28492,
        "matched_entities": 7021
      },
      "multiplicity=3": {
        "pair_recall": 0.6992654774396643,
        "complete_entity_recall": 0.491710388247639,
        "pairs": 14295,
        "matched_entities": 4765
      },
      "nonascii_target_name": {
        "pair_recall": 0.6865364850976362,
        "complete_entity_recall": 0.421195652173913,
        "pairs": 25298,
        "matched_entities": 6256
      },
      "multiplicity=2": {
        "pair_recall": 0.7033615990308903,
        "complete_entity_recall": 0.5729860690490611,
        "pairs": 6604,
        "matched_entities": 3302
      },
      "high_repeated_name": {
        "pair_recall": 0.6817740683229814,
        "complete_entity_recall": 0.45038979447200567,
        "pairs": 10304,
        "matched_entities": 2822
      },
      "profile=S3_only": {
        "pair_recall": 0.6925989672977625,
        "complete_entity_recall": 0.6046195652173914,
        "pairs": 2905,
        "matched_entities": 1472
      },
      "missing_target_address": {
        "pair_recall": 0.5466052934407365,
        "complete_entity_recall": 0.0,
        "pairs": 12166,
        "matched_entities": 2845
      },
      "multiplicity=1": {
        "pair_recall": 0.6980286738351255,
        "complete_entity_recall": 0.6980286738351255,
        "pairs": 1116,
        "matched_entities": 1116
      },
      "profile=singleton": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "multiplicity=0": {
        "pair_recall": null,
        "complete_entity_recall": null,
        "pairs": 0,
        "matched_entities": 0
      },
      "profile=S2_only": {
        "pair_recall": 0.7137047580317202,
        "complete_entity_recall": 0.6377708978328174,
        "pairs": 2459,
        "matched_entities": 1292
      }
    }
  }
}

## Remaining misses

The companion missed-positive report lists categories and representative examples. Counts overlap because one pair can have several failure properties.

## Efficiency

Total benchmark runtime: 4.8 minutes. Input index: SQLite disk cache. Generated artifact bytes: 77,785,663. Peak RAM was not reliably available from the host API. Each retrieval channel scans targets in batches/streaming form and keeps only bounded per-query heaps. Gzip TSV/JSON is used because no local Parquet engine is installed; this limitation is explicit and artifacts remain resumable.

## Decision

Blocking does not meet the stated freeze criteria. Next: inspect confirmation misses and add only a measured rescue channel targeted at the dominant class; do not start the matcher.
