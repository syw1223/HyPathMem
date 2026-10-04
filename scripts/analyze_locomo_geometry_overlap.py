"""Audit frozen pre-Card Euclidean/Lorentz evidence overlap without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "outputs/final/hypathmem_locomo_d0_time_v2_final1540/source_artifacts/hypathmem_locomo_pre_relation_card_four_route_union_pool.jsonl"
DEFAULT_GRAPH = ROOT / "outputs/graphs/locomo_graph_v3_6b_qwen_all.json"
STOPWORDS = set("a an the is are was were be been to of in on at for and or with from as by what which when where who how did does do has have had her his their she he they you i it my your kind kinds".split())


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_unique(groups):
    candidates = {}
    for group in groups:
        for candidate in group:
            node_id = candidate["evidence_node_id"]
            previous = candidates.get(node_id)
            if previous is None or candidate["cross_encoder_score"] > previous["cross_encoder_score"]:
                candidates[node_id] = candidate
    return sorted(candidates.values(), key=lambda c: (-c["cross_encoder_score"], c["evidence_node_id"]))


def raw_support(candidates):
    return {turn for c in candidates for turn in c["evidence_ids"]}


def coverage(candidates, gold):
    found = raw_support(candidates) & gold
    return dict(hit=float(bool(found)), recall=len(found) / len(gold) if gold else 0.0,
                full_cover=float(bool(gold) and gold <= found))


def set_metrics(left, right):
    shared, union = left & right, left | right
    return dict(jaccard=len(shared) / len(union) if union else 1.0,
                eu_only_union_share=len(left - right) / len(union) if union else 0.0,
                lor_only_union_share=len(right - left) / len(union) if union else 0.0,
                shared_union_share=len(shared) / len(union) if union else 0.0,
                eu_count=len(left), lor_count=len(right), shared_count=len(shared),
                lor_unique_within_lor=len(right - left) / len(right) if right else 0.0)


def clustered_ci(rows, key, seed=13):
    grouped = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            grouped[row["conversation_id"]].append(row[key])
    sums = np.array([sum(v) for v in grouped.values()])
    counts = np.array([len(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sums), size=(2000, len(sums)))
    estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return [float(x) for x in np.quantile(estimates, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/locomo_geometry_overlap_20261004")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph = json.loads(args.graph.read_text())
    nodes = graph["nodes"]
    parents = defaultdict(set)
    for edge in graph["edges"]:
        if edge["relation"] == "IS_SPECIFIC_OF" and nodes.get(edge["dst"], {}).get("type") in {"EVENT", "TOPIC"}:
            if edge["src"].split(":", 1)[0] == edge["dst"].split(":", 1)[0]:
                parents[edge["src"]].add(edge["dst"])
    ancestor_cache = {}

    def ancestors(node_id):
        if node_id not in ancestor_cache:
            seen, queue = set(), list(parents[node_id])
            while queue:
                current = queue.pop()
                if current in seen:
                    continue
                seen.add(current)
                queue.extend(parents[current] - seen)
            ancestor_cache[node_id] = seen
        return ancestor_cache[node_id]

    comparison_specs = [
        ("BU", ("BU_E",), ("BU_H",), 20),
        ("BU", ("BU_E",), ("BU_H",), 50),
        ("TD", ("TD_E",), ("TD_H",), 5),
        ("Geometry_union", ("BU_E", "TD_E"), ("BU_H", "TD_H"), 20),
        ("Geometry_union", ("BU_E", "TD_E"), ("BU_H", "TD_H"), 50),
    ]
    records = defaultdict(list)
    characterization = defaultdict(lambda: defaultdict(list))
    examples = []
    route_sizes = defaultdict(Counter)
    categories, conversations = Counter(), Counter()
    candidate_types = Counter()
    missing_nodes, duplicate_route_ids = Counter(), Counter()
    n = 0
    audit_file = args.output / "per_question.jsonl"
    with args.pool.open() as source, audit_file.open("w") as output:
        for line in source:
            item = json.loads(line)
            if int(item["category"]) not in {1, 2, 3, 4}:
                continue
            n += 1
            qid, question = item["question_id"], item["question"]
            conv = qid.split(":q", 1)[0]
            categories[item["category"]] += 1
            conversations[conv] += 1
            gold = set(item["gold_evidence"])
            route_candidates = {}
            for route, payload in item["route_pools"].items():
                candidates = payload["candidates"]
                assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
                route_sizes[route][len(candidates)] += 1
                seen = set()
                for c in candidates:
                    fid = c["evidence_node_id"]
                    if fid in seen:
                        duplicate_route_ids[route] += 1
                    else:
                        seen.add(fid)
                    if fid not in nodes:
                        missing_nodes[fid] += 1
                    else:
                        candidate_types[nodes[fid]["type"]] += 1
                # Stored route order is preserved; fusion alone uses stored CE.
                route_candidates[route] = list({c["evidence_node_id"]: c for c in reversed(candidates)}.values())[::-1]
            for direction, eu_routes, lor_routes, k in comparison_specs:
                eu = route_candidates[eu_routes[0]] if len(eu_routes) == 1 else ranked_unique([route_candidates[r] for r in eu_routes])
                lor = route_candidates[lor_routes[0]] if len(lor_routes) == 1 else ranked_unique([route_candidates[r] for r in lor_routes])
                eu, lor = eu[:k], lor[:k]
                eids, hids = {c["evidence_node_id"] for c in eu}, {c["evidence_node_id"] for c in lor}
                eraw, hraw = raw_support(eu), raw_support(lor)
                additional_gold = sorted((hraw - eraw) & gold)
                fixed = ranked_unique([eu, lor])[:k]
                em, hm, fm = coverage(eu, gold), coverage(lor, gold), coverage(fixed, gold)
                r = dict(question_id=qid, conversation_id=conv, category=item["category"], comparison=direction, k=k,
                         **set_metrics(eids, hids), raw_jaccard=set_metrics(eraw, hraw)["jaccard"],
                         additional_gold_count=len(additional_gold), additional_gold_query=float(bool(additional_gold)),
                         eu_hit=em["hit"], lor_hit=hm["hit"], fixed_union_hit=fm["hit"],
                         eu_recall=em["recall"], lor_recall=hm["recall"], fixed_union_recall=fm["recall"],
                         eu_full_cover=em["full_cover"], lor_full_cover=hm["full_cover"], fixed_union_full_cover=fm["full_cover"],
                         fixed_union_recall_delta=fm["recall"]-em["recall"],
                         expanded_union_recall_delta=len(additional_gold)/len(gold) if gold else None,
                         rescued_eu_hit=float(not em["hit"] and hm["hit"]),
                         eu_budget_short=float(len(eu) < k), lor_budget_short=float(len(lor) < k))
                if not gold:
                    for field in ["additional_gold_count", "additional_gold_query", "eu_hit", "lor_hit", "fixed_union_hit", "eu_recall", "lor_recall", "fixed_union_recall", "eu_full_cover", "lor_full_cover", "fixed_union_full_cover", "fixed_union_recall_delta", "rescued_eu_hit"]:
                        r[field] = None
                key = f"{direction}@{k}"
                records[key].append(r)
                eu_ancestors = set().union(*(ancestors(fid) for fid in eids)) if eids else set()
                unique_rows = []
                for lor_rank, c in enumerate(lor, start=1):
                    fid = c["evidence_node_id"]
                    if fid in eids:
                        continue
                    node = nodes.get(fid, {})
                    qtokens, ftokens = tokens(question), tokens(node.get("text", ""))
                    raw_new = set(c["evidence_ids"]) - eraw
                    gold_new = raw_new & gold
                    detail = dict(fact_id=fid, rank=lor_rank, source_route_rank=c["rank"], text=node.get("text", ""),
                                  evidence_ids=c["evidence_ids"], novel_raw_turns=sorted(raw_new),
                                  additional_gold=sorted(gold_new),
                                  ce_score=c["cross_encoder_score"],
                                  lexical_query_coverage=len(qtokens & ftokens)/len(qtokens) if qtokens else 0.,
                                  shares_hierarchy_ancestor_with_eu=bool(ancestors(fid) & eu_ancestors),
                                  has_hierarchy_parent=bool(ancestors(fid)),
                                  path_node_ids=c["path_node_ids"],
                                  path_labels=[nodes.get(pid, {}).get("text", pid) for pid in c["path_node_ids"]],
                                  candidate_type=node.get("type", "MISSING"))
                    unique_rows.append(detail)
                    for group in ("all_lor_only", "gold_novel_lor_only" if gold_new else "other_lor_only"):
                        characterization[key][group].append(detail)
                    if direction == "Geometry_union" and k == 20 and gold_new:
                        examples.append(dict(question_id=qid,question=question,category=item["category"],**detail))
                output.write(json.dumps(dict(**r,question=question,gold_evidence=sorted(gold),
                                             eu_fact_ids=[c["evidence_node_id"] for c in eu],
                                             lor_fact_ids=[c["evidence_node_id"] for c in lor],
                                             additional_gold=additional_gold,lor_only_facts=unique_rows),ensure_ascii=False)+"\n")
            if n % 250 == 0:
                print(f"Analyzed {n} questions",flush=True)
    assert n == 1540, f"Expected 1540 Cat1-4 questions, got {n}"
    assert not missing_nodes, f"Missing graph nodes: {list(missing_nodes)[:5]}"
    assert not duplicate_route_ids, f"Duplicate Fact IDs: {duplicate_route_ids}"
    summaries = {}
    for key, rows in records.items():
        summary = {field:mean(r[field] for r in rows if r[field] is not None) for field in rows[0] if isinstance(rows[0][field], (float,int)) and field not in {"category","k"}}
        summary.update(num_questions=len(rows),k=rows[0]["k"],
                       num_gold_annotated=sum(r["eu_recall"] is not None for r in rows),
                       jaccard_ci95=clustered_ci(rows,"jaccard"),
                       fixed_union_recall_delta_ci95=clustered_ci(rows,"fixed_union_recall_delta"),
                       additional_gold_queries=sum(int(r["additional_gold_query"] or 0) for r in rows),
                       rescued_eu_hit_queries=sum(int(r["rescued_eu_hit"] or 0) for r in rows))
        summary["by_category"] = {str(c): {metric:(mean(values) if (values:=[r[metric] for r in rows if r["category"]==c and r[metric] is not None]) else None) for metric in ["jaccard","raw_jaccard","additional_gold_query","eu_recall","lor_recall","fixed_union_recall"]} for c in categories}
        summary["lor_only_characterization"] = {}
        for group, values in characterization[key].items():
            summary["lor_only_characterization"][group] = dict(fact_query_occurrences=len(values),
                ce_score_mean=mean(v["ce_score"] for v in values),
                lexical_query_coverage_mean=mean(v["lexical_query_coverage"] for v in values),
                zero_lexical_overlap_fraction=mean(float(v["lexical_query_coverage"]==0) for v in values),
                hierarchy_parent_fraction=mean(float(v["has_hierarchy_parent"]) for v in values),
                shared_ancestor_with_eu_fraction=mean(float(v["shares_hierarchy_ancestor_with_eu"]) for v in values),
                novel_raw_turn_fraction=mean(float(bool(v["novel_raw_turns"])) for v in values),
                fact_types=dict(Counter(v["candidate_type"] for v in values)))
        summaries[key] = summary
    payload = dict(protocol=dict(stage="pre_relation_card_pre_selector",questions=n,categories=dict(categories),
                                conversations=dict(conversations),seed=13,cluster_bootstrap_replicates=2000,
                                route_candidate_lengths={r:dict(c) for r,c in route_sizes.items()},
                                ranking="stored route order; geometry unions deduplicated by Fact ID and ranked by existing CE score; ID tie-break",
                                scope="Frozen deployed route complementarity; BU-E and BU-H also differ in seed policy and hierarchy version. Not a pure metric-only controlled ablation.",
                                td_h_limitation="All 1540 stored TD-H route lists contain 5 candidates; TD reported at K=5 only.",
                                fact_overlap="Jaccard of target Fact IDs, not all intermediate path nodes",
                                raw_overlap="Jaccard of source turn IDs associated with target Facts, within each question",
                                uniqueness_denominator="Eu-only/Lor-only/shared percentages divide by set union; query-macro averaged",
                                gold_scope="Source-turn annotation overlap; not independent verification of each Fact claim",
                                semantic_scope="CE relevance and lexical overlap are diagnostics, not embedding cosine similarity",
                                union_budget="Expanded union has up to 2K Facts; fixed union reranks the two Top-K lists to K using stored CE",
                                candidate_type_counts=dict(candidate_types)),results=summaries)
    (args.output/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    fields=["comparison","k","num_questions","num_gold_annotated","jaccard","raw_jaccard","eu_only_union_share","lor_only_union_share","shared_union_share","lor_unique_within_lor","additional_gold_queries","eu_recall","lor_recall","fixed_union_recall","fixed_union_recall_delta","expanded_union_recall_delta"]
    with (args.output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for key,s in summaries.items():writer.writerow({f:(key.split("@")[0] if f=="comparison" else s[f]) for f in fields})
    examples.sort(key=lambda e:(e["lexical_query_coverage"],e["question_id"],e["rank"]))
    (args.output/"lorentz_only_gold_examples.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    manifest=dict(input_pool=str(args.pool),input_graph=str(args.graph),script=str(Path(__file__).resolve()),
                  sha256={"pool":sha256(args.pool),"graph":sha256(args.graph),"script":sha256(Path(__file__))},
                  inference_calls=0,gpu_required=False,retraining=False)
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    report=["# LoCoMo Evidence Overlap Analysis","",f"Frozen pre-Card route outputs; {n} Cat1-4 questions in {len(conversations)} conversations. No retraining, LLM calls, or GPU inference.","", "## Fact and Source-Turn Overlap","", "| Comparison | K | Fact Jaccard | Raw-turn Jaccard | Eu-only / union | Lor-only / union | Shared / union | Questions with additional Lorentz gold |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,s in summaries.items():
        report.append(f"| {key.split('@')[0]} | {s['k']} | {s['jaccard']:.4f} | {s['raw_jaccard']:.4f} | {100*s['eu_only_union_share']:.2f}% | {100*s['lor_only_union_share']:.2f}% | {100*s['shared_union_share']:.2f}% | {s['additional_gold_queries']}/{s['num_gold_annotated']} |")
    report += ["", "Geometry_union compares BU-E + TD-E against BU-H + TD-H; each group uses stored CE scores before taking K. BU preserves route-local order. TD-H is limited to five saved candidates, so only TD@5 is valid. Geometry_union is a deployed-route diagnostic and inherits this asymmetric TD budget.","", "Jaccard = |E intersection H| / |E union H|. All percentages are macro averages over questions. The three union shares sum to 100%. Fact identity can distinguish paraphrases from the same source turn, so raw-turn Jaccard is reported separately. Questions without gold supporting turns remain in overlap analysis and are excluded only from gold-coverage denominators.","", "## Added Evidence and Fixed Budget","", "| Comparison | Eu Recall | Lor Recall | Joint CE Recall (K fixed) | Delta vs Eu (pp) | 95% cluster CI (pp) | Added recall, union up to 2K (pp) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key,s in summaries.items():
        lo,hi=s['fixed_union_recall_delta_ci95']
        report.append(f"| {key} | {100*s['eu_recall']:.2f} | {100*s['lor_recall']:.2f} | {100*s['fixed_union_recall']:.2f} | {100*s['fixed_union_recall_delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {100*s['expanded_union_recall_delta']:+.2f} |")
    report += ["", "Confidence intervals use 2,000 paired conversation-cluster bootstrap resamples (seed 13, 10 conversation clusters). Expanded-union coverage has a larger budget and must not be presented as a same-budget accuracy gain. These are retrieval/provenance metrics, not QA accuracy.","", "## Lorentz-only Characterization","", "| Comparison | Subset | Fact-query occurrences | Query lexical coverage | Zero lexical overlap | Has Event/Topic ancestor | Shares ancestor with Eu | Novel raw support |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key,s in summaries.items():
        for group,v in s['lor_only_characterization'].items():
            if group=='other_lor_only':continue
            report.append(f"| {key} | {group} | {v['fact_query_occurrences']} | {100*v['lexical_query_coverage_mean']:.2f}% | {100*v['zero_lexical_overlap_fraction']:.2f}% | {100*v['hierarchy_parent_fraction']:.2f}% | {100*v['shared_ancestor_with_eu_fraction']:.2f}% | {100*v['novel_raw_turn_fraction']:.2f}% |")
    report += ["", "All target evidence nodes are Facts. This analysis cannot show that Lorentz-only results are higher-level nodes. Ancestor measures refer to same-conversation IS_SPECIFIC_OF paths to Event/Topic nodes in the frozen graph. Lexical query coverage uses unique lowercase alphanumeric non-stopwords; it is not a semantic similarity measure. Shared ancestors show structural association, not causal attribution to hyperbolic geometry.","", "## Auditable Examples","", "Examples below are selected for low lexical overlap and additional gold-turn coverage, not for QA outcome."]
    chosen=[];seen=set()
    for example in examples:
        if example['question_id'] in seen:continue
        seen.add(example['question_id']);chosen.append(example)
        if len(chosen)==5:break
    for e in chosen:
        report += ["",f"### {e['question_id']}","",f"Question: {e['question']}","",f"Lorentz-only Fact (rank {e['rank']}): {e['text']}","",f"Additional gold turns: {', '.join(e['additional_gold'])}. Lexical query coverage: {100*e['lexical_query_coverage']:.1f}%.","", "Saved path labels: " + " -> ".join(e['path_labels'])]
    report += ["", "## Reproduce", "", "```bash", "CUDA_VISIBLE_DEVICES='' /home/anonymous/miniconda3/envs/python311/bin/python scripts/analyze_locomo_geometry_overlap.py", "```", "", "Input hashes and exact paths are in manifest.json; per_question.jsonl records every set difference and novel gold turn. Existing method code/results are not modified."]
    (args.output/"REPORT.md").write_text("\n".join(report)+"\n")
    main20=summaries['Geometry_union@20']
    main50=summaries['Geometry_union@50']
    chars=main20['lor_only_characterization']['gold_novel_lor_only']
    paper=["# Minimal Evidence Overlap Experiment", "", "LoCoMo Cat1-4, 1,540 questions. E = BU-E + TD-E; H = BU-H + TD-H. Both geometry unions are deduplicated and sorted by frozen CE relevance scores before selecting K.", "", "| K | Fact Jaccard | Eu-only (%) | Lorentz-only (%) | Shared (%) | Raw-turn Jaccard | Extra-gold questions |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in [main20,main50]:
        paper.append(f"| {s['k']} | {s['jaccard']:.4f} | {100*s['eu_only_union_share']:.2f} | {100*s['lor_only_union_share']:.2f} | {100*s['shared_union_share']:.2f} | {s['raw_jaccard']:.4f} | {s['additional_gold_queries']} |")
    paper += ["", "Eu-only, Lorentz-only, and shared use the union as denominator and are macro-averaged across queries. Gold-coverage statistics use 1,536 questions with nonempty supporting-turn annotations; four questions remain in overlap analysis only. At K=50, seven H lists contain fewer than 50 Facts (mean H size 49.91).", "", "## Evidence of Utility", "",f"At K=20, adding Lorentz-route candidates and reranking to the same 20-Fact budget raises source-turn Recall from {100*main20['eu_recall']:.2f}% to {100*main20['fixed_union_recall']:.2f}% ({100*main20['fixed_union_recall_delta']:+.2f} pp; 95% conversation-cluster bootstrap CI [{100*main20['fixed_union_recall_delta_ci95'][0]:+.2f}, {100*main20['fixed_union_recall_delta_ci95'][1]:+.2f}] pp). Lorentz contributes additional annotated turns for {main20['additional_gold_queries']}/1,536 questions. Among {chars['fact_query_occurrences']} Lorentz-only Fact-query occurrences adding novel gold support, {100*chars['shared_ancestor_with_eu_fraction']:.2f}% share an Event/Topic ancestor with Euclidean-selected Facts.", "", "These results support complementary evidence coverage in the deployed routes, especially for multi-hop questions. They do not establish that Lorentz retrieval is stronger in isolation or that all novel candidates are useful. All retrieved targets are Facts, so describe them as hierarchy-associated Facts rather than higher-level evidence nodes. Weak lexical overlap is measured in REPORT.md; weak embedding-semantic similarity is not established by this experiment.","", "## Scope of the Comparison", "", "The saved TD-H file has only five Facts per query. Therefore TD-E vs TD-H is additionally compared at K=5, and the combined-geometry result inherits the actual asymmetric TD budget. BU-E and BU-H differ in seed policy and hierarchy version as well as geometry. Use the title 'Evidence Complementarity of Euclidean and Lorentz Retrieval Routes'; do not present this as an isolated metric-only intervention.","", "## Suggested Paper Text", "",f"We analyze evidence overlap between the deployed Euclidean and Lorentz retrieval routes using frozen pre-composition candidates. Their mean Fact-level Jaccard overlap is {main20['jaccard']:.4f} at K=20 and {main50['jaccard']:.4f} at K=50. At K=20, Lorentz routes supply additional annotated supporting turns for {main20['additional_gold_queries']} queries, and combining both route families with the existing relevance scorer at the same final budget improves retrieval recall by {100*main20['fixed_union_recall_delta']:.2f} percentage points. These results indicate complementary coverage rather than uniform superiority of either geometry. The comparison reflects the deployed route configurations, including the saved five-candidate TD-H output.",""]
    (args.output/'PAPER_TABLE.md').write_text('\n'.join(paper))
    print(json.dumps({k:{f:s[f] for f in ['jaccard','raw_jaccard','lor_only_union_share','additional_gold_queries','fixed_union_recall_delta']} for k,s in summaries.items()},indent=2),flush=True)
    print(f"Wrote analysis to {args.output}",flush=True)


if __name__ == "__main__":
    main()
