"""Conditional random controls for frozen Lorentz-only gold ancestor overlap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/analysis/locomo_geometry_overlap_20261004")
    parser.add_argument("--graph", type=Path, default=ROOT / "outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/locomo_ancestor_random_control_20261004")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph = json.loads(args.graph.read_text())
    nodes = graph["nodes"]
    parents = defaultdict(set)
    for edge in graph["edges"]:
        src, dst = edge["src"], edge["dst"]
        if edge["relation"] == "IS_SPECIFIC_OF" and nodes.get(dst, {}).get("type") in {"EVENT", "TOPIC"}:
            if src.split(":", 1)[0] == dst.split(":", 1)[0]:
                parents[src].add(dst)
    ancestor_cache = {}

    def ancestors(fid):
        if fid not in ancestor_cache:
            seen, queue = set(), list(parents[fid])
            while queue:
                current = queue.pop()
                if current not in seen:
                    seen.add(current)
                    queue.extend(parents[current] - seen)
            ancestor_cache[fid] = seen
        return ancestor_cache[fid]

    facts_by_conv = defaultdict(list)
    for fid, node in nodes.items():
        if node["type"] == "FACT":
            facts_by_conv[fid.split(":", 1)[0]].append(fid)
    for fids in facts_by_conv.values():
        fids.sort()

    input_path = args.analysis_dir / "per_question.jsonl"
    all_questions, selected_questions = [], []
    with input_path.open() as stream:
        for line in stream:
            q = json.loads(line)
            if q["comparison"] != "Geometry_union" or q["k"] != 20:
                continue
            all_questions.append(q)
            q["targets"] = [f for f in q["lor_only_facts"] if f["additional_gold"]]
            if q["targets"]:
                selected_questions.append(q)
    assert len(all_questions) == 1540
    assert len(selected_questions) == 80
    total_targets = sum(len(q["targets"]) for q in selected_questions)
    assert total_targets == 95

    control_names = ["Random_same_conversation", "Random_hierarchy_attached", "Random_source_ancestor_degree_matched"]
    levels = ["Event_or_Topic", "Event_only", "Topic_only"]
    rng = np.random.default_rng(args.seed)
    totals = {name: {level: np.zeros(args.replicates, dtype=int) for level in levels} for name in control_names}
    observations = Counter()
    query_rows, audits = [], []

    for q in selected_questions:
        qid, conv = q["question_id"], q["conversation_id"]
        excluded = set(q["eu_fact_ids"])
        reference = set().union(*(ancestors(fid) for fid in excluded))
        reference_by_level = {
            "Event_or_Topic": reference,
            "Event_only": {a for a in reference if nodes[a]["type"] == "EVENT"},
            "Topic_only": {a for a in reference if nodes[a]["type"] == "TOPIC"},
        }

        def flags(fid):
            return {level: bool(ancestors(fid) & ref) for level, ref in reference_by_level.items()}

        target_ids = [f["fact_id"] for f in q["targets"]]
        target_flags = {fid: flags(fid) for fid in target_ids}
        for fact in q["targets"]:
            assert target_flags[fact["fact_id"]]["Event_or_Topic"] == fact["shares_hierarchy_ancestor_with_eu"]
        observed = {level: sum(target_flags[fid][level] for fid in target_ids) for level in levels}
        observations.update(observed)
        base_pool = [fid for fid in facts_by_conv[conv] if fid not in excluded]

        def match_key(fid):
            asc = ancestors(fid)
            return (nodes[fid].get("source", ""), sum(nodes[a]["type"] == "EVENT" for a in asc),
                    sum(nodes[a]["type"] == "TOPIC" for a in asc))

        strata = defaultdict(list)
        for fid in base_pool:
            strata[match_key(fid)].append(fid)
        target_strata = Counter(match_key(fid) for fid in target_ids)
        control_groups = {
            "Random_same_conversation": [(base_pool, len(target_ids), "all FACT nodes")],
            "Random_hierarchy_attached": [([fid for fid in base_pool if ancestors(fid)], len(target_ids), "has Event/Topic ancestor")],
            "Random_source_ancestor_degree_matched": [(strata[key], number, str(key)) for key, number in sorted(target_strata.items())],
        }
        row = dict(question_id=qid,conversation_id=conv,category=q["category"],target_count=len(target_ids),
                   observed_counts=observed,controls={})
        audit = dict(question_id=qid,target_fact_ids=target_ids,eu_fact_ids=q["eu_fact_ids"],
                     target_flags=target_flags,controls={})
        for name, groups in control_groups.items():
            expected = {level: 0.0 for level in levels}
            group_audits = []
            for pool, number, label in groups:
                assert len(pool) >= number, (qid,name,label,len(pool),number)
                assert not set(pool) & excluded
                pool_flags = {fid: flags(fid) for fid in pool}
                positive = {level: sum(pool_flags[fid][level] for fid in pool) for level in levels}
                for level in levels:
                    expected[level] += number * positive[level] / len(pool)
                    # Exact count distribution of uniform sampling without replacement.
                    totals[name][level] += rng.hypergeometric(positive[level],len(pool)-positive[level],number,size=args.replicates)
                sample_rng = np.random.default_rng(np.random.SeedSequence([args.seed, len(audits), control_names.index(name), len(group_audits)]))
                sample_ids = [str(x) for x in sample_rng.choice(pool,size=number,replace=False)]
                group_audits.append(dict(label=label,sample_count=number,pool_size=len(pool),overlap_counts=positive,
                                        eligible_fact_ids=pool,example_draw=sample_ids,
                                        example_draw_flags={fid:pool_flags[fid] for fid in sample_ids}))
            row["controls"][name] = dict(expected_counts=expected,strata=[{k:g[k] for k in ["label","sample_count","pool_size","overlap_counts"]} for g in group_audits])
            audit["controls"][name] = group_audits
        query_rows.append(row)
        audits.append(audit)

    assert observations["Event_or_Topic"] == 67
    summaries = {}
    convs = sorted({q["conversation_id"] for q in query_rows})
    denominators = np.array([sum(q["target_count"] for q in query_rows if q["conversation_id"] == conv) for conv in convs])
    boot_rng = np.random.default_rng(args.seed + 1)
    boot_indices = boot_rng.integers(0,len(convs),size=(args.replicates,len(convs)))
    for name in control_names:
        summaries[name] = {}
        for level in levels:
            expected_count = sum(q["controls"][name]["expected_counts"][level] for q in query_rows)
            random_rates = totals[name][level] / total_targets
            observed_rate = observations[level] / total_targets
            random_expected_rate = expected_count / total_targets
            clustered_diffs = np.array([sum(q["observed_counts"][level]-q["controls"][name]["expected_counts"][level]
                                          for q in query_rows if q["conversation_id"] == conv) for conv in convs])
            boot_diffs = clustered_diffs[boot_indices].sum(axis=1)/denominators[boot_indices].sum(axis=1)
            summaries[name][level] = dict(observed_rate=observed_rate,random_expected_rate=random_expected_rate,
                random_mc_mean=float(random_rates.mean()),random_null_interval95=[float(x) for x in np.quantile(random_rates,[.025,.975])],
                delta=observed_rate-random_expected_rate,
                delta_cluster_bootstrap_ci95=[float(x) for x in np.quantile(boot_diffs,[.025,.975])],
                one_sided_conditional_randomization_p=(1+int((totals[name][level]>=observations[level]).sum()))/(args.replicates+1))

    result = dict(protocol=dict(comparison="Geometry_union@20",num_locomo_questions=1540,
        affected_queries=len(query_rows),fact_query_pairs=total_targets,conversation_clusters=len(convs),
        observed_shared_ancestor_pairs=observations["Event_or_Topic"],random_replicates=args.replicates,seed=args.seed,
        metric="fraction of target Fact-query pairs sharing at least one Event/Topic ancestor with Euclidean Top20",
        weighting="micro-average over 95 Fact-query pairs; per-query sample count exactly matches target count",
        random_pool="same conversation FACT nodes, excluding Euclidean Top20 Fact IDs; actual Lorentz targets remain eligible",
        controls=control_names,
        hierarchy="same-conversation IS_SPECIFIC_OF edges to EVENT/TOPIC nodes, identical to preceding overlap analysis",
        event_type_scope="EVENT includes episode-like nodes encoded as EVENT in this graph schema",
        random_generation="hypergeometric draws: exact success-count law of uniform within-query/stratum sampling without replacement; explicit example draws and eligible IDs saved",
        intervals="null interval quantifies random draws; delta CI uses paired conversation-cluster bootstrap against exact null expectations",
        inference_scope="observational structural enrichment of gold-turn-selected Lorentz-only Facts; not causal proof of hierarchy contribution; controls are not conditioned on gold relevance",
        gpu_required=False,llm_calls=0,retraining=False),results=summaries)
    (args.output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (args.output/"per_question.jsonl").open("w") as stream:
        for row in query_rows:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.output/"sampling_audit.json").write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    np.savez_compressed(args.output/"random_null_counts.npz",**{name+'__'+level:counts for name,v in totals.items() for level,counts in v.items()})
    with (args.output/"table.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["control","level","observed_pct","random_expected_pct","delta_pp","delta_cluster_ci_low_pp","delta_cluster_ci_high_pp","one_sided_p"])
        for name,values in summaries.items():
            for level,s in values.items():writer.writerow([name,level,100*s['observed_rate'],100*s['random_expected_rate'],100*s['delta'],*[100*x for x in s['delta_cluster_bootstrap_ci95']],s['one_sided_conditional_randomization_p']])
    manifest=dict(graph=str(args.graph),overlap_audit=str(input_path),script=str(Path(__file__).resolve()),
                  sha256=dict(graph=sha256(args.graph),overlap_audit=sha256(input_path),script=sha256(Path(__file__))))
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    report=["# Random-Control Ancestor Overlap Analysis","", "The analysis reuses the preceding Geometry_union@20 result: 95 Lorentz-only Fact-query pairs supplying novel gold support, across 80 questions and 10 conversations. The observed Event/Topic overlap is 67/95 = 70.53%.","", "Each random draw samples the same number of Facts per query. Sampling is within the same conversation and excludes Euclidean Top20 Fact IDs. Facts are sampled without replacement within each draw. Actual Lorentz-only targets remain eligible, keeping the null population inclusive.","", "## Main Table", "", "| Random control | Expected overlap | Lorentz-only gold overlap | Difference (pp) | 95% paired cluster CI (pp) | Conditional randomization p |", "|---|---:|---:|---:|---:|---:|"]
    for name,values in summaries.items():
        s=values['Event_or_Topic'];lo,hi=s['delta_cluster_bootstrap_ci95']
        report.append(f"| {name} | {100*s['random_expected_rate']:.2f}% | {100*s['observed_rate']:.2f}% | {100*s['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] | {s['one_sided_conditional_randomization_p']:.5f} |")
    report += ["", "The first null samples any Fact; the second samples only hierarchy-attached Facts; the third matches Fact source and the exact numbers of Event-type and Topic ancestors. The matched control preserves the hierarchy-connectivity profile of the 95 observed pairs.", "", "## Ancestor-Level Breakdown", "", "| Control | Ancestor level | Random expectation | Observed |", "|---|---|---:|---:|"]
    for name,values in summaries.items():
        for level,s in values.items():report.append(f"| {name} | {level} | {100*s['random_expected_rate']:.2f}% | {100*s['observed_rate']:.2f}% |")
    report += ["", "## Interpretation", "", "Use the matched control when assessing enrichment. A higher observed rate shows that added gold-supported Lorentz evidence is concentrated in branches related to Euclidean-selected evidence beyond generic hierarchy attachment/source/degree effects. It does not establish causality: the Lorentz subset is selected for gold support, whereas random controls are not matched on gold relevance or semantic relevance. A hierarchy-removal intervention would be needed to claim that hierarchy itself caused the gain.","", "Both Event and Topic measures are included because sharing a broad Topic is easier than sharing a finer Event. The graph represents some episode-like nodes with the EVENT type; Event_only follows this stored schema. This protocol matches the previous 70.53% measure exactly.","", "## Statistical Details", "", f"Uniform fixed-size sampling success counts are drawn from their exact hypergeometric distributions for {args.replicates:,} repetitions, seed {args.seed}. Results include analytical null expectations, Monte Carlo means, and null 95% intervals. A separate paired bootstrap resamples the {len(convs)} conversation clusters and preserves per-query sample counts. P-values are one-sided conditional descriptive tests; all three controls and level breakdowns are exploratory.","", "## Files", "", "summary.json: full protocol, exact expected rates and confidence intervals. table.csv: table values. per_question.jsonl: query-level observed and expected counts. sampling_audit.json: every eligible Fact ID and explicit example draws. random_null_counts.npz: all simulated success counts. manifest.json: input and code hashes.","", "## Reproduce", "", "```bash", "CUDA_VISIBLE_DEVICES='' /home/anonymous/miniconda3/envs/python311/bin/python scripts/analyze_locomo_ancestor_random_control.py", "```", ""]
    (args.output/"REPORT.md").write_text("\n".join(report))
    print(json.dumps({name:values['Event_or_Topic'] for name,values in summaries.items()},indent=2),flush=True)
    print(f"Saved random-control analysis: {args.output}")


if __name__ == "__main__":
    main()
