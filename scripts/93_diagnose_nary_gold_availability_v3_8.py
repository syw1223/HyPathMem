from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import MemoryGraph, Node, NodeType


VARIANT_SPECS = {
    "seed_top10": {"seed_topn": 10, "needed_roles": False, "pair_topn": 0, "card_topn": 0},
    "seed_top20": {"seed_topn": 20, "needed_roles": False, "pair_topn": 0, "card_topn": 0},
    "seed_top50": {"seed_topn": 50, "needed_roles": False, "pair_topn": 0, "card_topn": 0},
    "seed20_needed_roles": {"seed_topn": 20, "needed_roles": True, "pair_topn": 0, "card_topn": 0},
    "seed20_pairce3": {"seed_topn": 20, "needed_roles": True, "pair_topn": 3, "card_topn": 0},
    "seed20_pairce5": {"seed_topn": 20, "needed_roles": True, "pair_topn": 5, "card_topn": 0},
    "seed20_pairce10": {"seed_topn": 20, "needed_roles": True, "pair_topn": 10, "card_topn": 0},
    "seed20_card5_pairce10": {"seed_topn": 20, "needed_roles": True, "pair_topn": 10, "card_topn": 5},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--base-paths", default="outputs/nary_v3_6c_selector/base100_paths.json")
    parser.add_argument(
        "--raw-completion-paths",
        default="outputs/nary_v3_6c_selector/qwen_all_base100_completion50_paths.json",
    )
    parser.add_argument(
        "--variant-name",
        default="seed20_card5_pairce10",
        choices=sorted(VARIANT_SPECS),
    )
    parser.add_argument(
        "--variant-paths",
        default="outputs/nary_v3_8_query_conditioned/seed20_card5_pairce10_paths.json",
    )
    parser.add_argument(
        "--selector-cv-dir",
        default="outputs/eval/cv/v3_8_selector_seed20_card5_pairce10_top20",
    )
    parser.add_argument("--selector-method", default="base_completion_nary_point_features")
    parser.add_argument("--selector-topk", type=int, default=20)
    parser.add_argument("--example-limit", type=int, default=30)
    parser.add_argument("--output-json", default="outputs/eval/NARY_V3_8_MISSING_GOLD_ATTRIBUTION.json")
    parser.add_argument("--output-md", default="outputs/eval/NARY_V3_8_MISSING_GOLD_ATTRIBUTION.md")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    nary_index = NaryIndex.from_graph(graph)
    evidence_to_facts = build_evidence_to_facts(graph)

    base_items = keyed_items(read_json(resolve_path(args.base_paths)))
    raw_items = keyed_items(read_json(resolve_path(args.raw_completion_paths)))
    variant_items = keyed_items(read_json(resolve_path(args.variant_paths)))
    selected_items = keyed_items(load_fold_paths(resolve_path(args.selector_cv_dir), args.selector_method))

    spec = VARIANT_SPECS[args.variant_name]
    rows = []
    rescued = Counter()
    for qid, base_item in base_items.items():
        raw_item = raw_items.get(qid, base_item)
        variant_item = variant_items.get(qid, raw_item)
        selected_item = selected_items.get(qid, variant_item)

        gold = gold_set(base_item)
        if not gold:
            continue
        base_paths = [p for p in base_item.get("paths", []) if not is_completion(p)]
        base_evidence = selected_evidence(graph, base_paths)
        missing_gold = sorted(gold - base_evidence)
        if not missing_gold:
            continue

        base_fact_ids = {evidence_node_id(path) for path in base_paths if evidence_node_id(path)}
        raw_completion = [p for p in raw_item.get("paths", []) if is_completion(p)]
        variant_completion = [p for p in variant_item.get("paths", []) if is_completion(p)]
        selected_paths = list(selected_item.get("paths", []))[: args.selector_topk]

        raw_by_fact = paths_by_fact(raw_completion)
        variant_by_fact = paths_by_fact(variant_completion)
        selected_by_fact = paths_by_fact(selected_paths)
        raw_completion_facts = set(raw_by_fact)
        variant_completion_facts = set(variant_by_fact)
        selected_facts = set(selected_by_fact)
        selected_evid = selected_evidence(graph, selected_paths)

        for gold_eid in missing_gold:
            gold_fact_ids = sorted(evidence_to_facts.get(gold_eid, set()))
            row = attribute_missing_gold(
                graph=graph,
                nary_index=nary_index,
                qid=qid,
                item=base_item,
                gold_eid=gold_eid,
                gold_fact_ids=gold_fact_ids,
                base_fact_ids=base_fact_ids,
                raw_completion_facts=raw_completion_facts,
                variant_completion_facts=variant_completion_facts,
                selected_facts=selected_facts,
                selected_evidence_ids=selected_evid,
                raw_by_fact=raw_by_fact,
                variant_by_fact=variant_by_fact,
                spec=spec,
            )
            rows.append(row)
            if row["selector_rescued"]:
                rescued["selector_rescued"] += 1
            elif row["variant_pool_rescued"]:
                rescued["variant_pool_available_not_selected"] += 1

    payload = summarize(rows, rescued, args, nary_index)
    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


class NaryIndex:
    def __init__(self) -> None:
        self.hyperedges: dict[str, Node] = {}
        self.fact_to_hyperedges: dict[str, set[str]] = defaultdict(set)
        self.fact_role_by_hyperedge: dict[tuple[str, str], str] = {}
        self.fact_ids_by_hyperedge: dict[str, list[str]] = {}

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "NaryIndex":
        index = cls()
        for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
            if node.metadata.get("hierarchy_v3_6") != "typed_nary_hyperedge":
                continue
            index.hyperedges[node.node_id] = node
            role_map = fact_roles(node)
            index.fact_ids_by_hyperedge[node.node_id] = list(role_map)
            for fact_id, role in role_map.items():
                index.fact_to_hyperedges[fact_id].add(node.node_id)
                index.fact_role_by_hyperedge[(node.node_id, fact_id)] = role
        return index


def attribute_missing_gold(
    *,
    graph: MemoryGraph,
    nary_index: NaryIndex,
    qid: str,
    item: dict,
    gold_eid: str,
    gold_fact_ids: list[str],
    base_fact_ids: set[str],
    raw_completion_facts: set[str],
    variant_completion_facts: set[str],
    selected_facts: set[str],
    selected_evidence_ids: set[str],
    raw_by_fact: dict[str, list[dict]],
    variant_by_fact: dict[str, list[dict]],
    spec: dict,
) -> dict:
    fact_hyperedges = {
        fact_id: sorted(nary_index.fact_to_hyperedges.get(fact_id, set()))
        for fact_id in gold_fact_ids
    }
    candidate_hyperedges = sorted({hid for hids in fact_hyperedges.values() for hid in hids})
    triggered = triggered_hyperedges(nary_index, candidate_hyperedges, base_fact_ids)
    trigger_ranks = seed_ranks_for_hyperedges(nary_index, candidate_hyperedges, raw_by_fact, base_fact_ids)

    raw_gold_facts = sorted(set(gold_fact_ids) & raw_completion_facts)
    variant_gold_facts = sorted(set(gold_fact_ids) & variant_completion_facts)
    selected_gold_facts = sorted(set(gold_fact_ids) & selected_facts)
    selector_rescued = gold_eid in selected_evidence_ids
    variant_pool_rescued = bool(variant_gold_facts)
    raw_available = bool(raw_gold_facts)

    primary = ""
    subreason = ""
    if not gold_fact_ids:
        primary = "A_no_fact_node"
        subreason = "gold evidence has no FACT node in graph"
    elif not candidate_hyperedges:
        primary = "A_not_in_nary_hyperedge"
        subreason = "gold fact exists but is not a member of typed n-ary hyperedge"
    elif not triggered:
        primary = "B_no_base_seed_same_hyperedge"
        subreason = "gold hyperedge exists but no base top100 fact triggers it"
    elif not raw_available:
        primary = "B_completion_budget_or_dedup"
        subreason = "base seed can trigger hyperedge, but gold role fact is absent from raw completion50"
    elif not variant_pool_rescued:
        subreason = filter_subreason(
            item.get("question", ""),
            spec,
            raw_by_fact,
            raw_gold_facts,
        )
        primary = "C_query_role_filter" if subreason == "needed_role_score_zero" else "D_pair_or_card_budget"
    elif not selector_rescued:
        primary = "E_selector_not_selected"
        subreason = "gold completion is in variant pool but not selected in final topk"
    else:
        primary = "rescued_by_selector"
        subreason = "gold evidence is covered by selected topk"

    best_raw = best_path(raw_by_fact, raw_gold_facts)
    best_variant = best_path(variant_by_fact, variant_gold_facts)
    return {
        "question_id": qid,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "gold_evidence": gold_eid,
        "gold_fact_ids": gold_fact_ids,
        "primary": primary,
        "subreason": subreason,
        "candidate_hyperedges": candidate_hyperedges[:10],
        "triggered_hyperedges": sorted(triggered)[:10],
        "trigger_seed_ranks": trigger_ranks[:10],
        "raw_gold_facts": raw_gold_facts,
        "variant_gold_facts": variant_gold_facts,
        "selected_gold_facts": selected_gold_facts,
        "raw_available": raw_available,
        "variant_pool_rescued": variant_pool_rescued,
        "selector_rescued": selector_rescued,
        "best_raw_metadata": compact_completion_metadata(best_raw),
        "best_variant_metadata": compact_completion_metadata(best_variant),
        "gold_fact_texts": [graph.nodes[fid].text for fid in gold_fact_ids[:3] if fid in graph.nodes],
    }


def filter_subreason(question: str, spec: dict, raw_by_fact: dict[str, list[dict]], raw_gold_facts: list[str]) -> str:
    paths = [path for fact_id in raw_gold_facts for path in raw_by_fact.get(fact_id, [])]
    if not paths:
        return "not_in_raw_completion"
    min_seed_rank = min(int_float(path.get("metadata", {}).get("nary_seed_fact_rank"), 9999) for path in paths)
    if min_seed_rank > int(spec.get("seed_topn", 9999)):
        return "seed_rank_beyond_variant_topn"
    if spec.get("needed_roles"):
        profile = query_profile(question)
        scores = [
            needed_role_score(
                profile,
                str(path.get("metadata", {}).get("nary_hyperedge_type") or ""),
                str(path.get("metadata", {}).get("nary_role") or ""),
            )
            for path in paths
        ]
        if max(scores, default=0.0) <= 0.0:
            return "needed_role_score_zero"
    if spec.get("card_topn"):
        return "card_topn_or_pair_topn_budget"
    if spec.get("pair_topn"):
        return "pair_topn_budget"
    return "variant_budget_or_dedup"


def triggered_hyperedges(index: NaryIndex, hyperedge_ids: list[str], base_fact_ids: set[str]) -> set[str]:
    triggered = set()
    base_fact_ids = set(base_fact_ids)
    for hid in hyperedge_ids:
        if base_fact_ids & set(index.fact_ids_by_hyperedge.get(hid, [])):
            triggered.add(hid)
    return triggered


def seed_ranks_for_hyperedges(
    index: NaryIndex,
    hyperedge_ids: list[str],
    raw_by_fact: dict[str, list[dict]],
    base_fact_ids: set[str],
) -> list[dict]:
    rows = []
    for hid in hyperedge_ids:
        seeds = sorted(base_fact_ids & set(index.fact_ids_by_hyperedge.get(hid, [])))
        ranks = []
        for paths in raw_by_fact.values():
            for path in paths:
                md = path.get("metadata", {})
                if md.get("nary_hyperedge_id") == hid:
                    ranks.append(int_float(md.get("nary_seed_fact_rank"), 9999))
        rows.append({"hyperedge_id": hid, "base_seed_count": len(seeds), "best_completion_seed_rank": min(ranks) if ranks else 0})
    return sorted(rows, key=lambda row: (row["best_completion_seed_rank"] or 9999, -row["base_seed_count"]))


def build_evidence_to_facts(graph: MemoryGraph) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for node in graph.iter_nodes(NodeType.FACT):
        for eid in evidence_ids_for_node(graph, node.node_id):
            output[normalize_evidence_id(eid)].add(node.node_id)
    return output


def keyed_items(items: list[dict]) -> dict[str, dict]:
    return {str(item.get("question_id") or item.get("query_id")): item for item in items}


def load_fold_paths(cv_dir: Path, method: str) -> list[dict]:
    items = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        items.extend(read_json(path))
    if not items:
        raise FileNotFoundError(f"no fold paths found for {method} in {cv_dir}")
    return items


def paths_by_fact(paths: list[dict]) -> dict[str, list[dict]]:
    output = defaultdict(list)
    for path in paths:
        fact_id = evidence_node_id(path)
        if fact_id:
            output[fact_id].append(path)
    return dict(output)


def selected_evidence(graph: MemoryGraph, paths: list[dict]) -> set[str]:
    output = set()
    for path in paths:
        fact_id = evidence_node_id(path)
        if fact_id:
            output.update(fact_evidence(graph, fact_id))
    return output


def fact_evidence(graph: MemoryGraph, fact_id: str) -> set[str]:
    return {normalize_evidence_id(eid) for eid in evidence_ids_for_node(graph, fact_id)}


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def is_completion(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or metadata.get("candidate_source") == "nary_completion"


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def fact_roles(node: Node) -> dict[str, str]:
    output = {}
    roles = node.metadata.get("roles") or {}
    if isinstance(roles, dict):
        for role, payload in roles.items():
            if not isinstance(payload, dict):
                continue
            for fact_id in payload.get("fact_ids", []) or []:
                fact_id = str(fact_id)
                if fact_id and fact_id not in output:
                    output[fact_id] = str(role)
    for fact_id in node.metadata.get("fact_ids", []) or node.support_ids:
        fact_id = str(fact_id)
        if fact_id and fact_id not in output:
            output[fact_id] = "support"
    return output


def best_path(by_fact: dict[str, list[dict]], fact_ids: list[str]) -> dict | None:
    paths = [path for fact_id in fact_ids for path in by_fact.get(fact_id, [])]
    if not paths:
        return None
    return max(paths, key=path_score)


def path_score(path: dict) -> float:
    scores = path.get("scores", {})
    return max(
        float_safe(scores.get("nary_pair_ce"), float("-inf")),
        float_safe(scores.get("cross_encoder"), float("-inf")),
        float_safe(path.get("score"), float("-inf")),
    )


def compact_completion_metadata(path: dict | None) -> dict:
    if not path:
        return {}
    md = path.get("metadata", {})
    scores = path.get("scores", {})
    keys = [
        "nary_hyperedge_id",
        "nary_hyperedge_type",
        "nary_role",
        "nary_seed_fact_id",
        "nary_seed_fact_rank",
        "nary_completion_rank",
        "v3_8_seed_rank",
        "v3_8_pair_ce",
        "v3_8_card_ce",
        "v3_8_needed_role_score",
    ]
    return {
        "fact_id": evidence_node_id(path),
        "score": path_score(path),
        "cross_encoder": scores.get("cross_encoder"),
        "nary_pair_ce": scores.get("nary_pair_ce"),
        "nary_card_ce": scores.get("nary_card_ce"),
        **{key: md.get(key) for key in keys if key in md},
    }


def query_profile(query: str) -> dict:
    lowered = query.lower()
    type_weights = defaultdict(float)
    role_weights = defaultdict(float)
    if any(word in lowered for word in ["prefer", "preference", "favorite", "favourite", "like", "likes", "enjoy", "enjoys"]):
        type_weights["preference"] = 1.0
        role_weights["preference_value"] = 1.0
        role_weights["polarity"] = 0.7
    if any(word in lowered for word in ["constraint", "limit", "restriction", "cannot", "can't", "avoid", "allergy", "requirement"]):
        type_weights["preference"] = max(type_weights["preference"], 0.8)
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 0.8)
        role_weights["constraint"] = 1.0
        role_weights["exception"] = 0.8
    if any(word in lowered for word in ["change", "changed", "switch", "switched", "instead", "no longer", "used to", "previously", "before", "after"]):
        type_weights["change"] = 1.0
        role_weights["old_state"] = 0.8
        role_weights["new_state"] = 1.0
    if any(word in lowered for word in ["plan", "plans", "planned", "schedule", "trip", "travel", "meeting", "deadline", "appointment", "task"]):
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 1.0)
        role_weights["plan_goal"] = 1.0
        role_weights["constraint"] = max(role_weights["constraint"], 0.7)
    if any(word in lowered for word in ["status", "state", "current", "currently", "where", "what is", "what was", "how is", "how was"]):
        type_weights["state"] = max(type_weights["state"], 0.8)
        role_weights["state_value"] = 1.0
        role_weights["context"] = 0.5
    if lowered.startswith("why") or " why " in lowered or "because" in lowered or "reason" in lowered:
        type_weights["change"] = max(type_weights["change"], 0.8)
        type_weights["plan_constraint"] = max(type_weights["plan_constraint"], 0.8)
        role_weights["reason_or_trigger"] = 1.0
    if lowered.startswith("when") or "date" in lowered or "time" in lowered:
        role_weights["temporal_scope"] = 1.0
    return {"type_weights": dict(type_weights), "role_weights": dict(role_weights), "terms": set(content_terms(query))}


def needed_role_score(profile: dict, relation_type: str, role: str) -> float:
    type_score = profile["type_weights"].get(relation_type, 0.0)
    role_score = profile["role_weights"].get(role, 0.0)
    if role_score == 0.0 and role in {"constraint", "reason_or_trigger", "temporal_scope"} and type_score > 0.7:
        role_score = 0.35
    if role_score == 0.0 and relation_type == "state" and role in {"state_value", "context"}:
        role_score = 0.2 * max(type_score, 0.5)
    return max(role_score, 0.25 * type_score if role in {"preference_value", "plan_goal", "state_value", "new_state"} else 0.0)


def summarize(rows: list[dict], rescued: Counter, args: argparse.Namespace, nary_index: NaryIndex) -> dict:
    primary = Counter(row["primary"] for row in rows)
    subreason = Counter(row["subreason"] for row in rows)
    by_question_primary = defaultdict(Counter)
    for row in rows:
        by_question_primary[row["question_id"]][row["primary"]] += 1
    examples = defaultdict(list)
    for row in rows:
        if len(examples[row["primary"]]) < args.example_limit:
            examples[row["primary"]].append(row)
    return {
        "config": {
            "graph": str(resolve_path(args.graph)),
            "base_paths": str(resolve_path(args.base_paths)),
            "raw_completion_paths": str(resolve_path(args.raw_completion_paths)),
            "variant_name": args.variant_name,
            "variant_paths": str(resolve_path(args.variant_paths)),
            "selector_cv_dir": str(resolve_path(args.selector_cv_dir)),
            "selector_method": args.selector_method,
            "selector_topk": args.selector_topk,
            "nary_hyperedges": len(nary_index.hyperedges),
        },
        "total_missing_gold_evidence": len(rows),
        "questions_with_missing_gold": len(by_question_primary),
        "primary_counts": dict(primary.most_common()),
        "primary_rates": {key: value / max(len(rows), 1) for key, value in primary.items()},
        "subreason_counts": dict(subreason.most_common()),
        "rescued_counts": dict(rescued),
        "avg_candidate_hyperedges": mean([len(row["candidate_hyperedges"]) for row in rows]) if rows else 0.0,
        "avg_triggered_hyperedges": mean([len(row["triggered_hyperedges"]) for row in rows]) if rows else 0.0,
        "examples": dict(examples),
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.8 N-ary Missing Gold Attribution",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(payload["config"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Summary",
        "",
        f"- Missing gold evidence instances after Base100: {payload['total_missing_gold_evidence']}",
        f"- Questions with missing gold: {payload['questions_with_missing_gold']}",
        f"- Avg candidate n-ary hyperedges per missing gold: {payload['avg_candidate_hyperedges']:.2f}",
        f"- Avg triggered hyperedges per missing gold: {payload['avg_triggered_hyperedges']:.2f}",
        "",
        "## Attribution",
        "",
        "| Category | Count | Rate | Meaning |",
        "| --- | ---: | ---: | --- |",
    ]
    meanings = {
        "A_no_fact_node": "gold evidence has no FACT node in the graph",
        "A_not_in_nary_hyperedge": "gold FACT exists but is not covered by typed n-ary hyperedges",
        "B_no_base_seed_same_hyperedge": "gold hyperedge exists, but Base100 does not trigger it",
        "B_completion_budget_or_dedup": "trigger exists, but raw completion50 does not include the gold role fact",
        "C_query_role_filter": "raw gold completion exists, but query-needed role filter removes it",
        "D_pair_or_card_budget": "raw gold completion exists, but V3.8 pair/card/topK budget removes it",
        "E_selector_not_selected": "gold completion is in variant pool, but selector does not select it",
        "rescued_by_selector": "gold completion is selected in final topK",
    }
    for key, count in payload["primary_counts"].items():
        rate = payload["primary_rates"].get(key, 0.0)
        lines.append(f"| {key} | {count} | {rate:.3f} | {meanings.get(key, '')} |")
    lines.extend(["", "## Subreasons", "", "```json", json.dumps(payload["subreason_counts"], indent=2), "```", ""])
    lines.extend(["", "## Examples", ""])
    for key, rows in payload["examples"].items():
        lines.extend([f"### {key}", ""])
        for row in rows[:5]:
            lines.append(
                f"- `{row['question_id']}` gold={row['gold_evidence']} sub={row['subreason']} "
                f"facts={row['gold_fact_ids'][:2]} hyp={len(row['candidate_hyperedges'])} "
                f"trig={len(row['triggered_hyperedges'])}"
            )
            if row.get("gold_fact_texts"):
                lines.append(f"  - fact: {row['gold_fact_texts'][0][:180]}")
        lines.append("")
    return "\n".join(lines)


def int_float(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def float_safe(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    main()
