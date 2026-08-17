from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore


SETTINGS = {
    "base100": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_type_specific.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_base100_top20",
    },
    "qwen_type_completion20": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_type_specific.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_qwen_type_completion20_top20",
    },
    "gpt4o_clean_completion20": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_gpt4o_clean.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_gpt4o_clean_completion20_top20",
    },
    "qwen_all_completion50": {
        "graph": "outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
        "cv_dir": "outputs/eval/cv/nary_v3_6c_selector_qwen_all_completion50_top20",
    },
}

METHODS = [
    "base_completion_no_nary_features",
    "base_completion_nary_point_features",
    "base_completion_nary_point_set_features",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="outputs/eval/NARY_V3_6C_SELECTED_COMPLETION_ANALYSIS.json")
    parser.add_argument("--output-md", default="outputs/eval/NARY_V3_6C_SELECTED_COMPLETION_ANALYSIS.md")
    parser.add_argument("--example-limit", type=int, default=20)
    args = parser.parse_args()

    payload = {"selected_completion": {}, "qwen_all_vs_base100": {}}
    loaded: dict[tuple[str, str], list[dict]] = {}
    graphs = {}
    for setting_name, setting in SETTINGS.items():
        graph = JsonGraphStore().load(resolve_path(setting["graph"]))
        graphs[setting_name] = graph
        setting_payload = {}
        for method in METHODS:
            items = load_fold_paths(resolve_path(setting["cv_dir"]), method)
            loaded[(setting_name, method)] = items
            setting_payload[method] = {
                "top5": selected_completion_summary(graph, items, 5),
                "top20": selected_completion_summary(graph, items, 20),
            }
        payload["selected_completion"][setting_name] = setting_payload

    base_items = loaded[("base100", "base_completion_no_nary_features")]
    qwen_items_by_method = {
        method: loaded[("qwen_all_completion50", method)] for method in METHODS
    }
    qwen_graph = graphs["qwen_all_completion50"]
    for method, qwen_items in qwen_items_by_method.items():
        payload["qwen_all_vs_base100"][method] = {
            "top5": compare_against_base(
                qwen_graph,
                base_items,
                qwen_items,
                5,
                args.example_limit,
            ),
            "top20": compare_against_base(
                qwen_graph,
                base_items,
                qwen_items,
                20,
                args.example_limit,
            ),
        }

    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def selected_completion_summary(graph, items: list[dict], k: int) -> dict:
    total_questions = len(items)
    total_selected = 0
    total_completion = 0
    selected_completion_gold = 0
    questions_with_completion = 0
    questions_with_gold_completion = 0
    type_stats: dict[str, Counter] = defaultdict(Counter)
    role_stats: dict[str, Counter] = defaultdict(Counter)
    completion_rank_stats = []
    for item in items:
        gold = gold_set(item)
        selected = list(item.get("paths", []))[:k]
        total_selected += len(selected)
        question_completion = 0
        question_gold_completion = 0
        for rank, path in enumerate(selected, start=1):
            if not is_completion(path):
                continue
            question_completion += 1
            total_completion += 1
            completion_rank_stats.append(rank)
            relation_type = relation_type_of(path)
            role = role_of(path)
            matched = fact_evidence(graph, path) & gold
            type_stats[relation_type]["selected"] += 1
            role_stats[role]["selected"] += 1
            if matched:
                selected_completion_gold += 1
                question_gold_completion += 1
                type_stats[relation_type]["gold"] += 1
                role_stats[role]["gold"] += 1
        if question_completion:
            questions_with_completion += 1
        if question_gold_completion:
            questions_with_gold_completion += 1
    return {
        "questions": total_questions,
        "avg_selected_completion": total_completion / max(total_questions, 1),
        "completion_share_of_selected": total_completion / max(total_selected, 1),
        "questions_with_completion": questions_with_completion,
        "questions_with_completion_rate": questions_with_completion / max(total_questions, 1),
        "selected_completion": total_completion,
        "selected_completion_gold": selected_completion_gold,
        "selected_completion_gold_rate": selected_completion_gold / max(total_completion, 1),
        "questions_with_gold_completion": questions_with_gold_completion,
        "questions_with_gold_completion_rate": questions_with_gold_completion / max(total_questions, 1),
        "avg_completion_rank": sum(completion_rank_stats) / max(len(completion_rank_stats), 1),
        "type_stats": counter_payload(type_stats),
        "role_stats": counter_payload(role_stats),
    }


def compare_against_base(graph, base_items: list[dict], qwen_items: list[dict], k: int, example_limit: int) -> dict:
    base_by_qid = {item["question_id"]: item for item in base_items}
    rows = []
    type_fixes = defaultdict(Counter)
    role_fixes = defaultdict(Counter)
    hit_fixed = 0
    full_fixed = 0
    hit_fixed_by_completion = 0
    full_fixed_by_completion = 0
    examples = []
    for qwen_item in qwen_items:
        qid = qwen_item["question_id"]
        base_item = base_by_qid.get(qid)
        if base_item is None:
            continue
        gold = gold_set(qwen_item)
        base_selected = list(base_item.get("paths", []))[:k]
        qwen_selected = list(qwen_item.get("paths", []))[:k]
        base_match = selected_evidence(graph, base_selected) & gold
        qwen_match = selected_evidence(graph, qwen_selected) & gold
        completion_gold_paths = [
            path for path in qwen_selected if is_completion(path) and (fact_evidence(graph, path) & gold)
        ]
        base_hit = bool(base_match)
        qwen_hit = bool(qwen_match)
        base_full = bool(gold) and gold.issubset(base_match)
        qwen_full = bool(gold) and gold.issubset(qwen_match)
        if (not base_hit) and qwen_hit:
            hit_fixed += 1
            if completion_gold_paths:
                hit_fixed_by_completion += 1
        if (not base_full) and qwen_full:
            full_fixed += 1
            if completion_gold_paths:
                full_fixed_by_completion += 1
        if completion_gold_paths:
            for path in completion_gold_paths:
                relation_type = relation_type_of(path)
                role = role_of(path)
                type_fixes[relation_type]["gold_selected"] += 1
                role_fixes[role]["gold_selected"] += 1
                if (not base_hit) and qwen_hit:
                    type_fixes[relation_type]["hit_fix"] += 1
                    role_fixes[role]["hit_fix"] += 1
                if (not base_full) and qwen_full:
                    type_fixes[relation_type]["full_fix"] += 1
                    role_fixes[role]["full_fix"] += 1
        rows.append(
            {
                "question_id": qid,
                "base_hit": base_hit,
                "qwen_hit": qwen_hit,
                "base_full": base_full,
                "qwen_full": qwen_full,
                "completion_gold_count": len(completion_gold_paths),
            }
        )
        if ((not base_hit and qwen_hit) or (not base_full and qwen_full)) and completion_gold_paths and len(examples) < example_limit:
            examples.append(
                {
                    "question_id": qid,
                    "question": qwen_item.get("question", ""),
                    "answer": qwen_item.get("answer", ""),
                    "gold_evidence": sorted(gold),
                    "base_matched": sorted(base_match),
                    "qwen_matched": sorted(qwen_match),
                    "fix_type": "hit" if (not base_hit and qwen_hit) else "full_cover",
                    "completion_gold": [completion_path_payload(graph, path, gold) for path in completion_gold_paths],
                }
            )
    return {
        "questions": len(rows),
        "hit_fixed": hit_fixed,
        "full_cover_fixed": full_fixed,
        "hit_fixed_by_selected_completion": hit_fixed_by_completion,
        "full_cover_fixed_by_selected_completion": full_fixed_by_completion,
        "completion_explains_hit_fix_rate": hit_fixed_by_completion / max(hit_fixed, 1),
        "completion_explains_full_fix_rate": full_fixed_by_completion / max(full_fixed, 1),
        "type_fix_stats": counter_payload(type_fixes),
        "role_fix_stats": counter_payload(role_fixes),
        "examples": examples,
    }


def load_fold_paths(cv_dir: Path, method: str) -> list[dict]:
    items = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        items.extend(read_json(path))
    if not items:
        raise FileNotFoundError(f"no fold paths found for {method} in {cv_dir}")
    return items


def selected_evidence(graph, paths: list[dict]) -> set[str]:
    output = set()
    for path in paths:
        output.update(fact_evidence(graph, path))
    return output


def fact_evidence(graph, path: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in evidence_ids_for_node(graph, evidence_node_id(path))}


def completion_path_payload(graph, path: dict, gold: set[str]) -> dict:
    metadata = path.get("metadata", {})
    fact_id = evidence_node_id(path)
    node = graph.nodes.get(fact_id)
    return {
        "fact_id": fact_id,
        "text": node.text if node else "",
        "matched_gold": sorted(fact_evidence(graph, path) & gold),
        "type": relation_type_of(path),
        "role": role_of(path),
        "hyperedge_id": metadata.get("nary_hyperedge_id", ""),
        "seed_fact_id": metadata.get("nary_seed_fact_id", ""),
        "seed_rank": metadata.get("nary_seed_fact_rank", ""),
        "selector_score": path.get("scores", {}).get("topology_selector", path.get("score", 0.0)),
        "ce_score": path.get("scores", {}).get("cross_encoder", 0.0),
    }


def is_completion(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or metadata.get("candidate_source") == "nary_completion"


def relation_type_of(path: dict) -> str:
    return str(path.get("metadata", {}).get("nary_hyperedge_type") or "none")


def role_of(path: dict) -> str:
    return str(path.get("metadata", {}).get("nary_role") or "none")


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", item.get("evidence", []))}


def counter_payload(stats: dict[str, Counter]) -> dict:
    output = {}
    for key, counter in stats.items():
        row = dict(counter)
        if row.get("selected"):
            row["gold_rate"] = row.get("gold", 0) / max(row.get("selected", 0), 1)
        output[key] = row
    return dict(sorted(output.items(), key=lambda item: sum(item[1].values()), reverse=True))


def render_markdown(payload: dict) -> str:
    lines = [
        "# V3.6C Selected Completion Analysis",
        "",
        "## Selected Completion Rate",
        "",
        "| Setting | Method | K | Avg Completion | Q w/ Completion | Selected Completion | Gold Completion | Gold Rate | Q w/ Gold Completion |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting_name, setting in payload["selected_completion"].items():
        for method, by_k in setting.items():
            for k_name, row in by_k.items():
                k = k_name.replace("top", "")
                lines.append(
                    f"| {setting_name} | {method} | {k} | {row['avg_selected_completion']:.3f} | "
                    f"{row['questions_with_completion']} ({row['questions_with_completion_rate']:.3f}) | "
                    f"{row['selected_completion']} | {row['selected_completion_gold']} | "
                    f"{row['selected_completion_gold_rate']:.3f} | "
                    f"{row['questions_with_gold_completion']} ({row['questions_with_gold_completion_rate']:.3f}) |"
                )
    lines.extend(
        [
            "",
            "## Qwen-all Completion50 vs Base100 Fixes",
            "",
            "| Method | K | Hit Fixed | Hit Fixed by Completion | FullCover Fixed | FullCover Fixed by Completion |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, by_k in payload["qwen_all_vs_base100"].items():
        for k_name, row in by_k.items():
            k = k_name.replace("top", "")
            lines.append(
                f"| {method} | {k} | {row['hit_fixed']} | {row['hit_fixed_by_selected_completion']} "
                f"({row['completion_explains_hit_fix_rate']:.3f}) | {row['full_cover_fixed']} | "
                f"{row['full_cover_fixed_by_selected_completion']} ({row['completion_explains_full_fix_rate']:.3f}) |"
            )
    lines.extend(["", "## Qwen-all Type Contribution: Top20", ""])
    for method, by_k in payload["qwen_all_vs_base100"].items():
        row = by_k["top20"]
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Type | Gold Selected | Hit Fix | FullCover Fix |")
        lines.append("| --- | ---: | ---: | ---: |")
        for relation_type, stats in row["type_fix_stats"].items():
            lines.append(
                f"| {relation_type} | {stats.get('gold_selected', 0)} | "
                f"{stats.get('hit_fix', 0)} | {stats.get('full_fix', 0)} |"
            )
        lines.append("")
    lines.extend(["", "## Examples: Qwen-all no-nary Top5", ""])
    examples = payload["qwen_all_vs_base100"]["base_completion_no_nary_features"]["top5"]["examples"]
    for item in examples[:10]:
        lines.append(f"- `{item['question_id']}` {item['fix_type']}: {item['question']}")
        for comp in item["completion_gold"]:
            lines.append(
                f"  - {comp['type']} / {comp['role']} / seed_rank={comp['seed_rank']}: "
                f"{comp['text'][:180]}"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
