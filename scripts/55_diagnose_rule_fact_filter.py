from __future__ import annotations

import argparse
from collections import Counter

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType
from hytopomem.memory.semantic_hierarchy_builder import (
    extract_rule_statement,
    support_raw_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--output", default="outputs/eval/graph_v3_rule_filter_diagnostics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph or config["graph"]["graph_path"]))
    questions_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = read_json(questions_path)

    observation_supports: dict[str, set[str]] = {}
    rules = []
    for node in graph.iter_nodes(NodeType.FACT):
        conv_id = conversation_id(node.node_id)
        if node.source == "locomo_observation":
            observation_supports.setdefault(conv_id, set()).update(turn_ids(node))
        elif node.source == "rule_extracted":
            rules.append(node)

    selected = []
    rejected = []
    signal_counts: Counter[str] = Counter()
    selected_turns: set[tuple[str, str]] = set()
    uncovered_rules = []
    for rule in rules:
        conv_id = conversation_id(rule.node_id)
        ids = turn_ids(rule)
        if ids and set(ids).issubset(observation_supports.get(conv_id, set())):
            continue
        uncovered_rules.append(rule)
        statement = extract_rule_statement(rule)
        row = {
            "fact_id": rule.node_id,
            "raw_text": rule.text,
            "support_turn_ids": ids,
            "statement": statement.text if statement is not None else None,
            "score": statement.score if statement is not None else 0.0,
            "signals": list(statement.signals) if statement is not None else [],
        }
        if statement is not None and statement.score >= args.threshold:
            selected.append(row)
            signal_counts.update(statement.signals)
            selected_turns.update((conv_id, turn_id) for turn_id in ids)
        else:
            rejected.append(row)

    unique_gold = {
        (conversation_id(qa["question_id"]), normalize_evidence_id(evidence_id))
        for conversation in questions
        for qa in conversation.get("qa", [])
        for evidence_id in qa.get("evidence", [])
    }
    observation_gold = {
        (conv_id, turn_id)
        for conv_id, turn_id in unique_gold
        if turn_id in observation_supports.get(conv_id, set())
    }
    uncovered_gold = unique_gold - observation_gold
    recovered_gold = uncovered_gold & selected_turns
    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph or config["graph"]["graph_path"])),
            "questions": str(questions_path),
            "threshold": args.threshold,
            "selection_uses_qa_or_gold": False,
        },
        "rule_filter": {
            "total_rule_facts": len(rules),
            "uncovered_rule_facts": len(uncovered_rules),
            "selected": len(selected),
            "rejected": len(rejected),
            "selected_ratio": len(selected) / len(uncovered_rules) if uncovered_rules else 0.0,
            "signal_counts": dict(signal_counts),
        },
        "gold_diagnosis": {
            "unique_gold_turns": len(unique_gold),
            "observation_covered_gold": len(observation_gold),
            "observation_uncovered_gold": len(uncovered_gold),
            "filtered_rule_recovered_gold": len(recovered_gold),
            "filtered_rule_recovery_rate": len(recovered_gold) / len(uncovered_gold) if uncovered_gold else 0.0,
            "remaining_uncovered_gold": len(uncovered_gold - recovered_gold),
        },
        "selected_samples": sorted(selected, key=lambda row: (-row["score"], row["fact_id"]))[: args.sample_size],
        "rejected_samples": sorted(rejected, key=lambda row: (-row["score"], row["fact_id"]))[: args.sample_size],
    }
    output_path = resolve_path(args.output)
    write_json(payload, output_path)
    print(f"rule_filter={payload['rule_filter']}")
    print(f"gold_diagnosis={payload['gold_diagnosis']}")
    print(f"wrote {output_path}")


def conversation_id(node_id: str) -> str:
    return str(node_id).split(":fact:", 1)[0].split(":q", 1)[0]


def turn_ids(node) -> list[str]:
    return [raw_id.split(":raw:", 1)[-1] for raw_id in support_raw_ids(node)]


if __name__ == "__main__":
    main()
