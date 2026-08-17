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
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker


VARIANTS = {
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
    parser.add_argument(
        "--raw-candidates",
        default="outputs/nary_v3_6c_selector/qwen_all_base100_completion50_paths.json",
    )
    parser.add_argument("--output-dir", default="outputs/nary_v3_8_query_conditioned")
    parser.add_argument(
        "--ce-model",
        default="/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2",
    )
    parser.add_argument("--ce-device", default="cpu")
    parser.add_argument("--ce-batch-size", type=int, default=128)
    parser.add_argument("--skip-ce", action="store_true")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    raw_items = read_json(resolve_path(args.raw_candidates))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hyperedge_text = {
        node_id: node.text
        for node_id, node in graph.nodes.items()
        if getattr(node, "type", None) and str(node.type).endswith("EVIDENCE_PACK")
    }

    completion_records = collect_completion_records(graph, raw_items, hyperedge_text)
    if not args.skip_ce and completion_records:
        score_pair_and_card(args, completion_records)
    else:
        for record in completion_records:
            record["pair_ce"] = record["fact_ce"]
            record["card_ce"] = record["fact_ce"]
    annotate_paths(completion_records)

    diagnostics = build_diagnostics(graph, raw_items, completion_records)
    write_json(diagnostics, output_dir / "completion_diagnostics.json")
    (output_dir / "completion_diagnostics.md").write_text(render_diagnostics(diagnostics), encoding="utf-8")

    variant_payload = {}
    for variant_name, spec in VARIANTS.items():
        items = build_variant_items(raw_items, completion_records, spec, variant_name)
        out = output_dir / f"{variant_name}_paths.json"
        write_json(items, out)
        summary = summarize_variant(graph, items)
        variant_payload[variant_name] = {
            "path": str(out),
            "spec": spec,
            "summary": summary,
        }
        print(f"{variant_name}: {summary}")
    write_json(variant_payload, output_dir / "variant_summary.json")
    (output_dir / "variant_summary.md").write_text(render_variant_summary(variant_payload), encoding="utf-8")
    print(f"wrote {output_dir}")


def collect_completion_records(graph, items: list[dict], hyperedge_text: dict[str, str]) -> list[dict]:
    output = []
    for item in items:
        qid = item["question_id"]
        base_paths = [path for path in item.get("paths", []) if not is_completion(path)]
        base_rank_by_fact = {evidence_node_id(path): rank for rank, path in enumerate(base_paths, start=1)}
        base_text_by_fact = {
            fact_id: graph.nodes[fact_id].text for fact_id in base_rank_by_fact if graph.nodes.get(fact_id) is not None
        }
        profile = query_profile(item.get("question", ""))
        for path in item.get("paths", []):
            if not is_completion(path):
                continue
            metadata = path.get("metadata", {})
            seed_id = str(metadata.get("nary_seed_fact_id") or "")
            hyperedge_id = str(metadata.get("nary_hyperedge_id") or "")
            fact_id = evidence_node_id(path)
            node = graph.nodes.get(fact_id)
            if node is None:
                continue
            relation_type = str(metadata.get("nary_hyperedge_type") or "")
            role = str(metadata.get("nary_role") or "")
            seed_rank = int(_float(metadata.get("nary_seed_fact_rank")) or base_rank_by_fact.get(seed_id, 9999))
            relation_card = hyperedge_text.get(hyperedge_id, "")
            record = {
                "question_id": qid,
                "question": item.get("question", ""),
                "gold": {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])},
                "path": path,
                "fact_id": fact_id,
                "fact_text": node.text,
                "seed_id": seed_id,
                "seed_text": base_text_by_fact.get(seed_id, ""),
                "seed_rank": seed_rank,
                "hyperedge_id": hyperedge_id,
                "relation_card": relation_card,
                "relation_type": relation_type,
                "role": role,
                "needed_role_score": needed_role_score(profile, relation_type, role),
                "type_score": profile["type_weights"].get(relation_type, 0.0),
                "fact_ce": path_score(path),
                "is_gold": bool(fact_evidence(graph, fact_id) & {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}),
            }
            output.append(record)
    return output


def score_pair_and_card(args, records: list[dict]) -> None:
    reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
    pair_inputs = []
    card_inputs = []
    for record in records:
        relation_card = truncate(record["relation_card"], 220)
        seed_text = truncate(record["seed_text"], 120)
        fact_text = truncate(record["fact_text"], 120)
        pair_text = f"Relation: {relation_card}\nSeed evidence: {seed_text}\nCandidate role evidence: {fact_text}"
        card_text = f"Relation: {relation_card}"
        pair_inputs.append((record["question"], pair_text))
        card_inputs.append((record["question"], card_text))
    pair_scores = reranker.model.predict(pair_inputs, batch_size=reranker.batch_size, show_progress_bar=True)
    card_scores = reranker.model.predict(card_inputs, batch_size=reranker.batch_size, show_progress_bar=True)
    for record, pair_score, card_score in zip(records, pair_scores, card_scores):
        record["pair_ce"] = float(pair_score)
        record["card_ce"] = float(card_score)


def annotate_paths(records: list[dict]) -> None:
    for record in records:
        path = record["path"]
        scores = dict(path.get("scores", {}))
        scores["nary_pair_ce"] = record["pair_ce"]
        scores["nary_card_ce"] = record["card_ce"]
        scores["nary_needed_role_score"] = record["needed_role_score"]
        path["scores"] = scores
        metadata = dict(path.get("metadata", {}))
        metadata.update(
            {
                "v3_8_seed_rank": str(record["seed_rank"]),
                "v3_8_pair_ce": f"{record['pair_ce']:.6f}",
                "v3_8_card_ce": f"{record['card_ce']:.6f}",
                "v3_8_needed_role_score": f"{record['needed_role_score']:.6f}",
                "v3_8_type_score": f"{record['type_score']:.6f}",
            }
        )
        path["metadata"] = metadata


def build_variant_items(raw_items: list[dict], records: list[dict], spec: dict, variant_name: str) -> list[dict]:
    by_qid = defaultdict(list)
    for record in records:
        if record["seed_rank"] > spec["seed_topn"]:
            continue
        if spec["needed_roles"] and record["needed_role_score"] <= 0.0:
            continue
        by_qid[record["question_id"]].append(record)
    selected_by_qid = {}
    for qid, rows in by_qid.items():
        if spec["card_topn"]:
            top_cards = {
                hyperedge_id
                for hyperedge_id, _ in sorted(
                    best_card_scores(rows).items(), key=lambda item: item[1], reverse=True
                )[: spec["card_topn"]]
            }
            rows = [row for row in rows if row["hyperedge_id"] in top_cards]
        if spec["pair_topn"]:
            rows = sorted(rows, key=lambda row: (row["pair_ce"], row["card_ce"], row["needed_role_score"]), reverse=True)[
                : spec["pair_topn"]
            ]
        else:
            rows = sorted(rows, key=lambda row: (row["fact_ce"], row["pair_ce"], row["needed_role_score"]), reverse=True)
        selected_by_qid[qid] = rows

    output = []
    for item in raw_items:
        qid = item["question_id"]
        base_paths = [path for path in item.get("paths", []) if not is_completion(path)]
        completion_paths = [row["path"] for row in selected_by_qid.get(qid, [])]
        merged = merge_paths(base_paths, completion_paths)
        merged.sort(key=path_sort_score, reverse=True)
        copied = dict(item)
        copied["paths"] = merged
        metadata = dict(copied.get("metadata", {}))
        metadata.update({"method": f"v3_8_{variant_name}", "v3_8_variant": variant_name, "v3_8_spec": spec})
        copied["metadata"] = metadata
        output.append(copied)
    return output


def build_diagnostics(graph, raw_items: list[dict], records: list[dict]) -> dict:
    by_qid = defaultdict(list)
    for record in records:
        by_qid[record["question_id"]].append(record)
    rows = []
    for item in raw_items:
        qid = item["question_id"]
        q_records = by_qid.get(qid, [])
        gold_records = [row for row in q_records if row["is_gold"]]
        rows.append(
            {
                "question_id": qid,
                "completion_count": len(q_records),
                "gold_completion_count": len(gold_records),
                "best_gold_pair_rank": rank_of_first_gold(q_records, "pair_ce"),
                "best_gold_fact_rank": rank_of_first_gold(q_records, "fact_ce"),
                "best_gold_seed_rank": min([row["seed_rank"] for row in gold_records], default=0),
            }
        )
    gold_records = [row for row in records if row["is_gold"]]
    non_gold_records = [row for row in records if not row["is_gold"]]
    return {
        "questions": len(raw_items),
        "completion_records": len(records),
        "gold_completion_records": len(gold_records),
        "questions_with_gold_completion": sum(row["gold_completion_count"] > 0 for row in rows),
        "avg_completion_per_question": mean([row["completion_count"] for row in rows]) if rows else 0.0,
        "avg_gold_completion_per_question": mean([row["gold_completion_count"] for row in rows]) if rows else 0.0,
        "gold": distribution_summary(gold_records),
        "non_gold": distribution_summary(non_gold_records),
        "gold_by_type": type_counter(gold_records),
        "non_gold_by_type": type_counter(non_gold_records),
        "rank_summary": {
            "gold_fact_rank_le_5": sum(0 < row["best_gold_fact_rank"] <= 5 for row in rows),
            "gold_fact_rank_le_20": sum(0 < row["best_gold_fact_rank"] <= 20 for row in rows),
            "gold_pair_rank_le_5": sum(0 < row["best_gold_pair_rank"] <= 5 for row in rows),
            "gold_pair_rank_le_20": sum(0 < row["best_gold_pair_rank"] <= 20 for row in rows),
            "gold_seed_rank_le_10": sum(0 < row["best_gold_seed_rank"] <= 10 for row in rows),
            "gold_seed_rank_le_20": sum(0 < row["best_gold_seed_rank"] <= 20 for row in rows),
            "gold_seed_rank_le_50": sum(0 < row["best_gold_seed_rank"] <= 50 for row in rows),
        },
        "examples": gold_examples(graph, gold_records, limit=20),
    }


def summarize_variant(graph, items: list[dict]) -> dict:
    total_completion = 0
    total_gold_completion = 0
    total_paths = 0
    for item in items:
        gold = {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}
        for path in item.get("paths", []):
            total_paths += 1
            if is_completion(path):
                total_completion += 1
                if fact_evidence(graph, evidence_node_id(path)) & gold:
                    total_gold_completion += 1
    return {
        "questions": len(items),
        "avg_paths": total_paths / max(len(items), 1),
        "avg_completion": total_completion / max(len(items), 1),
        "completion_gold_rate": total_gold_completion / max(total_completion, 1),
        "gold_completion": total_gold_completion,
        "selected_completion": total_completion,
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


def rank_of_first_gold(records: list[dict], key: str) -> int:
    ranked = sorted(records, key=lambda row: row.get(key, 0.0), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        if row["is_gold"]:
            return rank
    return 0


def distribution_summary(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "avg_fact_ce": mean(row["fact_ce"] for row in records),
        "avg_pair_ce": mean(row["pair_ce"] for row in records),
        "avg_card_ce": mean(row["card_ce"] for row in records),
        "avg_seed_rank": mean(row["seed_rank"] for row in records),
        "avg_needed_role_score": mean(row["needed_role_score"] for row in records),
    }


def type_counter(records: list[dict]) -> dict:
    counter = Counter(row["relation_type"] for row in records)
    return dict(counter.most_common())


def gold_examples(graph, records: list[dict], limit: int) -> list[dict]:
    output = []
    for row in sorted(records, key=lambda record: record["pair_ce"], reverse=True)[:limit]:
        output.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "type": row["relation_type"],
                "role": row["role"],
                "seed_rank": row["seed_rank"],
                "fact_ce": row["fact_ce"],
                "pair_ce": row["pair_ce"],
                "card_ce": row["card_ce"],
                "fact_text": row["fact_text"],
                "seed_text": row["seed_text"],
            }
        )
    return output


def best_card_scores(records: list[dict]) -> dict[str, float]:
    scores = {}
    for row in records:
        scores[row["hyperedge_id"]] = max(scores.get(row["hyperedge_id"], float("-inf")), row["card_ce"])
    return scores


def merge_paths(base_paths: list[dict], completion_paths: list[dict]) -> list[dict]:
    output = []
    seen = set()
    for path in list(base_paths) + list(completion_paths):
        fact_id = evidence_node_id(path)
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            output.append(path)
    return output


def path_sort_score(path: dict) -> float:
    scores = path.get("scores", {})
    if is_completion(path):
        return max(_float(scores.get("cross_encoder")), _float(scores.get("nary_pair_ce")))
    return _float(scores.get("cross_encoder", path.get("score", 0.0)))


def is_completion(path: dict) -> bool:
    metadata = path.get("metadata", {})
    return str(metadata.get("is_nary_completion", "")).lower() == "true" or metadata.get("candidate_source") == "nary_completion"


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def path_score(path: dict) -> float:
    scores = path.get("scores", {})
    return _float(scores.get("cross_encoder", path.get("score", 0.0)))


def fact_evidence(graph, fact_id: str) -> set[str]:
    return {normalize_evidence_id(eid) for eid in evidence_ids_for_node(graph, fact_id)}


def truncate(text: str, words: int) -> str:
    parts = str(text).split()
    return " ".join(parts[:words])


def _float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def render_diagnostics(payload: dict) -> str:
    lines = [
        "# V3.8 Completion Diagnostics",
        "",
        f"- Questions: {payload['questions']}",
        f"- Completion records: {payload['completion_records']}",
        f"- Gold completion records: {payload['gold_completion_records']}",
        f"- Questions with gold completion: {payload['questions_with_gold_completion']}",
        "",
        "## Distribution",
        "",
        "| Split | Count | Avg Fact CE | Avg Pair CE | Avg Card CE | Avg Seed Rank | Avg Needed Role |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["gold", "non_gold"]:
        row = payload[name]
        lines.append(
            f"| {name} | {row.get('count', 0)} | {row.get('avg_fact_ce', 0.0):.4f} | "
            f"{row.get('avg_pair_ce', 0.0):.4f} | {row.get('avg_card_ce', 0.0):.4f} | "
            f"{row.get('avg_seed_rank', 0.0):.2f} | {row.get('avg_needed_role_score', 0.0):.4f} |"
        )
    lines.extend(["", "## Rank Summary", "", "```json", json.dumps(payload["rank_summary"], indent=2), "```", ""])
    lines.extend(["", "## Gold by Type", "", "```json", json.dumps(payload["gold_by_type"], indent=2), "```", ""])
    return "\n".join(lines)


def render_variant_summary(payload: dict) -> str:
    lines = [
        "# V3.8 Query-conditioned Candidate Variants",
        "",
        "| Variant | Avg Paths | Avg Completion | Completion Gold Rate | Gold Completion | Selected Completion |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload.items():
        row = item["summary"]
        lines.append(
            f"| {name} | {row['avg_paths']:.1f} | {row['avg_completion']:.2f} | "
            f"{row['completion_gold_rate']:.4f} | {row['gold_completion']} | {row['selected_completion']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
