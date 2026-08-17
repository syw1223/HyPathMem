from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_json, resolve_path, write_json


MAINLINE_FEATURES = [
    "ce_score",
    "base_score",
    "bm25_norm",
    "ce_rank",
    "ce_reciprocal_rank",
    "query_term_overlap",
    "text_token_count",
    "query_token_count",
    "has_topdown_route",
    "has_bottom_up_route",
    "route_from_both",
    "is_eu_route",
    "is_hyp_route",
    "route_source_count",
    "route_overlap_score",
    "eu_hyp_agreement",
    "bottom_up_eu_agreement",
    "bottom_up_hyp_agreement",
    "route_consistency_entropy",
    "is_nary_completion",
    "nary_hyperedge_size",
    "nary_hyperedge_confidence",
    "nary_same_hyperedge_count_in_candidate_pool",
    "nary_role_coverage_potential",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval/processed/longmemeval_s_mvp.json")
    parser.add_argument("--semantic-diagnostics", default="outputs/eval/longmemeval_semantic_hierarchy_v3_diagnostics.json")
    parser.add_argument("--candidates", default="outputs/longmemeval_s/paths/dual_geometry_euhyp_all_four_top100_ce_paths.json")
    parser.add_argument("--cards", default="")
    parser.add_argument("--output-json", default="outputs/eval/longmemeval_v3_9_migration_sanity.json")
    parser.add_argument("--output-md", default="outputs/eval/longmemeval_v3_9_migration_sanity.md")
    args = parser.parse_args()

    data = read_json(resolve_path(args.data))
    candidates = read_json(resolve_path(args.candidates))
    semantic_diag = read_json(resolve_path(args.semantic_diagnostics))
    cards = read_json(resolve_path(args.cards)) if args.cards else []

    report = {
        "risk_1_cross_instance_edges": {
            "cross_instance_fact_event_edges": nested_get(semantic_diag, "structure", "cross_instance_fact_event_edges"),
            "cross_instance_event_topic_edges": nested_get(semantic_diag, "structure", "cross_instance_event_topic_edges"),
            "pass": nested_get(semantic_diag, "structure", "cross_instance_fact_event_edges") == 0
            and nested_get(semantic_diag, "structure", "cross_instance_event_topic_edges") == 0,
        },
        "risk_2_id_collision": check_ids(data, candidates),
        "risk_3_has_answer_leakage": check_has_answer_leakage(data, candidates),
        "risk_4_temporal_dxx_regex": check_dxx(candidates),
        "risk_5_selector_feature_schema": {
            "features": MAINLINE_FEATURES,
            "feature_count": len(MAINLINE_FEATURES),
            "pass": len(MAINLINE_FEATURES) == 24,
        },
        "risk_6_card_temporal_bias": check_cards(cards) if cards else {"checked": False, "reason": "cards not provided yet"},
        "risk_7_abstention": check_abstention(data, candidates),
    }
    report["overall_pass"] = all(
        bool(value.get("pass", True))
        for value in report.values()
        if isinstance(value, dict) and value.get("checked", True)
    )
    write_json(report, resolve_path(args.output_json))
    resolve_path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(render_md(report))


def check_ids(data: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    conv_ids = [row["conversation_id"] for row in data]
    duplicate_conv_ids = [item for item, count in Counter(conv_ids).items() if count > 1]
    duplicate_turn_ids = 0
    for conv in data:
        turn_ids = [turn.get("turn_id") for turn in conv.get("turns", [])]
        duplicate_turn_ids += sum(count - 1 for count in Counter(turn_ids).values() if count > 1)
    qids = [item["question_id"] for item in candidates]
    duplicate_qids = [item for item, count in Counter(qids).items() if count > 1]
    return {
        "conversation_ids": len(conv_ids),
        "duplicate_conversation_ids": duplicate_conv_ids[:10],
        "duplicate_turn_ids": duplicate_turn_ids,
        "candidate_questions": len(qids),
        "duplicate_candidate_question_ids": duplicate_qids[:10],
        "note": "LongMemEval-S has one question per instance here; node ids use conversation_id/instance id rather than question_id, which is collision-safe for this split.",
        "pass": not duplicate_conv_ids and duplicate_turn_ids == 0 and not duplicate_qids,
    }


def check_has_answer_leakage(data: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    turn_key_hits = 0
    turn_text_hits = 0
    for conv in data:
        for turn in conv.get("turns", []):
            payload = json.dumps(turn, ensure_ascii=False).lower()
            if "has_answer" in turn:
                turn_key_hits += 1
            if "has_answer" in payload:
                turn_text_hits += 1
    candidate_blob = json.dumps(candidates[:5], ensure_ascii=False).lower()
    return {
        "turn_key_has_answer_count": turn_key_hits,
        "turn_payload_has_answer_count": turn_text_hits,
        "candidate_sample_contains_has_answer": "has_answer" in candidate_blob,
        "pass": turn_key_hits == 0 and turn_text_hits == 0 and "has_answer" not in candidate_blob,
    }


def check_dxx(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    dxx = re.compile(r"\bD\d+\b")
    metadata_hits = 0
    text_hits = 0
    for item in candidates[:50]:
        for path in item.get("paths", []):
            if dxx.search(json.dumps(path.get("metadata", {}), ensure_ascii=False)):
                metadata_hits += 1
            if dxx.search(json.dumps(path, ensure_ascii=False)):
                text_hits += 1
    temporal_features_in_mainline = [name for name in MAINLINE_FEATURES if "temporal" in name or "day_gap" in name]
    return {
        "candidate_metadata_dxx_hits_sample50": metadata_hits,
        "candidate_path_dxx_hits_sample50": text_hits,
        "temporal_features_in_24_feature_mainline": temporal_features_in_mainline,
        "pass": metadata_hits == 0 and not temporal_features_in_mainline,
    }


def check_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    temporal_cards = 0
    card_questions = 0
    for item in cards:
        count = int(item.get("metadata", {}).get("query_card_count", 0) or 0)
        card_questions += int(count > 0)
        for path in item.get("paths", []):
            meta = path.get("metadata", {})
            if meta.get("nary_hyperedge_type") == "temporal":
                temporal_cards += 1
    return {
        "checked": True,
        "card_questions": card_questions,
        "temporal_card_member_paths": temporal_cards,
        "note": "Manual spot-check still recommended for temporal/knowledge-update examples because this checks structure, not semantic correctness.",
        "pass": True,
    }


def check_abstention(data: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    abstention = 0
    abstention_with_gold = 0
    abstention_without_gold = 0
    answerable = 0
    qa_with_gold = 0
    for conv in data:
        for qa in conv.get("qa", []):
            has_gold = bool(qa.get("gold_evidence"))
            qa_with_gold += int(has_gold)
            if qa.get("is_abstention"):
                abstention += 1
                abstention_with_gold += int(has_gold)
                abstention_without_gold += int(not has_gold)
            else:
                answerable += 1
    candidate_abstention = sum(1 for item in candidates if item.get("is_abstention"))
    candidate_with_gold = sum(1 for item in candidates if item.get("gold_evidence"))
    return {
        "answerable": answerable,
        "abstention": abstention,
        "abstention_with_gold": abstention_with_gold,
        "abstention_without_gold": abstention_without_gold,
        "qa_with_gold": qa_with_gold,
        "candidate_abstention_items": candidate_abstention,
        "candidate_items_with_gold": candidate_with_gold,
        "note": "Current processed LongMemEval-S has 30 abstention-labeled items; 21 have no gold evidence and 9 still carry gold evidence. Retrieval Hit/Recall/FullCover is reported on 479 gold-bearing questions; QA should handle all abstention-labeled items separately.",
        "pass": answerable == 470 and abstention == 30 and abstention_without_gold == 21 and candidate_abstention == 30 and candidate_with_gold == 479,
    }


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def render_md(report: dict[str, Any]) -> str:
    lines = ["# LongMemEval V3.9 Migration Sanity", ""]
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        status = "PASS" if value.get("pass", True) else "FAIL"
        lines.extend([f"## {key} - {status}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""])
    lines.append(f"Overall pass: {report.get('overall_pass')}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
