from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evaluate_item, normalize_evidence_id, summarize
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument("--candidates", default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json")
    parser.add_argument("--card-cache", default="outputs/v3_9_query_cards/qwen3_cards_v3.jsonl")
    parser.add_argument("--base-cv-dir", default="outputs/eval/cv/nary_v3_6c_selector_base100_top20")
    parser.add_argument("--base-method", default="base_completion_no_nary_features")
    parser.add_argument("--ce-model", default=DEFAULT_LOCAL_CE)
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--top-cards", type=int, default=3)
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--output-dir", default="outputs/eval/v3_9_cardce_guided_ctx50")
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.candidates))
    records = load_card_records(resolve_path(args.card_cache))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    card_refs = []
    for item in items:
        record = records.get(item["question_id"], {})
        for card in record.get("cards", []):
            pairs.append((item.get("question", ""), card_text(card, graph)))
            card_refs.append((item["question_id"], int(card.get("card_index", 0))))

    reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
    scores = reranker.model.predict(pairs, batch_size=reranker.batch_size, show_progress_bar=True) if pairs else []
    score_map = {ref: float(score) for ref, score in zip(card_refs, scores)}

    selected_items = []
    for item in items:
        selected_items.append(cardce_guided_item(item, records.get(item["question_id"], {}), score_map, max(args.topk), args.top_cards))

    base_lgbm = load_cv_paths(resolve_path(args.base_cv_dir), args.base_method)
    methods = {
        "cardce_guided_topk": selected_items,
    }
    if base_lgbm:
        methods["base_lgbm_topk"] = base_lgbm

    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "candidates": str(resolve_path(args.candidates)),
            "card_cache": str(resolve_path(args.card_cache)),
            "ce_model": args.ce_model,
            "top_cards": args.top_cards,
            "topk": args.topk,
            "num_cards_scored": len(pairs),
        },
        "retrieval": {},
        "selected_card_analysis": {},
        "fixes_vs_base_lgbm": {},
    }
    for method, method_items in methods.items():
        write_json(method_items, output_dir / f"{method}_paths.json")
        payload["retrieval"][method] = {}
        payload["selected_card_analysis"][method] = {}
        for k in args.topk:
            eval_payload = evaluate_items(graph, method_items, k, method)
            write_json(eval_payload, output_dir / f"{method}_top{k}_eval.json")
            payload["retrieval"][method][f"top{k}"] = eval_payload["summary"]
            payload["selected_card_analysis"][method][f"top{k}"] = selected_card_summary(graph, method_items, k)

    if base_lgbm:
        for method, method_items in methods.items():
            if method == "base_lgbm_topk":
                continue
            payload["fixes_vs_base_lgbm"][method] = {}
            for k in args.topk:
                payload["fixes_vs_base_lgbm"][method][f"top{k}"] = compare_to_base(graph, base_lgbm, method_items, k)

    write_json(payload, output_dir / "v3_9_cardce_guided_eval.json")
    (output_dir / "v3_9_cardce_guided_eval.md").write_text(render_markdown(payload), encoding="utf-8")
    print(render_markdown(payload))
    print(f"wrote {output_dir / 'v3_9_cardce_guided_eval.md'}")


def cardce_guided_item(item: dict, record: dict, score_map: dict[tuple[str, int], float], topk: int, top_cards: int) -> dict:
    path_by_fact = {evidence_node_id(path): path for path in item.get("paths", []) if evidence_node_id(path)}
    cards = []
    for card in record.get("cards", []):
        card_index = int(card.get("card_index", 0))
        cards.append((score_map.get((item["question_id"], card_index), 0.0), card))
    cards.sort(key=lambda row: row[0], reverse=True)

    selected = []
    seen = set()
    for card_ce, card in cards[:top_cards]:
        for fact_id in card_fact_order(card):
            path = path_by_fact.get(fact_id)
            if path is None or fact_id in seen:
                continue
            copied_path = dict(path)
            metadata = dict(copied_path.get("metadata", {}))
            metadata["v3_9_cardce_selected"] = "true"
            metadata["v3_9_cardce_score"] = f"{card_ce:.6f}"
            metadata["v3_9_cardce_type"] = card.get("type", "")
            metadata["v3_9_cardce_summary"] = card.get("summary", "")
            copied_path["metadata"] = metadata
            scores = dict(copied_path.get("scores", {}))
            scores["v3_9_card_ce"] = card_ce
            copied_path["scores"] = scores
            selected.append(copied_path)
            seen.add(fact_id)
            if len(selected) >= topk:
                break
        if len(selected) >= topk:
            break

    for path in sorted(item.get("paths", []), key=ce_score, reverse=True):
        fact_id = evidence_node_id(path)
        if fact_id and fact_id not in seen:
            selected.append(path)
            seen.add(fact_id)
        if len(selected) >= topk:
            break

    copied = dict(item)
    copied["paths"] = selected
    metadata = dict(copied.get("metadata", {}))
    metadata["method"] = "v3_9_cardce_guided_no_expansion"
    copied["metadata"] = metadata
    return copied


def card_fact_order(card: dict) -> list[str]:
    ordered = []
    needed = set(card.get("needed_roles") or [])
    roles = card.get("roles") or {}
    for role in card.get("needed_roles") or []:
        payload = roles.get(role) or {}
        ordered.extend(str(fid) for fid in payload.get("fact_ids", []) or [])
    if not ordered and needed:
        for role, payload in roles.items():
            if role in needed:
                ordered.extend(str(fid) for fid in payload.get("fact_ids", []) or [])
    ordered.extend(str(fid) for fid in card.get("support_facts", []) or [])
    return dedupe(ordered)


def card_text(card: dict, graph) -> str:
    parts = [
        f"type: {card.get('type', '')}",
        f"entity: {card.get('entity', '')}",
        f"aspect: {card.get('aspect', '')}",
        f"summary: {card.get('summary', '')}",
        f"why relevant: {card.get('why_relevant_to_query', '')}",
    ]
    roles = card.get("roles") or {}
    for role, payload in roles.items():
        value = payload.get("value", "") if isinstance(payload, dict) else ""
        facts = []
        for fact_id in payload.get("fact_ids", []) if isinstance(payload, dict) else []:
            node = graph.nodes.get(str(fact_id))
            if node is not None:
                facts.append(f"{fact_id}: {node.text}")
        parts.append(f"role {role}: {value}; facts: {' | '.join(facts)}")
    return "\n".join(parts)


def load_card_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record = row.get("record", {})
            if record.get("question_id"):
                records[str(record["question_id"])] = record
    return records


def load_cv_paths(cv_dir: Path, method: str) -> list[dict]:
    paths = []
    for path in sorted(cv_dir.glob(f"fold_*/*{method}_paths.json")):
        paths.extend(read_json(path))
    return sorted(paths, key=lambda item: item["question_id"]) if paths else []


def evaluate_items(graph, items: list[dict], k: int, method: str) -> dict:
    rows = [evaluate_item(graph, item, k) for item in items]
    return {"method": method, "k": k, "summary": summarize(rows), "per_question": [row.__dict__ for row in rows]}


def selected_card_summary(graph, items: list[dict], k: int) -> dict:
    selected = 0
    selected_gold = 0
    questions_with_selected = 0
    questions_with_gold = 0
    for item in items:
        gold = gold_set(item)
        q_selected = 0
        q_gold = 0
        for path in item.get("paths", [])[:k]:
            if str(path.get("metadata", {}).get("v3_9_cardce_selected", "")).lower() != "true":
                continue
            q_selected += 1
            selected += 1
            if fact_evidence(graph, evidence_node_id(path)) & gold:
                q_gold += 1
                selected_gold += 1
        if q_selected:
            questions_with_selected += 1
        if q_gold:
            questions_with_gold += 1
    return {
        "selected_cardce_facts": selected,
        "selected_cardce_gold_facts": selected_gold,
        "selected_cardce_gold_rate": selected_gold / max(selected, 1),
        "questions_with_cardce": questions_with_selected,
        "questions_with_gold_cardce": questions_with_gold,
    }


def compare_to_base(graph, base_items: list[dict], method_items: list[dict], k: int) -> dict:
    base_by_qid = {item["question_id"]: item for item in base_items}
    hit_fixed = hit_regressed = full_fixed = full_regressed = 0
    for item in method_items:
        base = base_by_qid.get(item["question_id"])
        if base is None:
            continue
        base_eval = evaluate_item(graph, base, k)
        method_eval = evaluate_item(graph, item, k)
        hit_fixed += int((not base_eval.hit) and method_eval.hit)
        hit_regressed += int(base_eval.hit and (not method_eval.hit))
        full_fixed += int((not base_eval.full_cover) and method_eval.full_cover)
        full_regressed += int(base_eval.full_cover and (not method_eval.full_cover))
    return {
        "hit_fixed": hit_fixed,
        "hit_regressed": hit_regressed,
        "full_cover_fixed": full_fixed,
        "full_cover_regressed": full_regressed,
    }


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def ce_score(path: dict) -> float:
    scores = path.get("scores", {})
    return float(scores.get("cross_encoder", path.get("score", 0.0)) or 0.0)


def fact_evidence(graph, fact_id: str) -> set[str]:
    node = graph.nodes.get(fact_id)
    if node is None:
        return set()
    out = {normalize_evidence_id(eid) for eid in node.support_ids}
    if not out:
        out.add(normalize_evidence_id(fact_id))
    return out


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("gold_evidence", [])}


def dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def render_markdown(payload: dict) -> str:
    lines = ["# V3.9 CardCE-guided No-expansion Selection", ""]
    lines.extend(["## Retrieval", "", "| Method | K | Hit | Recall | FullCover |", "|---|---:|---:|---:|---:|"])
    for method, by_k in payload["retrieval"].items():
        for k, row in by_k.items():
            lines.append(f"| {method} | {k} | {row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |")
    lines.extend(["", "## Selected CardCE Facts", "", "| Method | K | Selected | Gold | GoldRate | QWithGold |", "|---|---:|---:|---:|---:|---:|"])
    for method, by_k in payload["selected_card_analysis"].items():
        for k, row in by_k.items():
            lines.append(
                f"| {method} | {k} | {row['selected_cardce_facts']} | {row['selected_cardce_gold_facts']} | "
                f"{row['selected_cardce_gold_rate']:.4f} | {row['questions_with_gold_cardce']} |"
            )
    if payload["fixes_vs_base_lgbm"]:
        lines.extend(["", "## Fixes vs Base LightGBM", "", "| Method | K | HitFixed | HitRegressed | FullFixed | FullRegressed |", "|---|---:|---:|---:|---:|---:|"])
        for method, by_k in payload["fixes_vs_base_lgbm"].items():
            for k, row in by_k.items():
                lines.append(
                    f"| {method} | {k} | {row['hit_fixed']} | {row['hit_regressed']} | "
                    f"{row['full_cover_fixed']} | {row['full_cover_regressed']} |"
                )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
