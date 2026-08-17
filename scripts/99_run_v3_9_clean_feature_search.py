from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from common import read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import NodeType, RelationType
from hytopomem.retrieval.hyperbolic_topdown_retriever import (
    load_hyperbolic_router,
    lorentz_distance_numpy,
)


GEOMETRY_FEATURES = [
    "v39_card_same_event_ratio",
    "v39_card_same_episode_ratio",
    "v39_card_same_topic_ratio",
    "v39_card_branch_entropy",
    "v39_card_bu_td_agreement",
    "v39_card_hyp_route_share",
    "v39_card_avg_hyp_distance",
    "v39_card_max_hyp_distance",
    "v39_fact_to_card_anchor_distance",
]

STRUCTURAL_GEOMETRY_FEATURES = [
    "v39_card_same_event_ratio",
    "v39_card_same_episode_ratio",
    "v39_card_same_topic_ratio",
    "v39_card_branch_entropy",
    "v39_card_bu_td_agreement",
    "v39_card_hyp_route_share",
]

HYPERBOLIC_DISTANCE_FEATURES = [
    "v39_card_avg_hyp_distance",
    "v39_card_max_hyp_distance",
    "v39_fact_to_card_anchor_distance",
]

CARDCE_FEATURES = ["v39_card_ce_score"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json",
    )
    parser.add_argument("--card-cache", default="outputs/v3_9_query_cards/qwen3_cards_v3.jsonl")
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument(
        "--geometry-graph",
        default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_3_episode.json",
    )
    parser.add_argument(
        "--embedding-cache",
        default="outputs/embeddings/graph_v3_3_episode_minilm_fact_event_episode_topic.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/models/graph_v2_lorentz_router/minilm_structure_router_v3_3_episode_hardneg.pt",
    )
    parser.add_argument(
        "--annotated-output",
        default="outputs/v3_9_query_cards/qwen3_card_ctx50_clean_geometry_paths.json",
    )
    parser.add_argument("--output-dir", default="outputs/eval/cv/v3_9_clean_feature_search_top5")
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--annotate-only", action="store_true")
    args = parser.parse_args()

    annotated_path = resolve_path(args.annotated_output)
    if not annotated_path.exists():
        annotate_geometry(args, annotated_path)
    if args.annotate_only:
        return
    run_clean_search(args, annotated_path)


def annotate_geometry(args, output_path: Path) -> None:
    items = read_json(resolve_path(args.candidates))
    card_records = load_card_records(resolve_path(args.card_cache))
    cardce_scores = load_cardce_scores(resolve_path(args.cardce_paths))
    geometry_graph = JsonGraphStore().load(resolve_path(args.geometry_graph))
    fact_event, event_episode, episode_topic = hierarchy_maps(geometry_graph)
    point_map = load_hyperbolic_points(
        resolve_path(args.embedding_cache),
        resolve_path(args.checkpoint),
        args.device,
        args.batch_size,
    )

    stats = Counter()
    output = []
    for item_index, item in enumerate(items, start=1):
        qid = str(item["question_id"])
        path_by_fact = {
            evidence_node_id(path): path
            for path in item.get("paths", [])
            if evidence_node_id(path)
        }
        payloads_by_fact = defaultdict(list)
        for card_rank, card in enumerate(card_records.get(qid, {}).get("cards", []), start=1):
            card_id = f"query_card:{card_rank:02d}"
            fact_ids = [
                fact_id
                for fact_id in card.get("support_facts", [])
                if fact_id in path_by_fact
            ]
            fact_ids = dedupe(fact_ids)
            if len(fact_ids) < 2:
                continue
            card_payload = card_geometry_payload(
                fact_ids=fact_ids,
                path_by_fact=path_by_fact,
                point_map=point_map,
                fact_event=fact_event,
                event_episode=event_episode,
                episode_topic=episode_topic,
                card_ce=cardce_scores.get((qid, card_id), 0.0),
            )
            for fact_id in fact_ids:
                payload = dict(card_payload)
                payload["v39_fact_to_card_anchor_distance"] = card_payload["anchor_distances"].get(fact_id, 0.0)
                payload.pop("anchor_distances", None)
                payloads_by_fact[fact_id].append(
                    (
                        float(card_payload["v39_card_ce_score"]),
                        float(card.get("confidence", 0.0)),
                        payload,
                    )
                )
            stats["cards"] += 1
            stats["card_facts"] += len(fact_ids)
            if card_payload["v39_card_avg_hyp_distance"] > 0.0:
                stats["cards_with_hyp_distance"] += 1

        copied = dict(item)
        copied_paths = []
        for path in item.get("paths", []):
            fact_id = evidence_node_id(path)
            copied_path = dict(path)
            metadata = dict(copied_path.get("metadata", {}))
            if payloads_by_fact.get(fact_id):
                _, _, best = max(payloads_by_fact[fact_id], key=lambda row: (row[0], row[1]))
                metadata.update({key: f"{value:.8f}" for key, value in best.items()})
                stats["annotated_facts"] += 1
            copied_path["metadata"] = metadata
            copied_paths.append(copied_path)
        copied["paths"] = copied_paths
        output.append(copied)
        if item_index % 250 == 0 or item_index == len(items):
            print(
                f"geometry annotation {item_index}/{len(items)} "
                f"cards={stats['cards']} annotated_facts={stats['annotated_facts']}",
                flush=True,
            )

    write_json(output, output_path)
    summary = {
        "output": str(output_path),
        "candidates": str(resolve_path(args.candidates)),
        "geometry_graph": str(resolve_path(args.geometry_graph)),
        "embedding_cache": str(resolve_path(args.embedding_cache)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        **dict(stats),
    }
    write_json(summary, output_path.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2))


def card_geometry_payload(
    *,
    fact_ids: list[str],
    path_by_fact: dict[str, dict],
    point_map: dict[str, np.ndarray],
    fact_event: dict[str, str],
    event_episode: dict[str, str],
    episode_topic: dict[str, str],
    card_ce: float,
) -> dict:
    pairs = list(itertools.combinations(fact_ids, 2))
    same_event = pair_ratio(pairs, lambda left, right: same_parent(left, right, fact_event))
    same_episode = pair_ratio(
        pairs,
        lambda left, right: same_parent(
            fact_event.get(left, ""),
            fact_event.get(right, ""),
            event_episode,
        ),
    )
    fact_topic = {
        fact_id: episode_topic.get(event_episode.get(fact_event.get(fact_id, ""), ""), "")
        for fact_id in fact_ids
    }
    same_topic = pair_ratio(pairs, lambda left, right: bool(fact_topic.get(left)) and fact_topic[left] == fact_topic.get(right))
    branch_entropy = normalized_entropy([fact_topic.get(fact_id, "") or "missing" for fact_id in fact_ids])

    route_rows = [route_flags(path_by_fact[fact_id]) for fact_id in fact_ids]
    bu_td_agreement = mean([float(row["bottom_up"] and row["top_down"]) for row in route_rows])
    hyp_route_share = mean([float(row["hyp"]) for row in route_rows])

    valid_fact_ids = [fact_id for fact_id in fact_ids if fact_id in point_map]
    distances = {}
    pair_distances = []
    for left, right in itertools.combinations(valid_fact_ids, 2):
        distance = float(lorentz_distance_numpy(point_map[left][None, :], point_map[right])[0])
        distances[(left, right)] = distance
        distances[(right, left)] = distance
        pair_distances.append(distance)
    avg_distance = mean(pair_distances)
    max_distance = max(pair_distances) if pair_distances else 0.0

    anchor_distances = {fact_id: 0.0 for fact_id in fact_ids}
    if len(valid_fact_ids) >= 2:
        medoid = min(
            valid_fact_ids,
            key=lambda fact_id: mean(
                [distances[(fact_id, other)] for other in valid_fact_ids if other != fact_id]
            ),
        )
        for fact_id in valid_fact_ids:
            anchor_distances[fact_id] = 0.0 if fact_id == medoid else distances[(fact_id, medoid)]

    return {
        "v39_card_ce_score": float(card_ce),
        "v39_card_same_event_ratio": same_event,
        "v39_card_same_episode_ratio": same_episode,
        "v39_card_same_topic_ratio": same_topic,
        "v39_card_branch_entropy": branch_entropy,
        "v39_card_bu_td_agreement": bu_td_agreement,
        "v39_card_hyp_route_share": hyp_route_share,
        "v39_card_avg_hyp_distance": avg_distance,
        "v39_card_max_hyp_distance": max_distance,
        "anchor_distances": anchor_distances,
    }


def load_hyperbolic_points(
    embedding_path: Path,
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    payload = np.load(embedding_path, allow_pickle=False)
    node_ids = [str(item) for item in payload["node_ids"]]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    router = load_hyperbolic_router(checkpoint_path, device)
    rows = []
    router.model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start : start + batch_size], dtype=torch.float32, device=device)
            rows.append(router.model(batch).detach().cpu().numpy())
    points = np.concatenate(rows, axis=0).astype(np.float32)
    return dict(zip(node_ids, points))


def hierarchy_maps(graph) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    fact_event = {}
    event_episode = {}
    episode_topic = {}
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role = edge.metadata.get("hierarchy_v3_3")
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            fact_event[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.EVENT and role == "event_episode":
            event_episode[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "episode_topic":
            episode_topic[edge.src] = edge.dst
    return fact_event, event_episode, episode_topic


def load_card_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line).get("record", {})
            if record.get("question_id"):
                records[str(record["question_id"])] = record
    return records


def load_cardce_scores(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    for item in read_json(path):
        qid = str(item["question_id"])
        for evidence_path in item.get("paths", []):
            metadata = evidence_path.get("metadata", {})
            card_id = str(metadata.get("nary_hyperedge_id") or "")
            if not card_id:
                continue
            score = float(
                evidence_path.get("scores", {}).get(
                    "v3_9_card_ce",
                    metadata.get("v3_9_cardce_score", 0.0),
                )
                or 0.0
            )
            scores[(qid, card_id)] = max(scores.get((qid, card_id), float("-inf")), score)
    return scores


def run_clean_search(args, annotated_path: Path) -> None:
    selector_script = Path(__file__).resolve().parent / "88_run_nary_completion_selector_cv_v3_6c.py"
    spec = importlib.util.spec_from_file_location("nary_selector_search", selector_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {selector_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    graph_v2 = module.load_graph_v2_selector_module()
    base = graph_v2.dedupe(
        graph_v2.RETRIEVAL_FEATURES
        + graph_v2.GRAPH_V2_FEATURES
        + graph_v2.ROUTE_ORIGIN_V1_FEATURES
        + graph_v2.ROUTE_AGREEMENT_V1_FEATURES
    )
    membership = ["is_nary_completion"]
    quality = [
        "nary_hyperedge_size",
        "nary_hyperedge_confidence",
        "nary_same_hyperedge_count_in_candidate_pool",
        "nary_role_coverage_potential",
    ]
    type_role = [
        "nary_type_change",
        "nary_type_preference",
        "nary_type_state",
        "nary_type_plan_constraint",
        "nary_role_old_state",
        "nary_role_new_state",
        "nary_role_preference_value",
        "nary_role_polarity",
        "nary_role_state_value",
        "nary_role_plan_goal",
        "nary_role_constraint",
        "nary_role_temporal_scope",
        "nary_role_reason_or_trigger",
        "nary_role_exception",
        "nary_role_context",
    ]
    variants = {
        "clean_base": base,
        "clean_membership": graph_v2.dedupe(base + membership),
        "clean_membership_quality": graph_v2.dedupe(base + membership + quality),
        "clean_membership_quality_cardce": graph_v2.dedupe(base + membership + quality + CARDCE_FEATURES),
        "clean_membership_quality_structural_geometry": graph_v2.dedupe(
            base + membership + quality + STRUCTURAL_GEOMETRY_FEATURES
        ),
        "clean_membership_quality_hyp_distance": graph_v2.dedupe(
            base + membership + quality + HYPERBOLIC_DISTANCE_FEATURES
        ),
        "clean_membership_quality_geometry": graph_v2.dedupe(base + membership + quality + GEOMETRY_FEATURES),
        "clean_membership_quality_cardce_geometry": graph_v2.dedupe(
            base + membership + quality + CARDCE_FEATURES + GEOMETRY_FEATURES
        ),
        "clean_plus_type_role": graph_v2.dedupe(
            base + membership + quality + CARDCE_FEATURES + GEOMETRY_FEATURES + type_role
        ),
    }
    run_selector_cv(
        module=module,
        graph_path="outputs/graphs/locomo_graph_v3_6b_qwen_all.json",
        candidates_path=annotated_path,
        output_dir=resolve_path(args.output_dir),
        variants=variants,
        final_topk=args.final_topk,
        n_estimators=args.n_estimators,
    )


def run_selector_cv(
    *,
    module,
    graph_path: str,
    candidates_path: Path,
    output_dir: Path,
    variants: dict[str, list[str]],
    final_topk: int,
    n_estimators: int,
) -> None:
    cv = module.load_cv_helpers()
    graph = JsonGraphStore().load(resolve_path(graph_path))
    candidates = read_json(candidates_path)
    feature_cache = cv.build_feature_cache(graph, candidates)
    conversations = cv.ordered_conversations(candidates)
    aggregate = defaultdict(list)
    fold_rows = []
    for fold_index, test_conversation in enumerate(conversations):
        train_items = [
            item for item in candidates
            if cv.conversation_id_from_question(item["question_id"]) != test_conversation
        ]
        test_items = [
            item for item in candidates
            if cv.conversation_id_from_question(item["question_id"]) == test_conversation
        ]
        train_rows_all, train_labels, train_groups, _ = cv.flatten_feature_cache(
            feature_cache,
            [item["question_id"] for item in train_items],
        )
        fold = {"fold": fold_index, "test_conversation": test_conversation, "methods": {}}
        for method, feature_names in variants.items():
            model = module.train_lightgbm_ranker(
                module.project_feature_rows(train_rows_all, feature_names),
                train_labels,
                train_groups,
                n_estimators=n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                n_jobs=8,
                random_state=13 + fold_index,
            )
            scores = cv.score_items_with_cache(test_items, model, feature_cache, feature_names)
            reranked = cv.rerank_items_with_scores(test_items, scores, final_topk)
            evaluation = cv.evaluate_items(graph, reranked, final_topk, method)
            fold["methods"][method] = evaluation["summary"]
            aggregate[method].extend(evaluation["per_question"])
            fold_dir = output_dir / f"fold_{fold_index:02d}_{test_conversation}"
            write_json(reranked, fold_dir / f"{method}_paths.json")
            write_json(evaluation, fold_dir / f"{method}_eval.json")
        write_json(fold, output_dir / f"fold_{fold_index:02d}_{test_conversation}" / "fold_summary.json")
        fold_rows.append(fold)
        print(f"clean search fold={fold_index} test={test_conversation}", flush=True)

    summary = {
        "method": "V3.9 clean feature search",
        "k": final_topk,
        "candidates": str(candidates_path),
        "variants": variants,
        "aggregate": {
            method: cv.summarize_rows(rows)
            for method, rows in aggregate.items()
        },
        "folds": fold_rows,
    }
    write_json(summary, output_dir / "clean_feature_search_summary.json")
    (output_dir / "clean_feature_search_summary.md").write_text(
        render_summary(summary),
        encoding="utf-8",
    )
    print(render_summary(summary))


def render_summary(summary: dict) -> str:
    lines = [
        "# V3.9 Clean Feature Search",
        "",
        f"K: {summary['k']}",
        "",
        "| Method | Features | Hit | Recall | FullCover |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in summary["aggregate"].items():
        lines.append(
            f"| {method} | {len(summary['variants'][method])} | "
            f"{row['hit']:.4f} | {row['recall']:.4f} | {row['full_cover']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def route_flags(path: dict) -> dict[str, bool]:
    metadata = path.get("metadata", {})
    source = str(metadata.get("route_source") or metadata.get("candidate_source") or "")
    top_down = any(token in source for token in ("eu_event", "eu_topic", "hyp_event", "hyp_topic"))
    return {
        "bottom_up": "bottom_up" in source,
        "top_down": top_down,
        "hyp": any(token in source for token in ("hyp_event", "hyp_topic", "hyp_bottom")),
    }


def same_parent(left: str, right: str, mapping: dict[str, str]) -> bool:
    return bool(left and right and mapping.get(left) and mapping.get(left) == mapping.get(right))


def pair_ratio(pairs: list[tuple[str, str]], predicate) -> float:
    if not pairs:
        return 0.0
    return sum(float(predicate(left, right)) for left, right in pairs) / len(pairs)


def normalized_entropy(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    entropy = -sum((count / len(values)) * math.log(count / len(values)) for count in counts.values())
    return entropy / math.log(len(values))


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def dedupe(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


if __name__ == "__main__":
    main()
