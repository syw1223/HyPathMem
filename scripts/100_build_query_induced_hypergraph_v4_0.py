from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from common import read_json, resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.retrieval.hyperbolic_topdown_retriever import load_hyperbolic_router
from hytopomem.retrieval.topology_features import (
    TopologyFeatureIndex,
    build_example,
    feature_indices,
    with_cached_query_terms,
)


ROLE_NAMES = [
    "old_state",
    "new_state",
    "preference_value",
    "polarity",
    "state_value",
    "plan_goal",
    "constraint",
    "temporal_scope",
    "reason_or_trigger",
    "exception",
    "context",
    "location",
    "decision",
    "progress",
    "evidence",
]
ROLE_TO_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v3_6b_qwen_all.json")
    parser.add_argument(
        "--candidates",
        default="outputs/v3_9_query_cards/qwen3_card_annotated_base100_paths_v3_clean.json",
    )
    parser.add_argument("--card-cache", default="outputs/v3_9_query_cards/qwen3_cards_v3.jsonl")
    parser.add_argument(
        "--cardce-paths",
        default="outputs/eval/v3_9_cardce_guided_ctx50/cardce_guided_topk_paths.json",
    )
    parser.add_argument("--output", default="outputs/v4_0/query_induced_hypergraphs_ctx50.pkl.gz")
    parser.add_argument("--include-lorentz", action="store_true")
    parser.add_argument(
        "--embedding-cache",
        default="outputs/embeddings/graph_v3_3_episode_minilm_fact_event_episode_topic.npz",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/models/graph_v2_lorentz_router/minilm_structure_router_v3_3_episode_hardneg.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=2048)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    items = read_json(resolve_path(args.candidates))
    card_records = load_card_records(resolve_path(args.card_cache))
    cardce_scores = load_cardce_scores(resolve_path(args.cardce_paths))
    base_feature_names = mainline_base_features()
    selected_indices = feature_indices(base_feature_names)
    topology_index = TopologyFeatureIndex.from_graph(graph)
    point_map = {}
    if args.include_lorentz:
        point_map = load_hyperbolic_points(
            resolve_path(args.embedding_cache),
            resolve_path(args.checkpoint),
            args.device,
            args.embedding_batch_size,
        )

    examples = []
    totals = {"facts": 0, "positive_facts": 0, "cards": 0, "memberships": 0, "positive_cards": 0}
    for item_index, item in enumerate(items, start=1):
        item_with_terms = with_cached_query_terms(item)
        fact_ids = []
        fact_features = []
        fact_labels = []
        for rank, path in enumerate(item.get("paths", []), start=1):
            fact_id = evidence_node_id(path)
            if not fact_id:
                continue
            example = build_example(graph, item_with_terms, path, rank, topology_index)
            fact_ids.append(fact_id)
            fact_features.append([float(example.features[index]) for index in selected_indices])
            fact_labels.append(int(example.label))
        fact_to_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
        fact_lorentz = None
        fact_lorentz_mask = None
        if args.include_lorentz:
            point_dim = next(iter(point_map.values())).shape[0]
            origin = np.zeros(point_dim, dtype=np.float32)
            origin[0] = 1.0
            fact_lorentz = np.stack([point_map.get(fact_id, origin) for fact_id in fact_ids])
            fact_lorentz_mask = np.asarray(
                [float(fact_id in point_map) for fact_id in fact_ids],
                dtype=np.float32,
            )

        cards = []
        for card_rank, card in enumerate(card_records.get(str(item["question_id"]), {}).get("cards", []), start=1):
            card_id = f"query_card:{card_rank:02d}"
            members = membership_rows(card, fact_to_index)
            if len(members) < 2:
                continue
            card_label = int(any(fact_labels[row["fact_index"]] for row in members))
            cards.append(
                {
                    "card_id": card_id,
                    "features": np.asarray(
                        [
                            float(card.get("confidence", 0.0)),
                            np.log1p(len({row["fact_index"] for row in members})),
                            len(card.get("roles") or {}) / max(len(ROLE_NAMES), 1),
                            float(cardce_scores.get((str(item["question_id"]), card_id), 0.0)),
                        ],
                        dtype=np.float32,
                    ),
                    "label": card_label,
                    "members": members,
                    "type": str(card.get("type") or "none"),
                }
            )
            totals["memberships"] += len(members)
            totals["positive_cards"] += card_label

        examples.append(
            {
                "question_id": str(item["question_id"]),
                "conversation_id": conversation_id(str(item["question_id"])),
                "fact_ids": fact_ids,
                "fact_features": np.asarray(fact_features, dtype=np.float64),
                "fact_labels": np.asarray(fact_labels, dtype=np.float32),
                "fact_lorentz": fact_lorentz,
                "fact_lorentz_mask": fact_lorentz_mask,
                "cards": cards,
            }
        )
        totals["facts"] += len(fact_ids)
        totals["positive_facts"] += sum(fact_labels)
        totals["cards"] += len(cards)
        if item_index % 250 == 0 or item_index == len(items):
            print(
                f"hypergraphs {item_index}/{len(items)} facts={totals['facts']} "
                f"cards={totals['cards']} memberships={totals['memberships']}",
                flush=True,
            )

    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "candidates": str(resolve_path(args.candidates)),
            "card_cache": str(resolve_path(args.card_cache)),
            "cardce_paths": str(resolve_path(args.cardce_paths)),
            "base_feature_names": base_feature_names,
            "card_feature_names": ["confidence", "log_size", "normalized_role_count", "card_ce"],
            "role_names": ROLE_NAMES,
            "lorentz_dim": (
                int(next(iter(point_map.values())).shape[0])
                if point_map
                else 0
            ),
            "embedding_cache": str(resolve_path(args.embedding_cache)) if args.include_lorentz else "",
            "checkpoint": str(resolve_path(args.checkpoint)) if args.include_lorentz else "",
            "totals": totals,
        },
        "examples": examples,
    }
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb", compresslevel=5) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(payload["metadata"], indent=2))
    print(f"wrote {output}")


def membership_rows(card: dict, fact_to_index: dict[str, int]) -> list[dict]:
    roles_by_fact: dict[str, np.ndarray] = {}
    for role_name, role_payload in (card.get("roles") or {}).items():
        role = normalize_role(role_name)
        role_index = ROLE_TO_INDEX.get(role, ROLE_TO_INDEX["evidence"])
        if not isinstance(role_payload, dict):
            continue
        for fact_id in role_payload.get("fact_ids", []) or []:
            fact_id = str(fact_id)
            if fact_id not in fact_to_index:
                continue
            roles_by_fact.setdefault(fact_id, np.zeros(len(ROLE_NAMES), dtype=np.float32))[role_index] = 1.0
    for fact_id in card.get("support_facts", []) or []:
        fact_id = str(fact_id)
        if fact_id in fact_to_index:
            roles_by_fact.setdefault(fact_id, np.zeros(len(ROLE_NAMES), dtype=np.float32))
    rows = []
    for fact_id, role_vector in roles_by_fact.items():
        if not role_vector.any():
            role_vector[ROLE_TO_INDEX["evidence"]] = 1.0
        rows.append({"fact_index": fact_to_index[fact_id], "roles": role_vector})
    rows.sort(key=lambda row: row["fact_index"])
    return rows


def mainline_base_features() -> list[str]:
    module = load_module("41_run_graph_v2_selector_cv.py", "graph_v2_selector_for_v4")
    return module.dedupe(
        module.RETRIEVAL_FEATURES
        + module.GRAPH_V2_FEATURES
        + module.ROUTE_ORIGIN_V1_FEATURES
        + module.ROUTE_AGREEMENT_V1_FEATURES
        + [
            "is_nary_completion",
            "nary_hyperedge_size",
            "nary_hyperedge_confidence",
            "nary_same_hyperedge_count_in_candidate_pool",
            "nary_role_coverage_potential",
        ]
    )


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


def load_hyperbolic_points(
    embedding_path: Path,
    checkpoint_path: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    payload = np.load(embedding_path, allow_pickle=False)
    node_ids = [str(item) for item in payload["node_ids"]]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    router = load_hyperbolic_router(checkpoint_path, device)
    rows = []
    router.model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(
                embeddings[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            rows.append(router.model(batch).detach().cpu().numpy())
    points = np.concatenate(rows, axis=0).astype(np.float32)
    return dict(zip(node_ids, points))


def evidence_node_id(path: dict) -> str:
    metadata = path.get("metadata", {})
    if metadata.get("evidence_node_id"):
        return str(metadata["evidence_node_id"])
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def normalize_role(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def conversation_id(question_id: str) -> str:
    return question_id.split(":", 1)[0]


def load_module(filename: str, name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
