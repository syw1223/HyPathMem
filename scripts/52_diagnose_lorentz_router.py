from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from common import resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.retrieval.hyperbolic_topdown_retriever import load_hyperbolic_router, lorentz_distance_numpy
from hytopomem.retrieval.topdown_semantic_retriever import SentenceTransformerEncoder, default_embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--hierarchy-version", choices=["v2", "v3", "v3_3"], default="v2")
    parser.add_argument("--checkpoint", default="outputs/models/graph_v2_lorentz_router/minilm_structure_router.pt")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v2_minilm_fact_event_topic.npz")
    parser.add_argument("--output-json", default="outputs/eval/hyperbolic_router_diagnostics.json")
    parser.add_argument("--output-md", default="outputs/eval/hyperbolic_router_diagnostics.md")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    node_ids, node_types, texts = build_node_rows(graph)
    embeddings = load_or_encode_embeddings(
        node_ids=node_ids,
        texts=texts,
        model_name_or_path=args.embedder,
        cache_path=resolve_path(args.embedding_cache),
    )
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    router = load_hyperbolic_router(resolve_path(args.checkpoint), device)
    points = map_embeddings(router.model, embeddings, device=device, batch_size=args.batch_size)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    fact_to_event, event_to_topic = build_hierarchy_maps(
        graph,
        hierarchy_key=f"hierarchy_{args.hierarchy_version}",
    )
    event_to_facts = invert(fact_to_event)
    topic_to_events = invert(event_to_topic)
    fact_to_topic = {fact_id: event_to_topic.get(event_id, "") for fact_id, event_id in fact_to_event.items()}

    payload = {
        "metadata": {
            "graph": str(resolve_path(args.graph)),
            "hierarchy_version": args.hierarchy_version,
            "checkpoint": str(resolve_path(args.checkpoint)),
            "checkpoint_metadata": router.metadata,
            "embedder": args.embedder,
            "samples": args.samples,
            "seed": args.seed,
        },
        "radius": radius_summary(points, node_types),
        "radial_order": radial_order_summary(points, node_to_idx, fact_to_event, event_to_topic),
        "edge_separation": edge_separation_summary(
            points,
            node_to_idx,
            fact_to_event,
            event_to_topic,
            topic_to_events,
            samples=args.samples,
        ),
        "branch_separation": branch_separation_summary(
            points,
            node_to_idx,
            event_to_facts,
            fact_to_topic,
            samples=args.samples,
        ),
    }
    write_json(payload, resolve_path(args.output_json))
    write_markdown(payload, resolve_path(args.output_md))
    print(f"wrote {resolve_path(args.output_json)}")
    print(f"wrote {resolve_path(args.output_md)}")
    print_markdown_summary(payload)


def build_node_rows(graph: MemoryGraph) -> tuple[list[str], list[NodeType], list[str]]:
    rows = [
        (node_id, node.type, node.text)
        for node_id, node in graph.nodes.items()
        if node.type in {NodeType.FACT, NodeType.EVENT, NodeType.TOPIC}
    ]
    rows.sort(key=lambda item: (item[1].value, item[0]))
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def load_or_encode_embeddings(
    *,
    node_ids: list[str],
    texts: list[str],
    model_name_or_path: str,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(item) for item in payload["node_ids"]]
        cached_model = str(payload["model"][0]) if "model" in payload.files else ""
        if cached_ids == node_ids and cached_model == model_name_or_path:
            return np.asarray(payload["embeddings"], dtype=np.float32)
    encoder = SentenceTransformerEncoder(model_name_or_path)
    embeddings = encoder.encode(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_ids=np.asarray(node_ids),
        embeddings=embeddings,
        model=np.asarray([model_name_or_path]),
    )
    return embeddings


def map_embeddings(model, embeddings: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[start : start + batch_size], dtype=torch.float32, device=device)
            rows.append(model(batch).detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def build_hierarchy_maps(
    graph: MemoryGraph,
    *,
    hierarchy_key: str = "hierarchy_v2",
) -> tuple[dict[str, str], dict[str, str]]:
    fact_to_event = {}
    event_to_topic = {}
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role = edge.metadata.get(hierarchy_key)
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            fact_to_event[edge.src] = edge.dst
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "event_topic":
            event_to_topic[edge.src] = edge.dst
    return fact_to_event, event_to_topic


def invert(mapping: dict[str, str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for child, parent in mapping.items():
        output.setdefault(parent, []).append(child)
    return output


def radius_summary(points: np.ndarray, node_types: list[NodeType]) -> dict[str, dict[str, float]]:
    radii = np.arccosh(np.clip(points[:, 0], 1.0 + 1e-7, None))
    output = {}
    for node_type in [NodeType.TOPIC, NodeType.EVENT, NodeType.FACT]:
        values = radii[np.asarray([item == node_type for item in node_types])]
        output[node_type.value] = describe(values)
    return output


def radial_order_summary(
    points: np.ndarray,
    node_to_idx: dict[str, int],
    fact_to_event: dict[str, str],
    event_to_topic: dict[str, str],
) -> dict[str, float]:
    radii = np.arccosh(np.clip(points[:, 0], 1.0 + 1e-7, None))
    fact_event = []
    event_topic = []
    chain = []
    for fact_id, event_id in fact_to_event.items():
        topic_id = event_to_topic.get(event_id)
        if fact_id not in node_to_idx or event_id not in node_to_idx:
            continue
        fact_radius = radii[node_to_idx[fact_id]]
        event_radius = radii[node_to_idx[event_id]]
        fact_event.append(fact_radius > event_radius)
        if topic_id and topic_id in node_to_idx:
            topic_radius = radii[node_to_idx[topic_id]]
            event_topic.append(event_radius > topic_radius)
            chain.append(topic_radius < event_radius < fact_radius)
    return {
        "fact_radius_gt_event": mean_bool(fact_event),
        "event_radius_gt_topic": mean_bool(event_topic),
        "topic_lt_event_lt_fact": mean_bool(chain),
        "num_fact_event": len(fact_event),
        "num_event_topic_chains": len(chain),
    }


def edge_separation_summary(
    points: np.ndarray,
    node_to_idx: dict[str, int],
    fact_to_event: dict[str, str],
    event_to_topic: dict[str, str],
    topic_to_events: dict[str, list[str]],
    *,
    samples: int,
) -> dict[str, dict[str, float]]:
    event_ids = sorted(set(event_to_topic))
    topic_ids = sorted(set(event_to_topic.values()))
    fact_edges = [(fact_id, event_id) for fact_id, event_id in fact_to_event.items() if fact_id in node_to_idx and event_id in node_to_idx]
    event_edges = [(event_id, topic_id) for event_id, topic_id in event_to_topic.items() if event_id in node_to_idx and topic_id in node_to_idx]
    random.shuffle(fact_edges)
    random.shuffle(event_edges)
    fact_edges = fact_edges[:samples]
    event_edges = event_edges[:samples]

    fact_true = []
    fact_random = []
    fact_true_for_hard = []
    fact_same_topic = []
    for fact_id, event_id in fact_edges:
        topic_id = event_to_topic.get(event_id, "")
        true_dist = distance(points, node_to_idx[fact_id], node_to_idx[event_id])
        random_event = random.choice(event_ids)
        fact_true.append(true_dist)
        fact_random.append(distance(points, node_to_idx[fact_id], node_to_idx[random_event]))
        alternatives = [item for item in topic_to_events.get(topic_id, []) if item != event_id and item in node_to_idx]
        if alternatives:
            hard_event = random.choice(alternatives)
            fact_true_for_hard.append(true_dist)
            fact_same_topic.append(distance(points, node_to_idx[fact_id], node_to_idx[hard_event]))

    event_true = []
    event_random = []
    event_true_for_hard = []
    event_same_conv = []
    topics_by_conv = bucket_by_conversation(topic_ids)
    for event_id, topic_id in event_edges:
        true_dist = distance(points, node_to_idx[event_id], node_to_idx[topic_id])
        random_topic = random.choice(topic_ids)
        event_true.append(true_dist)
        event_random.append(distance(points, node_to_idx[event_id], node_to_idx[random_topic]))
        conv_id = event_id.split(":", 1)[0]
        alternatives = [item for item in topics_by_conv.get(conv_id, []) if item != topic_id and item in node_to_idx]
        if alternatives:
            hard_topic = random.choice(alternatives)
            event_true_for_hard.append(true_dist)
            event_same_conv.append(distance(points, node_to_idx[event_id], node_to_idx[hard_topic]))

    return {
        "fact_true_event": describe(np.asarray(fact_true)),
        "fact_random_event": describe(np.asarray(fact_random)),
        "fact_same_topic_diff_event": describe(np.asarray(fact_same_topic)),
        "fact_true_lt_random_event": paired_less(fact_true, fact_random),
        "fact_true_lt_same_topic_diff_event": paired_less(fact_true_for_hard, fact_same_topic),
        "event_true_topic": describe(np.asarray(event_true)),
        "event_random_topic": describe(np.asarray(event_random)),
        "event_same_conv_diff_topic": describe(np.asarray(event_same_conv)),
        "event_true_lt_random_topic": paired_less(event_true, event_random),
        "event_true_lt_same_conv_diff_topic": paired_less(event_true_for_hard, event_same_conv),
    }


def branch_separation_summary(
    points: np.ndarray,
    node_to_idx: dict[str, int],
    event_to_facts: dict[str, list[str]],
    fact_to_topic: dict[str, str],
    *,
    samples: int,
) -> dict[str, dict[str, float]]:
    topic_to_facts = invert({fact_id: topic_id for fact_id, topic_id in fact_to_topic.items() if topic_id})
    eligible_events = [event_id for event_id, facts in event_to_facts.items() if len(facts) >= 2]
    same_event = []
    same_event_for_hard = []
    same_topic_diff_event = []
    same_topic_for_diff = []
    diff_topic = []
    topic_ids = sorted(topic_to_facts)
    for _ in range(samples):
        if not eligible_events:
            break
        event_id = random.choice(eligible_events)
        facts = [fact_id for fact_id in event_to_facts[event_id] if fact_id in node_to_idx]
        if len(facts) < 2:
            continue
        left, right = random.sample(facts, 2)
        same_event.append(distance(points, node_to_idx[left], node_to_idx[right]))
        topic_id = fact_to_topic.get(left, "")
        same_topic_candidates = [
            fact_id
            for fact_id in topic_to_facts.get(topic_id, [])
            if fact_id in node_to_idx and fact_id not in facts
        ]
        if same_topic_candidates:
            same_event_for_hard.append(same_event[-1])
            same_topic_diff_event.append(distance(points, node_to_idx[left], node_to_idx[random.choice(same_topic_candidates)]))
            different_topics = [item for item in topic_ids if item != topic_id and topic_to_facts.get(item)]
            if different_topics:
                negative_topic = random.choice(different_topics)
                negative_fact = random.choice(topic_to_facts[negative_topic])
                if negative_fact in node_to_idx:
                    same_topic_for_diff.append(same_topic_diff_event[-1])
                    diff_topic.append(distance(points, node_to_idx[left], node_to_idx[negative_fact]))
    return {
        "same_event_fact": describe(np.asarray(same_event)),
        "same_topic_diff_event_fact": describe(np.asarray(same_topic_diff_event)),
        "diff_topic_fact": describe(np.asarray(diff_topic)),
        "same_event_lt_same_topic_diff_event": paired_less(same_event_for_hard, same_topic_diff_event),
        "same_topic_lt_diff_topic": paired_less(same_topic_for_diff, diff_topic),
    }


def distance(points: np.ndarray, left: int, right: int) -> float:
    return float(lorentz_distance_numpy(points[[left]], points[right])[0])


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def mean_bool(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def paired_less(left: list[float], right: list[float]) -> float:
    count = min(len(left), len(right))
    if count == 0:
        return 0.0
    return float(np.mean(np.asarray(left[:count]) < np.asarray(right[:count])))


def bucket_by_conversation(node_ids: list[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for node_id in node_ids:
        output.setdefault(node_id.split(":", 1)[0], []).append(node_id)
    return output


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Hyperbolic Router Diagnostics",
        "",
        "## Radius",
        "",
        "| Type | Count | Mean | Std | P50 | P90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for node_type, row in payload["radius"].items():
        lines.append(
            f"| {node_type} | {row['count']:.0f} | {row['mean']:.4f} | {row['std']:.4f} | "
            f"{row['p50']:.4f} | {row['p90']:.4f} |"
        )
    lines.extend(["", "## Radial Order", ""])
    for key, value in payload["radial_order"].items():
        lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
    lines.extend(["", "## Edge Separation", ""])
    add_named_rows(lines, payload["edge_separation"])
    lines.extend(["", "## Branch Separation", ""])
    add_named_rows(lines, payload["branch_separation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_named_rows(lines: list[str], section: dict) -> None:
    for key, value in section.items():
        if isinstance(value, dict):
            lines.append(
                f"- {key}: count={value['count']:.0f}, mean={value['mean']:.4f}, "
                f"std={value['std']:.4f}, p50={value['p50']:.4f}, p90={value['p90']:.4f}"
            )
        else:
            lines.append(f"- {key}: {value:.4f}")


def print_markdown_summary(payload: dict) -> None:
    radius = payload["radius"]
    order = payload["radial_order"]
    print(
        "radius mean: "
        f"TOPIC={radius['TOPIC']['mean']:.4f} "
        f"EVENT={radius['EVENT']['mean']:.4f} "
        f"FACT={radius['FACT']['mean']:.4f}"
    )
    print(
        "radial order: "
        f"T<E={order['event_radius_gt_topic']:.4f} "
        f"E<F={order['fact_radius_gt_event']:.4f} "
        f"T<E<F={order['topic_lt_event_lt_fact']:.4f}"
    )
    edge = payload["edge_separation"]
    print(
        "edge separation: "
        f"fact_true_lt_random={edge['fact_true_lt_random_event']:.4f} "
        f"event_true_lt_random={edge['event_true_lt_random_topic']:.4f}"
    )


if __name__ == "__main__":
    main()
