from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.models.graph_v2_hyperbolic import GraphV2HyperbolicMapper, lorentz_radius
from hytopomem.retrieval.topdown_semantic_retriever import SentenceTransformerEncoder, default_embedder
from hytopomem.training.graph_v2_hyperbolic_losses import (
    BranchBatch,
    StructureBatch,
    branch_triplet_loss,
    edge_contrastive_loss,
    radial_order_loss,
    sample_branch_batch,
    sample_structure_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--hierarchy-version", choices=["v2", "v3", "v3_3"], default="v2")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v2_minilm_fact_event_topic.npz")
    parser.add_argument("--output", default="outputs/models/graph_v2_lorentz_router/minilm_structure_router.pt")
    parser.add_argument("--tangent-dim", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--branch-batch-size", type=int, default=768)
    parser.add_argument("--num-negatives", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--radial-margin", type=float, default=0.08)
    parser.add_argument("--branch-margin", type=float, default=0.15)
    parser.add_argument("--edge-alpha", type=float, default=1.0)
    parser.add_argument("--radial-alpha", type=float, default=0.35)
    parser.add_argument("--branch-alpha", type=float, default=0.25)
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="random")
    parser.add_argument("--disable-pca-init", action="store_true")
    parser.add_argument("--disable-radial", action="store_true")
    parser.add_argument("--disable-branch", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    started = time.perf_counter()
    graph = JsonGraphStore().load(resolve_path(args.graph))
    node_ids, node_types, texts = build_training_nodes(graph)
    embeddings = load_or_encode_node_embeddings(
        node_ids=node_ids,
        texts=texts,
        model_name_or_path=args.embedder,
        cache_path=resolve_path(args.embedding_cache),
        batch_size=args.embedding_batch_size,
    )
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    hierarchy_key = f"hierarchy_{args.hierarchy_version}"
    edges, event_to_facts = build_training_edges(graph, node_to_idx, hierarchy_key=hierarchy_key)
    if not edges:
        raise RuntimeError("no Graph v2 hierarchy edges found")
    fact_pool = [idx for idx, node_type in enumerate(node_types) if node_type == NodeType.FACT]
    parent_pools = {
        "EVENT": [idx for idx, node_type in enumerate(node_types) if node_type == NodeType.EVENT],
        "TOPIC": [idx for idx, node_type in enumerate(node_types) if node_type == NodeType.TOPIC],
    }
    hard_index = (
        build_hard_negative_index(graph, node_to_idx, hierarchy_key=hierarchy_key)
        if args.negative_strategy == "hard"
        else {}
    )
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    model = GraphV2HyperbolicMapper(input_dim=x.shape[1], tangent_dim=args.tangent_dim).to(device)
    if not args.disable_pca_init:
        initialize_mapper_with_pca(model, x)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(
        f"device={device} nodes={len(node_ids)} dim={x.shape[1]} tangent_dim={args.tangent_dim} "
        f"edges={len(edges)} fact_pool={len(fact_pool)} event_groups={len(event_to_facts)} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        totals = {"edge": 0.0, "radial": 0.0, "branch": 0.0, "loss": 0.0}
        for _step in range(args.steps_per_epoch):
            if args.negative_strategy == "hard":
                batch = sample_hard_structure_batch(
                    edges=edges,
                    parent_pools=parent_pools,
                    hard_index=hard_index,
                    batch_size=args.batch_size,
                    num_negatives=args.num_negatives,
                    device=device,
                )
            else:
                batch = sample_structure_batch(
                    edges=edges,
                    parent_pools=parent_pools,
                    batch_size=args.batch_size,
                    num_negatives=args.num_negatives,
                    device=device,
                )
            optimizer.zero_grad(set_to_none=True)
            child_points = model(x[batch.child_idx])
            parent_points = model(x[batch.parent_idx])
            negative_parent_points = model(x[batch.negative_parent_idx.reshape(-1)]).reshape(
                batch.negative_parent_idx.shape[0],
                batch.negative_parent_idx.shape[1],
                -1,
            )
            edge_loss = edge_contrastive_loss(child_points, parent_points, negative_parent_points, args.tau)
            radial_loss = (
                torch.tensor(0.0, device=device)
                if args.disable_radial
                else radial_order_loss(child_points, parent_points, args.radial_margin)
            )
            branch_loss = torch.tensor(0.0, device=device)
            if not args.disable_branch:
                if args.negative_strategy == "hard":
                    branch_batch = sample_hard_branch_batch(
                        event_to_facts=event_to_facts,
                        fact_pool=fact_pool,
                        hard_index=hard_index,
                        batch_size=args.branch_batch_size,
                        device=device,
                    )
                else:
                    branch_batch = sample_branch_batch(
                        event_to_facts=event_to_facts,
                        fact_pool=fact_pool,
                        batch_size=args.branch_batch_size,
                        device=device,
                    )
                if branch_batch is not None:
                    branch_loss = branch_triplet_loss(
                        model(x[branch_batch.anchor_idx]),
                        model(x[branch_batch.positive_idx]),
                        model(x[branch_batch.negative_idx]),
                        args.branch_margin,
                    )
            loss = args.edge_alpha * edge_loss + args.radial_alpha * radial_loss + args.branch_alpha * branch_loss
            loss.backward()
            optimizer.step()
            totals["edge"] += float(edge_loss.detach().cpu())
            totals["radial"] += float(radial_loss.detach().cpu())
            totals["branch"] += float(branch_loss.detach().cpu())
            totals["loss"] += float(loss.detach().cpu())
        denom = float(args.steps_per_epoch)
        print(
            f"epoch={epoch} loss={totals['loss']/denom:.4f} "
            f"edge={totals['edge']/denom:.4f} radial={totals['radial']/denom:.4f} "
            f"branch={totals['branch']/denom:.4f}",
            flush=True,
        )

    radius_summary = compute_radius_summary(model, x, node_types)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "tangent_dim": args.tangent_dim,
            "scale": model.scale,
            "metadata": {
                "graph": str(resolve_path(args.graph)),
                "hierarchy_version": args.hierarchy_version,
                "embedder": args.embedder,
                "embedding_cache": str(resolve_path(args.embedding_cache)),
                "training": "graph_v2_minilm_lorentz_router_global_no_qa",
                "negative_strategy": args.negative_strategy,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_size": args.batch_size,
                "branch_batch_size": args.branch_batch_size,
                "num_negatives": args.num_negatives,
                "lr": args.lr,
                "tau": args.tau,
                "radial_margin": args.radial_margin,
                "branch_margin": args.branch_margin,
                "edge_alpha": args.edge_alpha,
                "radial_alpha": 0.0 if args.disable_radial else args.radial_alpha,
                "branch_alpha": 0.0 if args.disable_branch else args.branch_alpha,
                "pca_init": not args.disable_pca_init,
                "seed": args.seed,
                "num_nodes": len(node_ids),
                "num_edges": len(edges),
                "radius_summary": radius_summary,
            },
        },
        output_path,
    )
    print(f"radius_summary={radius_summary}", flush=True)
    print(f"wrote {output_path}", flush=True)


def build_training_nodes(graph: MemoryGraph) -> tuple[list[str], list[NodeType], list[str]]:
    rows = [
        (node_id, node.type, node.text)
        for node_id, node in graph.nodes.items()
        if node.type in {NodeType.FACT, NodeType.EVENT, NodeType.TOPIC}
    ]
    rows.sort(key=lambda item: (item[1].value, item[0]))
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def load_or_encode_node_embeddings(
    *,
    node_ids: list[str],
    texts: list[str],
    model_name_or_path: str,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(item) for item in payload["node_ids"]]
        cached_model = str(payload["model"][0]) if "model" in payload.files else ""
        if cached_ids == node_ids and cached_model == model_name_or_path:
            return np.asarray(payload["embeddings"], dtype=np.float32)
    encoder = SentenceTransformerEncoder(model_name_or_path, batch_size=batch_size)
    embeddings = encoder.encode(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        node_ids=np.asarray(node_ids),
        embeddings=embeddings,
        model=np.asarray([model_name_or_path]),
    )
    return embeddings


def build_training_edges(
    graph: MemoryGraph,
    node_to_idx: dict[str, int],
    *,
    hierarchy_key: str = "hierarchy_v2",
) -> tuple[list[tuple[int, int, str]], dict[int, list[int]]]:
    edges: list[tuple[int, int, str]] = []
    event_to_facts: dict[int, list[int]] = {}
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role = edge.metadata.get(hierarchy_key)
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            child_idx = node_to_idx[edge.src]
            parent_idx = node_to_idx[edge.dst]
            edges.append((child_idx, parent_idx, "EVENT"))
            event_to_facts.setdefault(parent_idx, []).append(child_idx)
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "event_topic":
            edges.append((node_to_idx[edge.src], node_to_idx[edge.dst], "TOPIC"))
        elif hierarchy_key == "hierarchy_v3_3" and src.type == NodeType.EVENT and dst.type == NodeType.EVENT and role == "event_episode":
            edges.append((node_to_idx[edge.src], node_to_idx[edge.dst], "EVENT"))
        elif hierarchy_key == "hierarchy_v3_3" and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "episode_topic":
            edges.append((node_to_idx[edge.src], node_to_idx[edge.dst], "TOPIC"))
    return edges, event_to_facts


def build_hard_negative_index(
    graph: MemoryGraph,
    node_to_idx: dict[str, int],
    *,
    hierarchy_key: str = "hierarchy_v2",
) -> dict[str, dict[int, list[int]]]:
    fact_to_event: dict[str, str] = {}
    event_to_topic: dict[str, str] = {}
    event_to_episode: dict[str, str] = {}
    episode_to_topic: dict[str, str] = {}
    event_to_facts: dict[str, list[str]] = {}
    topic_to_events: dict[str, list[str]] = {}
    episode_to_events: dict[str, list[str]] = {}
    topic_to_episodes: dict[str, list[str]] = {}
    conv_to_topics: dict[str, list[str]] = {}
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
            event_to_facts.setdefault(edge.dst, []).append(edge.src)
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "event_topic":
            event_to_topic[edge.src] = edge.dst
            topic_to_events.setdefault(edge.dst, []).append(edge.src)
            conv_id = str(graph.nodes[edge.dst].metadata.get("conversation_id", "") or "").strip()
            if conv_id:
                conv_to_topics.setdefault(conv_id, []).append(edge.dst)
        elif hierarchy_key == "hierarchy_v3_3" and src.type == NodeType.EVENT and dst.type == NodeType.EVENT and role == "event_episode":
            event_to_episode[edge.src] = edge.dst
            episode_to_events.setdefault(edge.dst, []).append(edge.src)
        elif hierarchy_key == "hierarchy_v3_3" and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role == "episode_topic":
            episode_to_topic[edge.src] = edge.dst
            topic_to_episodes.setdefault(edge.dst, []).append(edge.src)
            conv_id = str(graph.nodes[edge.dst].metadata.get("conversation_id", "") or "").strip()
            if conv_id:
                conv_to_topics.setdefault(conv_id, []).append(edge.dst)

    fact_hard_events: dict[int, list[int]] = {}
    fact_hard_facts: dict[int, list[int]] = {}
    for fact_id, event_id in fact_to_event.items():
        topic_id = event_to_topic.get(event_id)
        sibling_events: list[str] = []
        if hierarchy_key == "hierarchy_v3_3":
            episode_id = event_to_episode.get(event_id)
            if episode_id is None:
                continue
            sibling_events = [item for item in episode_to_events.get(episode_id, []) if item != event_id]
            topic_id = episode_to_topic.get(episode_id)
        if topic_id is None:
            continue
        if not sibling_events:
            sibling_events = [item for item in topic_to_events.get(topic_id, []) if item != event_id]
        hard_event_indices = [node_to_idx[item] for item in sibling_events if item in node_to_idx]
        if hard_event_indices and fact_id in node_to_idx:
            fact_hard_events[node_to_idx[fact_id]] = hard_event_indices
        sibling_facts: list[int] = []
        for sibling_event in sibling_events:
            sibling_facts.extend(node_to_idx[item] for item in event_to_facts.get(sibling_event, []) if item in node_to_idx)
        if sibling_facts and fact_id in node_to_idx:
            fact_hard_facts[node_to_idx[fact_id]] = sibling_facts

    event_hard_topics: dict[int, list[int]] = {}
    event_hard_episodes: dict[int, list[int]] = {}
    if hierarchy_key == "hierarchy_v3_3":
        for event_id, episode_id in event_to_episode.items():
            topic_id = episode_to_topic.get(episode_id)
            if not topic_id:
                continue
            sibling_episodes = [item for item in topic_to_episodes.get(topic_id, []) if item != episode_id]
            hard_episode_indices = [node_to_idx[item] for item in sibling_episodes if item in node_to_idx]
            if hard_episode_indices and event_id in node_to_idx:
                event_hard_episodes[node_to_idx[event_id]] = hard_episode_indices
        episode_topic_items = episode_to_topic.items()
    else:
        episode_topic_items = event_to_topic.items()
    for event_id, topic_id in episode_topic_items:
        topic_node = graph.nodes.get(topic_id)
        if topic_node is None:
            continue
        conv_id = str(topic_node.metadata.get("conversation_id", "") or "").strip()
        sibling_topics = [item for item in conv_to_topics.get(conv_id, []) if item != topic_id]
        hard_topic_indices = [node_to_idx[item] for item in sibling_topics if item in node_to_idx]
        if hard_topic_indices and event_id in node_to_idx:
            event_hard_topics[node_to_idx[event_id]] = hard_topic_indices

    return {
        "fact_hard_events": fact_hard_events,
        "fact_hard_facts": fact_hard_facts,
        "event_hard_topics": event_hard_topics,
        "event_hard_episodes": event_hard_episodes,
    }


def sample_hard_structure_batch(
    *,
    edges: list[tuple[int, int, str]],
    parent_pools: dict[str, list[int]],
    hard_index: dict[str, dict[int, list[int]]],
    batch_size: int,
    num_negatives: int,
    device: torch.device,
) -> StructureBatch:
    sampled = random.choices(edges, k=min(batch_size, len(edges)))
    child_idx = torch.tensor([item[0] for item in sampled], dtype=torch.long, device=device)
    parent_idx = torch.tensor([item[1] for item in sampled], dtype=torch.long, device=device)
    negatives = []
    for child, parent, parent_type in sampled:
        if parent_type == "EVENT":
            hard_pool = hard_index.get("fact_hard_events", {}).get(child, [])
            if not hard_pool:
                hard_pool = hard_index.get("event_hard_episodes", {}).get(child, [])
        elif parent_type == "TOPIC":
            hard_pool = hard_index.get("event_hard_topics", {}).get(child, [])
        else:
            hard_pool = []
        global_pool = parent_pools[parent_type]
        row = []
        hard_quota = max(1, num_negatives // 2) if hard_pool else 0
        while len(row) < hard_quota:
            candidate = random.choice(hard_pool)
            if candidate != parent:
                row.append(candidate)
        while len(row) < num_negatives:
            pool = hard_pool if hard_pool and random.random() < 0.35 else global_pool
            candidate = random.choice(pool)
            if candidate != parent:
                row.append(candidate)
        negatives.append(row)
    return StructureBatch(
        child_idx=child_idx,
        parent_idx=parent_idx,
        negative_parent_idx=torch.tensor(negatives, dtype=torch.long, device=device),
    )


def sample_hard_branch_batch(
    *,
    event_to_facts: dict[int, list[int]],
    fact_pool: list[int],
    hard_index: dict[str, dict[int, list[int]]],
    batch_size: int,
    device: torch.device,
) -> BranchBatch | None:
    eligible_events = [event for event, facts in event_to_facts.items() if len(facts) >= 2]
    if not eligible_events:
        return None
    hard_facts = hard_index.get("fact_hard_facts", {})
    anchors = []
    positives = []
    negatives = []
    for _ in range(batch_size):
        event = random.choice(eligible_events)
        facts = event_to_facts[event]
        anchor, positive = random.sample(facts, 2)
        pool = hard_facts.get(anchor, [])
        negative = random.choice(pool) if pool and random.random() < 0.8 else random.choice(fact_pool)
        tries = 0
        while negative in facts and tries < 20:
            negative = random.choice(fact_pool)
            tries += 1
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    return BranchBatch(
        anchor_idx=torch.tensor(anchors, dtype=torch.long, device=device),
        positive_idx=torch.tensor(positives, dtype=torch.long, device=device),
        negative_idx=torch.tensor(negatives, dtype=torch.long, device=device),
    )


def initialize_mapper_with_pca(model: GraphV2HyperbolicMapper, x: torch.Tensor) -> None:
    with torch.no_grad():
        centered = x - x.mean(dim=0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: model.linear.out_features]
        model.linear.weight.zero_()
        model.linear.weight[: components.shape[0], : components.shape[1]] = components
        model.linear.bias.zero_()


def compute_radius_summary(model: GraphV2HyperbolicMapper, x: torch.Tensor, node_types: list[NodeType]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        radii = lorentz_radius(model(x)).detach().cpu().numpy()
    for node_type in [NodeType.TOPIC, NodeType.EVENT, NodeType.FACT]:
        values = [float(radius) for radius, item_type in zip(radii, node_types) if item_type == node_type]
        output[node_type.value] = {
            "count": float(len(values)),
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
        }
    return output


if __name__ == "__main__":
    main()
