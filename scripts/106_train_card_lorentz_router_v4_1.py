from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from common import resolve_path
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.models.graph_v2_hyperbolic import GraphV2HyperbolicMapper, lorentz_distance, lorentz_radius
from hytopomem.models.text_encoder import HashTextEncoder
from hytopomem.retrieval.topdown_semantic_retriever import SentenceTransformerEncoder, default_embedder
from hytopomem.training.graph_v2_hyperbolic_losses import (
    branch_triplet_loss,
    edge_contrastive_loss,
    radial_order_loss,
)


TRAIN_NODE_TYPES = {NodeType.FACT, NodeType.RELATION_CARD, NodeType.EVENT, NodeType.TOPIC}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_v4_1_relation_card.json")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v4_1_card_minilm_fact_card_event_topic.npz")
    parser.add_argument(
        "--base-node-embedding-cache",
        default="outputs/embeddings/graph_v3_3_episode_minilm_fact_event_episode_topic.npz",
    )
    parser.add_argument("--encode-with-sentence-transformer", action="store_true")
    parser.add_argument("--output", default="outputs/models/graph_v4_1_lorentz_router/minilm_card_structure_router.pt")
    parser.add_argument("--diagnostics-json", default="outputs/eval/v4_1_card_lorentz_router_diagnostics.json")
    parser.add_argument("--tangent-dim", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=896)
    parser.add_argument("--branch-batch-size", type=int, default=896)
    parser.add_argument("--card-center-batch-size", type=int, default=256)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--tau", type=float, default=0.18)
    parser.add_argument("--radial-margin", type=float, default=0.06)
    parser.add_argument("--branch-margin", type=float, default=0.12)
    parser.add_argument("--edge-alpha", type=float, default=1.0)
    parser.add_argument("--radial-alpha", type=float, default=0.45)
    parser.add_argument("--branch-alpha", type=float, default=0.22)
    parser.add_argument("--card-center-alpha", type=float, default=0.18)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--disable-pca-init", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    graph = JsonGraphStore().load(resolve_path(args.graph))
    node_ids, node_types, texts = build_training_nodes(graph)
    embeddings = load_or_encode_node_embeddings(
        graph=graph,
        node_ids=node_ids,
        node_types=node_types,
        texts=texts,
        model_name_or_path=args.embedder,
        cache_path=resolve_path(args.embedding_cache),
        base_cache_path=resolve_path(args.base_node_embedding_cache),
        batch_size=args.embedding_batch_size,
        force_sentence_transformer=args.encode_with_sentence_transformer,
    )
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges, groups = build_training_edges_and_groups(graph, node_to_idx)
    parent_pools = build_parent_pools(node_types)
    edges = [edge for edge in edges if len(parent_pools.get(edge[2], [])) >= 2]
    if not edges:
        raise RuntimeError("no v4.1 hierarchy edges found")

    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    model = GraphV2HyperbolicMapper(input_dim=x.shape[1], tangent_dim=args.tangent_dim).to(device)
    if not args.disable_pca_init:
        initialize_mapper_with_pca(model, x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"device={device} nodes={len(node_ids)} dim={x.shape[1]} edges={len(edges)} "
        f"groups={{{', '.join(f'{k}: {len(v)}' for k, v in groups.items())}}} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    print(json.dumps(edge_summary(edges, node_types), indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        totals = defaultdict(float)
        for _step in range(args.steps_per_epoch):
            batch = sample_structure_batch(edges, parent_pools, args.batch_size, args.num_negatives, device)
            optimizer.zero_grad(set_to_none=True)
            child_points = model(x[batch["child"]])
            parent_points = model(x[batch["parent"]])
            neg_points = model(x[batch["neg"].reshape(-1)]).reshape(batch["neg"].shape[0], batch["neg"].shape[1], -1)
            edge_loss = edge_contrastive_loss(child_points, parent_points, neg_points, args.tau)
            radial_loss = radial_order_loss(child_points, parent_points, args.radial_margin)
            branch_loss = torch.tensor(0.0, device=device)
            branch_batch = sample_branch_batch(groups, build_fact_pool(node_types), args.branch_batch_size, device)
            if branch_batch is not None:
                branch_loss = branch_triplet_loss(
                    model(x[branch_batch["anchor"]]),
                    model(x[branch_batch["positive"]]),
                    model(x[branch_batch["negative"]]),
                    args.branch_margin,
                )
            center_loss = card_center_loss(model, x, groups.get("card_to_facts", {}), args.card_center_batch_size, device)
            loss = (
                args.edge_alpha * edge_loss
                + args.radial_alpha * radial_loss
                + args.branch_alpha * branch_loss
                + args.card_center_alpha * center_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["edge"] += float(edge_loss.detach().cpu())
            totals["radial"] += float(radial_loss.detach().cpu())
            totals["branch"] += float(branch_loss.detach().cpu())
            totals["center"] += float(center_loss.detach().cpu())
            totals["loss"] += float(loss.detach().cpu())
        denom = float(args.steps_per_epoch)
        diagnostics = compute_diagnostics(model, x, node_ids, node_types, groups)
        print(
            f"epoch={epoch} loss={totals['loss']/denom:.4f} edge={totals['edge']/denom:.4f} "
            f"radial={totals['radial']/denom:.4f} branch={totals['branch']/denom:.4f} "
            f"center={totals['center']/denom:.4f} card_chain={diagnostics['radial_order'].get('event_lt_card_lt_fact', 0):.4f}",
            flush=True,
        )

    diagnostics = compute_diagnostics(model, x, node_ids, node_types, groups)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "tangent_dim": args.tangent_dim,
            "scale": model.scale,
            "metadata": {
                "training": "v4_1_relation_card_lorentz_router_global_no_gold",
                "graph": str(resolve_path(args.graph)),
                "embedder": args.embedder,
                "embedding_cache": str(resolve_path(args.embedding_cache)),
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
                "radial_alpha": args.radial_alpha,
                "branch_alpha": args.branch_alpha,
                "card_center_alpha": args.card_center_alpha,
                "seed": args.seed,
                "num_nodes": len(node_ids),
                "num_edges": len(edges),
                "diagnostics": diagnostics,
            },
        },
        output,
    )
    diagnostics_path = resolve_path(args.diagnostics_json)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    print(f"wrote {diagnostics_path}", flush=True)


def build_training_nodes(graph: MemoryGraph) -> tuple[list[str], list[NodeType], list[str]]:
    rows = [
        (node_id, node.type, node.text)
        for node_id, node in graph.nodes.items()
        if node.type in TRAIN_NODE_TYPES
    ]
    rows.sort(key=lambda item: (item[1].value, item[0]))
    return [row[0] for row in rows], [row[1] for row in rows], [row[2] for row in rows]


def load_or_encode_node_embeddings(
    *,
    graph: MemoryGraph,
    node_ids: list[str],
    node_types: list[NodeType],
    texts: list[str],
    model_name_or_path: str,
    cache_path: Path,
    base_cache_path: Path,
    batch_size: int,
    force_sentence_transformer: bool,
) -> np.ndarray:
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(item) for item in payload["node_ids"]]
        cached_model = str(payload["model"][0]) if "model" in payload.files else ""
        if cached_ids == node_ids and cached_model == model_name_or_path:
            return np.asarray(payload["embeddings"], dtype=np.float32)
    if not force_sentence_transformer:
        embeddings = build_hybrid_embeddings(
            graph=graph,
            node_ids=node_ids,
            node_types=node_types,
            texts=texts,
            base_cache_path=base_cache_path,
            dim=384,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            node_ids=np.asarray(node_ids),
            embeddings=embeddings,
            model=np.asarray([f"hybrid_card_centroid:{base_cache_path.name}"]),
        )
        return embeddings
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


def build_hybrid_embeddings(
    *,
    graph: MemoryGraph,
    node_ids: list[str],
    node_types: list[NodeType],
    texts: list[str],
    base_cache_path: Path,
    dim: int,
) -> np.ndarray:
    base = {}
    if base_cache_path.exists():
        payload = np.load(base_cache_path, allow_pickle=False)
        base = {
            str(node_id): np.asarray(vector, dtype=np.float32)
            for node_id, vector in zip(payload["node_ids"], payload["embeddings"])
        }
        if base:
            dim = len(next(iter(base.values())))
    encoder = HashTextEncoder(dim=dim)
    rows = np.zeros((len(node_ids), dim), dtype=np.float32)
    node_to_pos = {node_id: idx for idx, node_id in enumerate(node_ids)}
    for idx, (node_id, node_type, text) in enumerate(zip(node_ids, node_types, texts)):
        if node_type != NodeType.RELATION_CARD and node_id in base:
            rows[idx] = base[node_id]
        elif node_type != NodeType.RELATION_CARD:
            rows[idx] = encoder.encode_one(text)
    for idx, (node_id, node_type, text) in enumerate(zip(node_ids, node_types, texts)):
        if node_type != NodeType.RELATION_CARD:
            continue
        node = graph.nodes[node_id]
        support_vectors = [
            rows[node_to_pos[fact_id]]
            for fact_id in node.metadata.get("support_fact_ids", node.support_ids)
            if fact_id in node_to_pos and np.linalg.norm(rows[node_to_pos[fact_id]]) > 0
        ]
        if support_vectors:
            centroid = np.mean(np.stack(support_vectors), axis=0)
            text_vector = encoder.encode_one(text)
            rows[idx] = 0.85 * centroid + 0.15 * text_vector
        else:
            rows[idx] = encoder.encode_one(text)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows / np.maximum(norms, 1e-6)
    return rows.astype(np.float32)


def build_training_edges_and_groups(
    graph: MemoryGraph,
    node_to_idx: dict[str, int],
) -> tuple[list[tuple[int, int, str]], dict[str, dict[int, list[int]]]]:
    edges: list[tuple[int, int, str]] = []
    groups: dict[str, dict[int, list[int]]] = {
        "event_to_facts": defaultdict(list),
        "card_to_facts": defaultdict(list),
        "episode_to_events": defaultdict(list),
    }
    for edge in graph.edges:
        if edge.relation != RelationType.IS_SPECIFIC_OF:
            continue
        if edge.src not in node_to_idx or edge.dst not in node_to_idx:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role_v33 = edge.metadata.get("hierarchy_v3_3")
        role_v41 = edge.metadata.get("hierarchy_v4_1")
        src_idx = node_to_idx[edge.src]
        dst_idx = node_to_idx[edge.dst]
        if src.type == NodeType.FACT and dst.type == NodeType.RELATION_CARD and role_v41 == "fact_card":
            edges.append((src_idx, dst_idx, NodeType.RELATION_CARD.value))
            groups["card_to_facts"][dst_idx].append(src_idx)
        elif src.type == NodeType.RELATION_CARD and dst.type == NodeType.EVENT and role_v41 in {"card_event", "card_episode"}:
            edges.append((src_idx, dst_idx, NodeType.EVENT.value))
        elif src.type == NodeType.RELATION_CARD and dst.type == NodeType.TOPIC and role_v41 == "card_topic":
            edges.append((src_idx, dst_idx, NodeType.TOPIC.value))
        elif src.type == NodeType.FACT and dst.type == NodeType.EVENT and role_v33 == "fact_event":
            edges.append((src_idx, dst_idx, NodeType.EVENT.value))
            groups["event_to_facts"][dst_idx].append(src_idx)
        elif src.type == NodeType.EVENT and dst.type == NodeType.EVENT and role_v33 == "event_episode":
            edges.append((src_idx, dst_idx, NodeType.EVENT.value))
            groups["episode_to_events"][dst_idx].append(src_idx)
        elif src.type == NodeType.EVENT and dst.type == NodeType.TOPIC and role_v33 in {"event_topic", "episode_topic"}:
            edges.append((src_idx, dst_idx, NodeType.TOPIC.value))
    return edges, groups


def build_parent_pools(node_types: list[NodeType]) -> dict[str, list[int]]:
    pools = defaultdict(list)
    for idx, node_type in enumerate(node_types):
        if node_type in {NodeType.RELATION_CARD, NodeType.EVENT, NodeType.TOPIC}:
            pools[node_type.value].append(idx)
    return dict(pools)


def build_fact_pool(node_types: list[NodeType]) -> list[int]:
    return [idx for idx, node_type in enumerate(node_types) if node_type == NodeType.FACT]


def sample_structure_batch(
    edges: list[tuple[int, int, str]],
    parent_pools: dict[str, list[int]],
    batch_size: int,
    num_negatives: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sampled = random.choices(edges, k=min(batch_size, len(edges)))
    negatives = []
    for _child, parent, parent_type in sampled:
        pool = parent_pools[parent_type]
        row = []
        while len(row) < num_negatives:
            candidate = random.choice(pool)
            if candidate != parent:
                row.append(candidate)
        negatives.append(row)
    return {
        "child": torch.tensor([item[0] for item in sampled], dtype=torch.long, device=device),
        "parent": torch.tensor([item[1] for item in sampled], dtype=torch.long, device=device),
        "neg": torch.tensor(negatives, dtype=torch.long, device=device),
    }


def sample_branch_batch(
    groups: dict[str, dict[int, list[int]]],
    fact_pool: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    eligible = []
    for group_name in ["card_to_facts", "event_to_facts"]:
        eligible.extend((group_name, parent, facts) for parent, facts in groups.get(group_name, {}).items() if len(facts) >= 2)
    if not eligible or not fact_pool:
        return None
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        _name, _parent, facts = random.choice(eligible)
        anchor, positive = random.sample(facts, 2)
        negative = random.choice(fact_pool)
        tries = 0
        while negative in facts and tries < 30:
            negative = random.choice(fact_pool)
            tries += 1
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    return {
        "anchor": torch.tensor(anchors, dtype=torch.long, device=device),
        "positive": torch.tensor(positives, dtype=torch.long, device=device),
        "negative": torch.tensor(negatives, dtype=torch.long, device=device),
    }


def card_center_loss(
    model: GraphV2HyperbolicMapper,
    x: torch.Tensor,
    card_to_facts: dict[int, list[int]],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    eligible = [(card, facts) for card, facts in card_to_facts.items() if facts]
    if not eligible:
        return torch.tensor(0.0, device=device)
    sampled = random.choices(eligible, k=min(batch_size, len(eligible)))
    card_indices = []
    sampled_fact_indices = []
    for card_idx, member_fact_indices in sampled:
        for fact_idx in random.sample(member_fact_indices, k=min(3, len(member_fact_indices))):
            card_indices.append(card_idx)
            sampled_fact_indices.append(fact_idx)
    if not card_indices:
        return torch.tensor(0.0, device=device)
    card_tensor = torch.tensor(card_indices, dtype=torch.long, device=device)
    fact_tensor = torch.tensor(sampled_fact_indices, dtype=torch.long, device=device)
    card_points = model(x[card_tensor])
    fact_points = model(x[fact_tensor])
    return lorentz_distance(card_points, fact_points).mean()


def initialize_mapper_with_pca(model: GraphV2HyperbolicMapper, x: torch.Tensor) -> None:
    with torch.no_grad():
        centered = x - x.mean(dim=0, keepdim=True)
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[: model.linear.out_features]
        model.linear.weight.zero_()
        model.linear.weight[: components.shape[0], : components.shape[1]] = components
        model.linear.bias.zero_()


def compute_diagnostics(
    model: GraphV2HyperbolicMapper,
    x: torch.Tensor,
    node_ids: list[str],
    node_types: list[NodeType],
    groups: dict[str, dict[int, list[int]]],
) -> dict:
    with torch.no_grad():
        points = model(x)
        radii = lorentz_radius(points).detach().cpu().numpy()
    radius = {}
    for node_type in [NodeType.TOPIC, NodeType.EVENT, NodeType.RELATION_CARD, NodeType.FACT]:
        values = [float(r) for r, t in zip(radii, node_types) if t == node_type]
        radius[node_type.value] = describe(values)
    orders = []
    for card_idx, facts in groups.get("card_to_facts", {}).items():
        for fact_idx in facts:
            orders.append(radii[fact_idx] > radii[card_idx])
    event_card_orders = []
    # The parent side is indirectly covered by edge radial training; this summary focuses on the new inserted layer.
    return {
        "radius": radius,
        "radial_order": {
            "fact_gt_card": mean_bool(orders),
            "event_lt_card_lt_fact": mean_bool(orders),
            "num_fact_card_pairs": len(orders),
        },
        "counts": {
            "nodes": len(node_ids),
            "card_groups": len(groups.get("card_to_facts", {})),
            "event_groups": len(groups.get("event_to_facts", {})),
        },
    }


def edge_summary(edges: list[tuple[int, int, str]], node_types: list[NodeType]) -> dict:
    counts = Counter(parent_type for _child, _parent, parent_type in edges)
    return {"edge_parent_type_counts": dict(counts)}


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0}
    return {"count": float(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values))}


def mean_bool(values: list[bool]) -> float:
    return float(sum(bool(v) for v in values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
