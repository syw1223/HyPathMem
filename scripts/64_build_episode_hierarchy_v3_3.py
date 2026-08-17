from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from statistics import mean, median

from common import resolve_path, write_json
from hytopomem.memory.episode_hierarchy_builder import EpisodeHierarchyBuilder, EpisodeHierarchyConfig
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType
from hytopomem.retrieval.topdown_semantic_retriever import SentenceTransformerEncoder, default_embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_2_gpt4o_semantic.json")
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_3_episode.json")
    parser.add_argument("--graph-id", default="locomo_semantic_hierarchy_v3_3_episode")
    parser.add_argument("--diagnostics", default="outputs/eval/graph_v3_3_episode_structure_diagnostics.json")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--episode-similarity-threshold", type=float, default=0.50)
    parser.add_argument("--episode-max-events", type=int, default=6)
    parser.add_argument("--min-episode-events", type=int, default=2)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    encoder = SentenceTransformerEncoder(args.embedder, device=args.device, batch_size=args.embedding_batch_size)
    builder = EpisodeHierarchyBuilder(
        encoder,
        EpisodeHierarchyConfig(
            episode_similarity_threshold=args.episode_similarity_threshold,
            episode_max_events=args.episode_max_events,
            min_episode_events=args.min_episode_events,
        ),
    )
    output = builder.build(graph, graph_id=args.graph_id)
    output_path = resolve_path(args.output)
    JsonGraphStore().save(output, output_path)
    diagnostics = graph_v3_3_diagnostics(output)
    diagnostics_path = resolve_path(args.diagnostics)
    write_json(diagnostics, diagnostics_path)
    print(f"wrote {output_path}")
    print(f"wrote {diagnostics_path}")
    print(f"hierarchy_v3_3={output.metadata.get('hierarchy_v3_3', {})}")
    print(f"structure={diagnostics['structure']}")


def graph_v3_3_diagnostics(graph: MemoryGraph) -> dict:
    episode_to_events: dict[str, list[str]] = defaultdict(list)
    topic_to_episodes: dict[str, list[str]] = defaultdict(list)
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        role = edge.metadata.get("hierarchy_v3_3")
        if role == "fact_event":
            event_to_facts[edge.dst].append(edge.src)
        elif role == "event_episode":
            episode_to_events[edge.dst].append(edge.src)
        elif role == "episode_topic":
            topic_to_episodes[edge.dst].append(edge.src)

    episode_sizes = [len(values) for values in episode_to_events.values()]
    topic_sizes = [len(values) for values in topic_to_episodes.values()]
    event_sizes = [len(values) for values in event_to_facts.values()]
    episode_coherence = [
        float(graph.nodes[episode_id].metadata.get("coherence", 0.0))
        for episode_id in episode_to_events
    ]
    node_counts = Counter(node.type.value for node in graph.nodes.values())
    episode_nodes = [
        node_id
        for node_id, node in graph.nodes.items()
        if node.metadata.get("hierarchy_v3_3") == "episode"
    ]
    return {
        "metadata": graph.metadata.get("hierarchy_v3_3", {}),
        "node_counts": dict(node_counts),
        "structure": {
            "events": len(event_sizes),
            "episodes": len(episode_sizes),
            "topics": len(topic_sizes),
            "episode_nodes_by_metadata": len(episode_nodes),
            "mean_facts_per_event": safe_mean(event_sizes),
            "mean_events_per_episode": safe_mean(episode_sizes),
            "median_events_per_episode": safe_median(episode_sizes),
            "singleton_episode_ratio": ratio_equal(episode_sizes, 1),
            "episode_size_p95": percentile(episode_sizes, 0.95),
            "episode_size_max": max(episode_sizes, default=0),
            "mean_episodes_per_topic": safe_mean(topic_sizes),
            "median_episodes_per_topic": safe_median(topic_sizes),
            "topic_episode_size_max": max(topic_sizes, default=0),
            "episode_coherence_mean": safe_mean(episode_coherence),
            "episode_coherence_median": safe_median(episode_coherence),
        },
    }


def safe_mean(values: list[float] | list[int]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: list[float] | list[int]) -> float:
    return float(median(values)) if values else 0.0


def ratio_equal(values: list[int], target: int) -> float:
    return sum(value == target for value in values) / len(values) if values else 0.0


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(quantile * (len(ordered) - 1))))
    return int(ordered[index])


if __name__ == "__main__":
    main()
