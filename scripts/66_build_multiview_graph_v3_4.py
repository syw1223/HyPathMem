from __future__ import annotations

import argparse
from pathlib import Path

from common import resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.multiview_graph_builder import MultiViewGraphBuilder, MultiViewGraphConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_3_episode.json")
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_multiview_v3_4_ab.json")
    parser.add_argument("--diagnostics", default="outputs/eval/graph_v3_4_ab_multiview_diagnostics.json")
    parser.add_argument("--graph-id", default="locomo_multiview_v3_4_ab")
    parser.add_argument("--no-raw-provenance", action="store_true")
    parser.add_argument("--no-temporal-view", action="store_true")
    args = parser.parse_args()

    graph_path = resolve_path(args.graph)
    output_path = resolve_path(args.output)
    diagnostics_path = resolve_path(args.diagnostics)

    graph = JsonGraphStore().load(graph_path)
    builder = MultiViewGraphBuilder(
        MultiViewGraphConfig(
            add_raw_provenance=not args.no_raw_provenance,
            add_temporal_view=not args.no_temporal_view,
        )
    )
    output, stats = builder.build(graph, graph_id=args.graph_id)
    stats["input_graph"] = str(graph_path)
    stats["output_graph"] = str(output_path)

    JsonGraphStore().save(output, output_path)
    write_json(stats, diagnostics_path)
    write_markdown(stats, diagnostics_path.with_suffix(".md"))

    print(f"wrote graph {output_path}")
    print(f"wrote diagnostics {diagnostics_path}")
    print(stats)


def write_markdown(stats: dict, path: Path) -> None:
    temporal = stats.get("temporal_view", {})
    raw = stats.get("raw_provenance", {})
    diagnostics = stats.get("diagnostics", {})
    lines = [
        "# Graph V3.4-A+B Multiview Diagnostics",
        "",
        f"Input graph: `{stats.get('input_graph')}`",
        f"Output graph: `{stats.get('output_graph')}`",
        "",
        "## Raw Provenance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "total_facts",
        "fixed_facts",
        "missing_raw_facts",
        "fact_with_raw_ratio",
        "added_support_edges",
        "unresolved_support_ids",
    ]:
        if key in raw:
            lines.append(f"| {key} | {format_value(raw[key])} |")
    lines.extend([
        "",
        "## Temporal View",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ])
    for key in [
        "semantic_events",
        "conversation_nodes",
        "session_nodes",
        "session_conversation_edges",
        "event_session_edges",
        "events_without_session",
        "event_with_session_ratio",
        "mean_sessions_per_event",
        "mean_events_per_session",
        "median_events_per_session",
        "max_events_per_session",
        "singleton_session_ratio",
    ]:
        if key in temporal:
            lines.append(f"| {key} | {format_value(temporal[key])} |")
    lines.extend([
        "",
        "## Node Counts",
        "",
        "| Type | Count |",
        "| --- | ---: |",
    ])
    for key, value in diagnostics.get("node_counts", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Temporal Edge Counts",
        "",
        "| Role | Count |",
        "| --- | ---: |",
    ])
    for key, value in diagnostics.get("temporal_edge_counts", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ])
    for key in ["fact_with_raw_ratio", "semantic_event_with_temporal_ratio"]:
        if key in diagnostics:
            lines.append(f"| {key} | {format_value(diagnostics[key])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
