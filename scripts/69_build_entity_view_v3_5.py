from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_path, write_json
from hytopomem.memory.entity_view_builder import EntityViewBuilder, EntityViewConfig
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.models.text_encoder import HashTextEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_4_ab.json")
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_multiview_v3_5_entity.json")
    parser.add_argument("--diagnostics-json", default="outputs/eval/graph_v3_5_entity_view_diagnostics.json")
    parser.add_argument("--diagnostics-md", default="outputs/eval/graph_v3_5_entity_view_diagnostics.md")
    parser.add_argument("--graph-id", default="locomo_multiview_v3_5_entity")
    parser.add_argument("--min-state-facts", type=int, default=2)
    parser.add_argument("--max-state-facts", type=int, default=20)
    parser.add_argument("--cluster-threshold", type=float, default=0.50)
    parser.add_argument("--encoder-dim", type=int, default=256)
    args = parser.parse_args()

    graph_path = resolve_path(args.graph)
    graph = JsonGraphStore().load(graph_path)
    builder = EntityViewBuilder(
        encoder=HashTextEncoder(dim=args.encoder_dim),
        config=EntityViewConfig(
            min_state_facts=args.min_state_facts,
            max_state_facts=args.max_state_facts,
            cluster_threshold=args.cluster_threshold,
        ),
    )
    output, stats = builder.build(graph, graph_id=args.graph_id)

    output_path = resolve_path(args.output)
    JsonGraphStore().save(output, output_path)
    diagnostics_path = resolve_path(args.diagnostics_json)
    write_json(stats, diagnostics_path)
    md_path = resolve_path(args.diagnostics_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(args, output_path, diagnostics_path, stats), encoding="utf-8")

    print(f"wrote graph {output_path}")
    print(f"wrote diagnostics {diagnostics_path}")
    print(f"wrote markdown {md_path}")
    print(json.dumps(stats["diagnostics"], indent=2, ensure_ascii=False))


def render_markdown(args, output_path: Path, diagnostics_path: Path, stats: dict) -> str:
    d = stats.get("diagnostics", {})
    rows = [
        ("Input fact-entity units", stats.get("input_fact_entity_units", 0)),
        ("Entity nodes added", stats.get("entity_nodes_added", 0)),
        ("EntityState nodes added", stats.get("entity_state_nodes_added", 0)),
        ("Fact -> EntityState edges", stats.get("fact_state_edges_added", 0)),
        ("EntityState -> Entity edges", stats.get("state_entity_edges_added", 0)),
        ("Direct Fact -> Entity edges", stats.get("direct_fact_entity_edges_added", 0)),
        ("Hub entity split ratio", f"{stats.get('hub_entity_split_ratio', 0.0):.4f}"),
        ("Fact with entity path ratio", f"{d.get('fact_with_entity_path_ratio', 0.0):.4f}"),
        ("Mean facts / EntityState", f"{d.get('mean_facts_per_entity_state', 0.0):.4f}"),
        ("Median facts / EntityState", f"{d.get('median_facts_per_entity_state', 0.0):.4f}"),
        ("Max facts / EntityState", d.get("max_facts_per_entity_state", 0)),
        ("Singleton state ratio", f"{d.get('singleton_state_ratio', 0.0):.4f}"),
        ("EntityState coherence mean", f"{d.get('entity_state_coherence_mean', 0.0):.4f}"),
        ("EntityState coherence median", f"{d.get('entity_state_coherence_median', 0.0):.4f}"),
        ("High-frequency entity count", d.get("high_freq_entity_count", 0)),
        ("Split high-frequency entity count", d.get("split_high_freq_entity_count", 0)),
    ]
    lines = [
        "# Graph V3.5-A Entity View Diagnostics",
        "",
        "## Config",
        "",
        f"- Source graph: `{args.graph}`",
        f"- Output graph: `{output_path}`",
        f"- Diagnostics JSON: `{diagnostics_path}`",
        f"- min_state_facts: `{args.min_state_facts}`",
        f"- max_state_facts: `{args.max_state_facts}`",
        f"- cluster_threshold: `{args.cluster_threshold}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.extend(
        [
            "",
            "## Largest Entities",
            "",
            "| Entity | Facts | Node |",
            "| --- | ---: | --- |",
        ]
    )
    for item in d.get("largest_entity_fact_counts", []):
        lines.append(f"| {item.get('entity')} | {item.get('fact_count')} | `{item.get('node_id')}` |")
    lines.extend(
        [
            "",
            "## Largest EntityState Sizes",
            "",
            ", ".join(str(item) for item in d.get("largest_entity_state_sizes", [])) or "N/A",
            "",
            "## Interpretation Checklist",
            "",
            "- `fact_with_entity_path_ratio` should be high enough to make Entity View useful.",
            "- `max_facts_per_entity_state` should stay bounded by `max_state_facts` to avoid hub states.",
            "- `hub_entity_split_ratio` should be close to 1.0; otherwise high-frequency entities are not split.",
            "- `singleton_state_ratio` should be 0 because singleton entity facts are kept as direct Fact -> Entity edges.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
