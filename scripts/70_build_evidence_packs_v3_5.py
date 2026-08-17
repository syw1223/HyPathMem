from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_path, write_json
from hytopomem.memory.evidence_pack_builder import EvidencePackBuilder, EvidencePackConfig
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.models.text_encoder import HashTextEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_5_entity.json")
    parser.add_argument("--output", default="outputs/graphs/locomo_graph_multiview_v3_5_packs.json")
    parser.add_argument("--diagnostics-json", default="outputs/eval/graph_v3_5_evidence_pack_diagnostics.json")
    parser.add_argument("--diagnostics-md", default="outputs/eval/graph_v3_5_evidence_pack_diagnostics.md")
    parser.add_argument("--graph-id", default="locomo_multiview_v3_5_packs")
    parser.add_argument("--min-bridge-facts", type=int, default=2)
    parser.add_argument("--max-pack-facts", type=int, default=24)
    parser.add_argument("--encoder-dim", type=int, default=256)
    args = parser.parse_args()

    graph = JsonGraphStore().load(resolve_path(args.graph))
    builder = EvidencePackBuilder(
        encoder=HashTextEncoder(dim=args.encoder_dim),
        config=EvidencePackConfig(
            min_bridge_facts=args.min_bridge_facts,
            max_pack_facts=args.max_pack_facts,
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
        ("Pack specs", stats.get("pack_specs", 0)),
        ("Pack nodes added", stats.get("pack_nodes_added", 0)),
        ("Pack member edges added", stats.get("pack_member_edges_added", 0)),
        ("Pack count", d.get("pack_count", 0)),
        ("Mean pack size", f"{d.get('mean_pack_size', 0.0):.4f}"),
        ("Median pack size", f"{d.get('median_pack_size', 0.0):.4f}"),
        ("Pack size p90", f"{d.get('pack_size_p90', 0.0):.4f}"),
        ("Max pack size", d.get("max_pack_size", 0)),
        ("Pack coherence mean", f"{d.get('pack_coherence_mean', 0.0):.4f}"),
        ("Pack coherence median", f"{d.get('pack_coherence_median', 0.0):.4f}"),
        ("Pack coherence min", f"{d.get('pack_coherence_min', 0.0):.4f}"),
        ("Pack coherence max", f"{d.get('pack_coherence_max', 0.0):.4f}"),
        ("Membership weight mean", f"{d.get('membership_weight_mean', 0.0):.4f}"),
        ("Membership weight median", f"{d.get('membership_weight_median', 0.0):.4f}"),
        ("Membership weight min", f"{d.get('membership_weight_min', 0.0):.4f}"),
        ("Membership weight max", f"{d.get('membership_weight_max', 0.0):.4f}"),
    ]
    lines = [
        "# Graph V3.5-B Evidence Pack Diagnostics",
        "",
        "## Config",
        "",
        f"- Source graph: `{args.graph}`",
        f"- Output graph: `{output_path}`",
        f"- Diagnostics JSON: `{diagnostics_path}`",
        f"- min_bridge_facts: `{args.min_bridge_facts}`",
        f"- max_pack_facts: `{args.max_pack_facts}`",
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
            "## Pack Type Counts",
            "",
            "| Pack Type | Count | Mean Size | Max Size | Mean Coherence |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for pack_type, item in d.get("pack_type_size_summary", {}).items():
        lines.append(
            f"| {pack_type} | {item.get('count', 0)} | "
            f"{item.get('mean_size', 0.0):.4f} | {item.get('max_size', 0)} | "
            f"{item.get('mean_coherence', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Pack Sizes",
            "",
            ", ".join(str(item) for item in d.get("largest_pack_sizes", [])) or "N/A",
            "",
            "## Notes",
            "",
            "- Pack membership uses `MEMBER_OF`, not `IS_SPECIFIC_OF`, so pack edges are not part of the hyperbolic partial-order backbone.",
            "- BridgePack requires at least two shared facts between an EntityState and an Episode.",
            "- RAW ids are stored in pack metadata for provenance, but RAW nodes are not added as pack member edges to avoid very large membership cliques.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
