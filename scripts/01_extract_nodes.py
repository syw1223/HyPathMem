from __future__ import annotations

import argparse

from common import load_config, read_json, resolve_path, write_json
from hytopomem.memory.node_extractor import RuleBasedNodeExtractor, nodes_from_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-observations", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["data"]["processed_path"])
    output_path = resolve_path(args.output or config["graph"]["nodes_path"])
    conversations = read_json(input_path)
    extractor = RuleBasedNodeExtractor()
    nodes = []
    for conversation in conversations:
        conversation_id = conversation["conversation_id"]
        extracted = extractor.extract(conversation_id, conversation.get("turns", []))
        nodes.extend(extracted.raw_nodes)
        nodes.extend(extracted.fact_nodes)
        nodes.extend(extracted.anchor_nodes)
        if args.use_observations:
            nodes.extend(
                nodes_from_observations(
                    conversation_id,
                    conversation.get("observation", {}),
                    conversation.get("evidence_lookup", {}),
                )
            )

    write_json([node.model_dump(mode="json") for node in nodes], output_path)
    print(f"wrote {len(nodes)} nodes to {output_path}")


if __name__ == "__main__":
    main()
