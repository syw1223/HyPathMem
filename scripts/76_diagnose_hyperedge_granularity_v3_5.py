from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import evidence_ids_for_node, normalize_evidence_id
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, Node, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever


@dataclass(frozen=True)
class VirtualPack:
    node_id: str
    pack_type: str
    text: str
    fact_ids: tuple[str, ...]
    raw_ids: tuple[str, ...]

    def to_node(self) -> Node:
        return Node(
            node_id=self.node_id,
            type=NodeType.EVIDENCE_PACK,
            text=self.text,
            source="virtual_hyperedge_diagnostic",
            support_ids=list(self.fact_ids),
            metadata={
                "pack_type": self.pack_type,
                "fact_ids": list(self.fact_ids),
                "raw_ids": list(self.raw_ids),
                "num_facts": len(self.fact_ids),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_5_packs.json")
    parser.add_argument(
        "--base-candidates",
        default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_notemporal_top150_D_true_bu_euhyp_td_euhyp_paths.json",
    )
    parser.add_argument("--questions", default=None)
    parser.add_argument("--candidate-topn", type=int, default=150)
    parser.add_argument("--pack-topk", type=int, default=20)
    parser.add_argument("--oracle-pick", default="1,2,3")
    parser.add_argument("--max-change-packs-per-conv", type=int, default=600)
    parser.add_argument("--output-json", default="outputs/eval/graph_v3_5_hyperedge_granularity_diagnosis.json")
    parser.add_argument("--output-md", default="outputs/eval/graph_v3_5_hyperedge_granularity_diagnosis.md")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = helpers.flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), 0)
    base_items = {str(item["question_id"]): item for item in read_json(resolve_path(args.base_candidates))}
    oracle_pick_values = [int(value) for value in args.oracle_pick.split(",") if value.strip()]
    pack_sets = build_pack_sets(graph, args.max_change_packs_per_conv)

    payload = {
        "graph": str(resolve_path(args.graph)),
        "base_candidates": str(resolve_path(args.base_candidates)),
        "candidate_topn": args.candidate_topn,
        "pack_topk": args.pack_topk,
        "oracle_pick_values": oracle_pick_values,
        "pack_sets": {},
    }
    for pack_name, packs in pack_sets.items():
        print(f"diagnosing {pack_name}: packs={len(packs)}", flush=True)
        payload["pack_sets"][pack_name] = diagnose_pack_set(
            graph,
            questions,
            base_items,
            packs,
            args.candidate_topn,
            args.pack_topk,
            oracle_pick_values,
        )

    out_json = resolve_path(args.output_json)
    write_json(payload, out_json)
    out_md = resolve_path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({name: item["summary"] for name, item in payload["pack_sets"].items()}, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


def build_pack_sets(graph: MemoryGraph, max_change_packs_per_conv: int) -> dict[str, list[VirtualPack]]:
    existing = existing_pack_sets(graph)
    event_fact = event_fact_packs(graph)
    change = change_relation_packs(graph, max_change_packs_per_conv)
    output = {
        "episode": existing["episode"],
        "episode_small_le5": [pack for pack in existing["episode"] if len(pack.fact_ids) <= 5],
        "event_fact": event_fact,
        "entity_state": existing["entity_state"],
        "entity_state_small_le5": [pack for pack in existing["entity_state"] if len(pack.fact_ids) <= 5],
        "bridge": existing["bridge_entity_episode"],
        "bridge_small_le5": [pack for pack in existing["bridge_entity_episode"] if len(pack.fact_ids) <= 5],
        "change_relation": change,
    }
    return {name: packs for name, packs in output.items() if packs}


def existing_pack_sets(graph: MemoryGraph) -> dict[str, list[VirtualPack]]:
    output: dict[str, list[VirtualPack]] = defaultdict(list)
    for node in graph.iter_nodes(NodeType.EVIDENCE_PACK):
        pack_type = str(node.metadata.get("pack_type", "unknown"))
        fact_ids = tuple(str(item) for item in node.metadata.get("fact_ids", []))
        raw_ids = tuple(str(item) for item in node.metadata.get("raw_ids", []))
        if not fact_ids:
            continue
        output[pack_type].append(
            VirtualPack(
                node_id=node.node_id,
                pack_type=pack_type,
                text=pack_text(node, graph, fact_ids),
                fact_ids=fact_ids,
                raw_ids=raw_ids,
            )
        )
    return output


def event_fact_packs(graph: MemoryGraph) -> list[VirtualPack]:
    event_to_facts: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None:
            continue
        role = edge.metadata.get("hierarchy_v3_3") or edge.metadata.get("hierarchy_v3")
        if src.type == NodeType.FACT and dst.type == NodeType.EVENT and role == "fact_event":
            event_to_facts[dst.node_id].append(src.node_id)
    packs = []
    for event_id, fact_ids in sorted(event_to_facts.items()):
        event = graph.nodes.get(event_id)
        if event is None or not fact_ids:
            continue
        raw_ids = raw_ids_for_facts(graph, fact_ids)
        packs.append(
            VirtualPack(
                node_id=f"{event_id}:virtual_pack:event_fact",
                pack_type="event_fact",
                text=pack_text(event, graph, fact_ids),
                fact_ids=tuple(sorted(set(fact_ids))),
                raw_ids=tuple(raw_ids),
            )
        )
    return packs


def change_relation_packs(graph: MemoryGraph, max_per_conv: int) -> list[VirtualPack]:
    rows_by_conv: dict[str, list[tuple[float, str, str, RelationType]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.relation not in {RelationType.UPDATES, RelationType.CONFLICTS_WITH}:
            continue
        src = graph.nodes.get(edge.src)
        dst = graph.nodes.get(edge.dst)
        if src is None or dst is None or src.type != NodeType.FACT or dst.type != NodeType.FACT:
            continue
        conv_id = conversation_id(edge.src)
        if conv_id != conversation_id(edge.dst):
            continue
        rows_by_conv[conv_id].append((float(edge.confidence), edge.src, edge.dst, edge.relation))
    packs = []
    seen = set()
    for conv_id, rows in sorted(rows_by_conv.items()):
        rows.sort(key=lambda item: item[0], reverse=True)
        for rank, (confidence, src, dst, relation) in enumerate(rows[:max_per_conv], start=1):
            key = tuple(sorted([src, dst]))
            if key in seen:
                continue
            seen.add(key)
            fact_ids = tuple(key)
            raw_ids = tuple(raw_ids_for_facts(graph, fact_ids))
            text = " ".join(
                [
                    f"ChangeRelation {relation.value} confidence {confidence:.3f}.",
                    graph.nodes[src].text,
                    graph.nodes[dst].text,
                ]
            )
            packs.append(
                VirtualPack(
                    node_id=f"{conv_id}:virtual_pack:change:{rank:04d}",
                    pack_type="change_relation",
                    text=text,
                    fact_ids=fact_ids,
                    raw_ids=raw_ids,
                )
            )
    return packs


def diagnose_pack_set(
    graph: MemoryGraph,
    questions: list[dict],
    base_items: dict[str, dict],
    packs: list[VirtualPack],
    candidate_topn: int,
    pack_topk: int,
    oracle_pick_values: list[int],
) -> dict:
    nodes = [pack.to_node() for pack in packs]
    retriever = BM25Retriever(nodes)
    pack_by_id = {pack.node_id: pack for pack in packs}
    rows = []
    base_miss_rows = []
    oracle_rows = {m: [] for m in oracle_pick_values}
    size_rows = defaultdict(list)
    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        gold = gold_set(qa)
        hits = [
            (pack_by_id[node.node_id], score)
            for node, score in retriever.search(qa["question"], top_k=max(pack_topk * 10, pack_topk))
            if conversation_id(node.node_id) == conversation_id(qid)
        ][:pack_topk]
        row = pack_eval(hits, gold)
        rows.append(row)
        for pack, _score in hits:
            size_rows[size_bucket(len(pack.fact_ids))].append(pack_single_purity(pack, gold))
        base_ids = fact_ids_from_paths(base_items.get(qid, {}).get("paths", [])[:candidate_topn])
        base_gold = predicted_gold_for_facts(graph, base_ids)
        missing_gold = gold - base_gold
        if missing_gold:
            base_miss_rows.append(pack_eval(hits, missing_gold))
        for m in oracle_pick_values:
            oracle_rows[m].append(oracle_pick_eval(graph, hits, gold, m))
        if index % 500 == 0:
            print(f"  {index}/{len(questions)}", flush=True)
    return {
        "pack_count": len(packs),
        "pack_size": pack_size_summary(packs),
        "summary": summarize_pack_rows(rows),
        "base_miss_subset": summarize_pack_rows(base_miss_rows),
        "base_miss_questions": len(base_miss_rows),
        "oracle_pick": {str(m): summarize_pack_rows(oracle_rows[m]) for m in oracle_pick_values},
        "size_bucket_purity": {
            bucket: {
                "num_selected_packs": len(values),
                "mean_purity": mean(values) if values else 0.0,
            }
            for bucket, values in sorted(size_rows.items())
        },
    }


def pack_eval(hits: list[tuple[VirtualPack, float]], gold: set[str]) -> dict:
    predicted = set()
    best_purity = 0.0
    best_cover = 0.0
    selected_sizes = []
    for pack, _score in hits:
        pack_gold = set(normalize_raw_id(raw_id) for raw_id in pack.raw_ids)
        matched = pack_gold & gold
        predicted.update(pack_gold)
        selected_sizes.append(len(pack.fact_ids))
        best_cover = max(best_cover, len(matched) / len(gold) if gold else 0.0)
        best_purity = max(best_purity, len(matched) / len(pack_gold) if pack_gold else 0.0)
    matched_all = predicted & gold
    return {
        "hit": bool(matched_all),
        "recall": len(matched_all) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "best_pack_cover": best_cover,
        "best_pack_purity": best_purity,
        "avg_selected_pack_size": mean(selected_sizes) if selected_sizes else 0.0,
    }


def oracle_pick_eval(graph: MemoryGraph, hits: list[tuple[VirtualPack, float]], gold: set[str], m: int) -> dict:
    predicted = set()
    selected_sizes = []
    for pack, _score in hits:
        fact_rows = []
        for fact_id in pack.fact_ids:
            fact_gold = evidence_ids_for_node(graph, fact_id)
            overlap = fact_gold & gold
            fact_rows.append((len(overlap), fact_id, fact_gold))
        fact_rows.sort(key=lambda item: (-item[0], item[1]))
        for _overlap, _fact_id, fact_gold in fact_rows[:m]:
            predicted.update(fact_gold)
        selected_sizes.append(min(m, len(fact_rows)))
    matched = predicted & gold
    return {
        "hit": bool(matched),
        "recall": len(matched) / len(gold) if gold else 0.0,
        "full_cover": bool(gold) and gold.issubset(predicted),
        "best_pack_cover": 0.0,
        "best_pack_purity": 0.0,
        "avg_selected_pack_size": mean(selected_sizes) if selected_sizes else 0.0,
    }


def pack_single_purity(pack: VirtualPack, gold: set[str]) -> float:
    pack_gold = set(normalize_raw_id(raw_id) for raw_id in pack.raw_ids)
    return len(pack_gold & gold) / len(pack_gold) if pack_gold else 0.0


def summarize_pack_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "hit": 0.0,
            "recall": 0.0,
            "full_cover": 0.0,
            "best_pack_cover": 0.0,
            "best_pack_purity": 0.0,
            "avg_selected_pack_size": 0.0,
        }
    return {
        "num_questions": len(rows),
        "hit": mean(float(row["hit"]) for row in rows),
        "recall": mean(float(row["recall"]) for row in rows),
        "full_cover": mean(float(row["full_cover"]) for row in rows),
        "best_pack_cover": mean(float(row["best_pack_cover"]) for row in rows),
        "best_pack_purity": mean(float(row["best_pack_purity"]) for row in rows),
        "avg_selected_pack_size": mean(float(row["avg_selected_pack_size"]) for row in rows),
    }


def pack_size_summary(packs: list[VirtualPack]) -> dict:
    sizes = [len(pack.fact_ids) for pack in packs]
    return {
        "mean": mean(sizes) if sizes else 0.0,
        "median": median(sizes) if sizes else 0.0,
        "max": max(sizes, default=0),
        "le5_ratio": sum(float(size <= 5) for size in sizes) / len(sizes) if sizes else 0.0,
    }


def pack_text(node: Node, graph: MemoryGraph, fact_ids: tuple[str, ...] | list[str]) -> str:
    fact_text = " ".join(graph.nodes[fact_id].text for fact_id in list(fact_ids)[:8] if fact_id in graph.nodes)
    keywords = []
    for key in ["entity", "aspect_keywords", "episode_keywords", "keywords", "entities"]:
        value = node.metadata.get(key)
        if isinstance(value, list):
            keywords.extend(str(item) for item in value)
        elif value:
            keywords.append(str(value))
    return " ".join([node.text, " ".join(keywords), fact_text]).strip()


def raw_ids_for_facts(graph: MemoryGraph, fact_ids: tuple[str, ...] | list[str]) -> list[str]:
    raw_ids = []
    seen = set()
    for fact_id in fact_ids:
        node = graph.nodes.get(str(fact_id))
        if node is None:
            continue
        for raw_id in list(node.metadata.get("support_raw_ids") or node.support_ids):
            normalized = normalize_raw_id(raw_id)
            if normalized in seen:
                continue
            seen.add(normalized)
            raw_ids.append(normalized)
    return raw_ids


def predicted_gold_for_facts(graph: MemoryGraph, fact_ids: list[str]) -> set[str]:
    predicted = set()
    for fact_id in fact_ids:
        predicted.update(evidence_ids_for_node(graph, fact_id))
    return predicted


def fact_ids_from_paths(paths: list[dict]) -> list[str]:
    output = []
    seen = set()
    for path in paths:
        fact_id = evidence_node_id(path)
        if fact_id and fact_id not in seen:
            seen.add(fact_id)
            output.append(fact_id)
    return output


def evidence_node_id(path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id:
        return metadata_id
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def gold_set(item: dict) -> set[str]:
    return {normalize_evidence_id(eid) for eid in item.get("evidence", item.get("gold_evidence", []))}


def normalize_raw_id(value: object) -> str:
    text = str(value)
    if ":raw:" in text:
        text = text.rsplit(":raw:", 1)[-1]
    return normalize_evidence_id(text)


def conversation_id(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def size_bucket(size: int) -> str:
    if size <= 2:
        return "01_le2"
    if size <= 5:
        return "02_3_5"
    if size <= 10:
        return "03_6_10"
    if size <= 20:
        return "04_11_20"
    return "05_gt20"


def render_markdown(payload: dict) -> str:
    lines = [
        "# Graph V3.5 Hyperedge Granularity Diagnosis",
        "",
        f"- Graph: `{payload['graph']}`",
        f"- Base candidates: `{payload['base_candidates']}`",
        f"- Candidate topN: `{payload['candidate_topn']}`",
        f"- Pack topK: `{payload['pack_topk']}`",
        "",
        "## Pack@K Summary",
        "",
        "| Pack Set | Count | Mean Size | Pack Hit | Pack Recall | Pack Full | Best Purity | Base-Miss Recall | Oracle Pick@1 Recall | Oracle Pick@3 Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload["pack_sets"].items():
        summary = item["summary"]
        base_miss = item["base_miss_subset"]
        pick1 = item["oracle_pick"].get("1", {})
        pick3 = item["oracle_pick"].get("3", {})
        lines.append(
            f"| {name} | {item['pack_count']} | {item['pack_size']['mean']:.2f} | "
            f"{summary['hit']:.4f} | {summary['recall']:.4f} | {summary['full_cover']:.4f} | "
            f"{summary['best_pack_purity']:.4f} | {base_miss['recall']:.4f} | "
            f"{pick1.get('recall', 0.0):.4f} | {pick3.get('recall', 0.0):.4f} |"
        )
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(
        [
            "- `event_fact` tests whether smaller event-level hyperedges are cleaner than EpisodePack.",
            "- `change_relation` uses existing `UPDATES` and `CONFLICTS_WITH` fact pairs as tentative n-ary/change units.",
            "- `base_miss_subset` only evaluates questions where the base top150 does not fully cover gold evidence.",
            "- `oracle_pick@m` estimates whether the pack is good but representative fact selection is weak.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_topdown_helpers():
    path = Path(__file__).with_name("49_run_topdown_semantic_retrieval.py")
    spec = importlib.util.spec_from_file_location("topdown_semantic_retrieval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
