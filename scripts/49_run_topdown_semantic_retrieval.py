from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.retrieval_metrics import (
    evaluate_item,
    evidence_ids_for_node,
    normalize_evidence_id,
    summarize,
)
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import EvidencePath, NodeType
from hytopomem.retrieval.cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankCandidate,
    reranked_paths,
)
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    TopDownSemanticConfig,
    TopDownSemanticRetriever,
    default_embedder,
    merge_route_metadata,
)


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_hierarchy_v2.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--bottom-up-paths", default=None)
    parser.add_argument("--mode", choices=["event", "topic", "both"], default="both")
    parser.add_argument("--hierarchy-version", choices=["v2", "v3", "v3_3"], default="v2")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v2_minilm_event_topic.npz")
    parser.add_argument("--event-topk", type=int, default=20)
    parser.add_argument("--topic-topk", type=int, default=3)
    parser.add_argument("--events-per-topic", type=int, default=3)
    parser.add_argument("--facts-per-event", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--ce-model", default=DEFAULT_LOCAL_CE)
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--skip-ce", action="store_true")
    parser.add_argument("--final-topk", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-prefix", default="outputs/topdown/eu_minilm_both")
    args = parser.parse_args()

    config = load_config(args.config)
    graph_path = resolve_path(args.graph)
    graph = JsonGraphStore().load(graph_path)
    question_path = resolve_path(args.questions or config["data"]["processed_path"])
    questions = flatten_questions(read_json(question_path), args.limit)
    bottom_items = load_item_map(args.bottom_up_paths)

    encoder = SentenceTransformerEncoder(
        args.embedder,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    retriever = TopDownSemanticRetriever(
        graph,
        encoder=encoder,
        config=TopDownSemanticConfig(
            event_topk=args.event_topk,
            topic_topk=args.topic_topk,
            events_per_topic=args.events_per_topic,
            facts_per_event=args.facts_per_event,
            max_candidates=args.max_candidates,
            mode=args.mode,
            restrict_conversation=not args.global_search,
            hierarchy_version=args.hierarchy_version,
        ),
        cache_path=resolve_path(args.embedding_cache),
    )
    query_vectors = encoder.encode([item["question"] for item in questions])
    topdown_groups: dict[str, list[RerankCandidate]] = {}
    union_groups: dict[str, list[RerankCandidate]] = {}
    topdown_pools = {}
    union_pools = {}
    for index, (qa, query_vector) in enumerate(zip(questions, query_vectors), start=1):
        topdown_candidates = retriever.candidates_from_vector(qa["question_id"], query_vector)
        topdown_groups[qa["question_id"]] = topdown_candidates
        topdown_pools[qa["question_id"]] = candidate_ids(topdown_candidates)
        if bottom_items:
            bottom_candidates = candidates_from_paths(graph, bottom_items.get(qa["question_id"], {}).get("paths", []))
            union_candidates = merge_candidates(bottom_candidates, topdown_candidates)
            union_groups[qa["question_id"]] = union_candidates
            union_pools[qa["question_id"]] = candidate_ids(union_candidates)
        if index % 500 == 0 or index == len(questions):
            print(f"generated candidates {index}/{len(questions)} questions", flush=True)

    score_map = {}
    if not args.skip_ce:
        reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
        score_map = score_candidate_groups(
            reranker,
            {item["question_id"]: item["question"] for item in questions},
            topdown_groups,
            union_groups,
        )

    topdown_outputs = []
    union_outputs = []
    for index, qa in enumerate(questions, start=1):
        topdown_paths = rank_candidates(
            qa["question_id"],
            qa["question"],
            topdown_groups.get(qa["question_id"], []),
            score_map=score_map,
            top_k=args.final_topk,
            retriever_name=f"eu_topdown_{args.mode}",
        )
        topdown_outputs.append(output_item(qa, topdown_paths, f"eu_topdown_{args.mode}", args))

        if bottom_items:
            union_paths = rank_candidates(
                qa["question_id"],
                qa["question"],
                union_groups.get(qa["question_id"], []),
                score_map=score_map,
                top_k=args.final_topk,
                retriever_name=f"bottom_union_eu_topdown_{args.mode}",
            )
            union_outputs.append(output_item(qa, union_paths, f"bottom_union_eu_topdown_{args.mode}", args))
        if index % 500 == 0 or index == len(questions):
            print(f"ranked {index}/{len(questions)} questions", flush=True)

    prefix = resolve_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    topdown_paths_path = Path(f"{prefix}_paths.json")
    topdown_eval_path = Path(f"{prefix}_eval.json")
    write_json(topdown_outputs, topdown_paths_path)
    topdown_eval = evaluation_payload(
        graph,
        topdown_outputs,
        topdown_pools,
        args.final_topk,
        method=f"eu_topdown_{args.mode}",
    )
    write_json(topdown_eval, topdown_eval_path)
    print(f"topdown summary={topdown_eval['summary']}")
    print(f"topdown candidate_pool={topdown_eval['candidate_pool']}")

    if bottom_items:
        union_paths_path = Path(f"{prefix}_union_paths.json")
        union_eval_path = Path(f"{prefix}_union_eval.json")
        diagnostics_path = Path(f"{prefix}_paired_diagnostics.json")
        write_json(union_outputs, union_paths_path)
        union_eval = evaluation_payload(
            graph,
            union_outputs,
            union_pools,
            args.final_topk,
            method=f"bottom_union_eu_topdown_{args.mode}",
        )
        write_json(union_eval, union_eval_path)
        bottom_rows = [bottom_items[item["question_id"]] for item in questions if item["question_id"] in bottom_items]
        diagnostics = paired_diagnostics(
            graph,
            bottom_rows,
            topdown_outputs,
            union_outputs,
            topdown_pools,
            args.final_topk,
        )
        write_json(diagnostics, diagnostics_path)
        print(f"union summary={union_eval['summary']}")
        print(f"union candidate_pool={union_eval['candidate_pool']}")
        print(f"paired={diagnostics['summary']}")
        print(f"wrote {union_paths_path}, {union_eval_path}, {diagnostics_path}")
    print(f"wrote {topdown_paths_path}, {topdown_eval_path}")


def rank_candidates(
    question_id: str,
    question: str,
    candidates: list[RerankCandidate],
    *,
    score_map: dict[tuple[str, str], float],
    top_k: int,
    retriever_name: str,
) -> list[EvidencePath]:
    if not score_map:
        ranked = [(candidate, candidate.base_score) for candidate in candidates[:top_k]]
    else:
        ranked = sorted(
            (
                (candidate, score_map.get((question_id, candidate.node.node_id), candidate.base_score))
                for candidate in candidates
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    return reranked_paths(question_id, ranked, retriever_name=retriever_name)


def score_candidate_groups(
    reranker: CrossEncoderReranker,
    questions: dict[str, str],
    *groups: dict[str, list[RerankCandidate]],
) -> dict[tuple[str, str], float]:
    pairs = []
    keys = []
    seen = set()
    for group in groups:
        for question_id, candidates in group.items():
            question = questions.get(question_id, "")
            for candidate in candidates:
                key = (question_id, candidate.node.node_id)
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                pairs.append((question, candidate.node.text))
    print(f"cross-encoder scoring pairs={len(pairs)} batch_size={reranker.batch_size}", flush=True)
    if not pairs:
        return {}
    scores = reranker.model.predict(pairs, batch_size=reranker.batch_size, show_progress_bar=True)
    return {key: float(score) for key, score in zip(keys, scores)}


def candidates_from_paths(graph, paths: list[dict]) -> list[RerankCandidate]:
    candidates = []
    seen = set()
    for rank, path in enumerate(paths, start=1):
        node_id = evidence_node_id(graph, path)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        node = graph.nodes[node_id]
        metadata = dict(path.get("metadata", {}))
        previous_source = str(metadata.get("candidate_source", "bottom_up"))
        metadata["candidate_source"] = f"bottom_up:{previous_source}"
        metadata["route_source"] = "bottom_up"
        metadata["bottom_up_rank"] = str(rank)
        candidates.append(
            RerankCandidate(
                node=node,
                base_score=float(path.get("score", 0.0)),
                path_node_ids=list(path.get("node_ids", [])) or [node_id],
                path_edge_ids=list(path.get("edge_ids", [])),
                metadata=metadata,
            )
        )
    return candidates


def merge_candidates(
    bottom_candidates: list[RerankCandidate],
    topdown_candidates: list[RerankCandidate],
) -> list[RerankCandidate]:
    merged: dict[str, RerankCandidate] = {}
    for candidate in [*bottom_candidates, *topdown_candidates]:
        node_id = candidate.node.node_id
        previous = merged.get(node_id)
        if previous is None:
            merged[node_id] = candidate
            continue
        sources = set(str(previous.metadata.get("route_source", "")).split("+"))
        sources.update(str(candidate.metadata.get("route_source", "")).split("+"))
        previous.metadata["route_source"] = "+".join(sorted(item for item in sources if item))
        previous.metadata["candidate_source"] = previous.metadata["route_source"]
        previous.metadata["from_both"] = "True"
        previous.metadata = merge_route_metadata(previous.metadata, candidate.metadata or {})
        previous.metadata["from_both"] = "True"
        if candidate.base_score > previous.base_score:
            previous.base_score = candidate.base_score
            previous.path_node_ids = candidate.path_node_ids
            previous.path_edge_ids = candidate.path_edge_ids
    return list(merged.values())


def evaluation_payload(graph, items: list[dict], pools: dict[str, list[str]], k: int, *, method: str) -> dict:
    results = [evaluate_item(graph, item, k) for item in items]
    return {
        "summary": summarize(results),
        "candidate_pool": candidate_pool_summary(graph, items, pools),
        "per_question": [result.__dict__ for result in results],
        "metadata": {"method": method, "final_topk": k},
    }


def candidate_pool_summary(graph, items: list[dict], pools: dict[str, list[str]]) -> dict:
    rows = []
    for item in items:
        gold = {normalize_evidence_id(value) for value in item.get("gold_evidence", [])}
        node_ids = pools.get(item["question_id"], [])
        matched_gold = set()
        positive_candidates = 0
        for node_id in node_ids:
            evidence = evidence_ids_for_node(graph, node_id)
            overlap = gold & evidence
            if overlap:
                positive_candidates += 1
                matched_gold.update(overlap)
        rows.append(
            {
                "num_candidates": len(node_ids),
                "oracle_hit": bool(matched_gold),
                "oracle_recall": len(matched_gold) / len(gold) if gold else 0.0,
                "oracle_full_cover": bool(gold) and gold.issubset(matched_gold),
                "gold_density": positive_candidates / len(node_ids) if node_ids else 0.0,
            }
        )
    return {
        "num_questions": len(rows),
        "oracle_hit": mean(float(row["oracle_hit"]) for row in rows) if rows else 0.0,
        "oracle_recall": mean(row["oracle_recall"] for row in rows) if rows else 0.0,
        "oracle_full_cover": mean(float(row["oracle_full_cover"]) for row in rows) if rows else 0.0,
        "avg_candidates": mean(row["num_candidates"] for row in rows) if rows else 0.0,
        "gold_density": mean(row["gold_density"] for row in rows) if rows else 0.0,
    }


def paired_diagnostics(
    graph,
    bottom_items: list[dict],
    topdown_items: list[dict],
    union_items: list[dict],
    topdown_pools: dict[str, list[str]],
    k: int,
) -> dict:
    bottom = {item["question_id"]: evaluate_item(graph, item, k) for item in bottom_items}
    topdown = {item["question_id"]: evaluate_item(graph, item, k) for item in topdown_items}
    union = {item["question_id"]: evaluate_item(graph, item, k) for item in union_items}
    question_ids = sorted(set(bottom) & set(topdown) & set(union))
    buckets = {"both_hit": [], "bottom_only": [], "topdown_only": [], "both_miss": []}
    oracle_topdown_only = []
    for qid in question_ids:
        bottom_hit = bottom[qid].hit
        topdown_hit = topdown[qid].hit
        if bottom_hit and topdown_hit:
            buckets["both_hit"].append(qid)
        elif bottom_hit:
            buckets["bottom_only"].append(qid)
        elif topdown_hit:
            buckets["topdown_only"].append(qid)
        else:
            buckets["both_miss"].append(qid)
        if not bottom_hit and pool_hits_gold(graph, topdown_pools.get(qid, []), topdown[qid].gold_evidence_ids):
            oracle_topdown_only.append(qid)
    return {
        "summary": {
            "num_questions": len(question_ids),
            **{name: len(values) for name, values in buckets.items()},
            "topdown_oracle_recovers_bottom_miss": len(oracle_topdown_only),
            "bottom_hit": mean(float(bottom[qid].hit) for qid in question_ids) if question_ids else 0.0,
            "topdown_hit": mean(float(topdown[qid].hit) for qid in question_ids) if question_ids else 0.0,
            "union_hit": mean(float(union[qid].hit) for qid in question_ids) if question_ids else 0.0,
        },
        "question_ids": {**buckets, "topdown_oracle_recovers_bottom_miss": oracle_topdown_only},
    }


def pool_hits_gold(graph, node_ids: list[str], gold_ids: list[str]) -> bool:
    gold = {normalize_evidence_id(value) for value in gold_ids}
    return any(gold & evidence_ids_for_node(graph, node_id) for node_id in node_ids)


def output_item(qa: dict, paths: list[EvidencePath], method: str, args) -> dict:
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "category": qa.get("category"),
        "gold_evidence": qa.get("evidence", []),
        "paths": [path.model_dump(mode="json") for path in paths],
        "metadata": {
            "method": method,
            "mode": args.mode,
            "hierarchy_version": args.hierarchy_version,
            "embedder": args.embedder,
            "event_topk": args.event_topk,
            "topic_topk": args.topic_topk,
            "events_per_topic": args.events_per_topic,
            "facts_per_event": args.facts_per_event,
            "max_candidates": args.max_candidates,
            "restrict_conversation": not args.global_search,
            "ce_model": None if args.skip_ce else args.ce_model,
            "final_topk": args.final_topk,
        },
    }


def candidate_ids(candidates: list[RerankCandidate]) -> list[str]:
    return [candidate.node.node_id for candidate in candidates]


def evidence_node_id(graph, path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id in graph.nodes:
        return metadata_id
    for node_id in reversed(path.get("node_ids", [])):
        node = graph.nodes.get(node_id)
        if node is not None and node.type in {NodeType.FACT, NodeType.RAW}:
            return node_id
    return ""


def load_item_map(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    rows = read_json(resolve_path(path))
    return {str(item["question_id"]): item for item in rows}


def flatten_questions(conversations: list[dict], limit: int) -> list[dict]:
    questions = []
    for conversation in conversations:
        for qa in conversation.get("qa", []):
            questions.append(qa)
            if limit and len(questions) >= limit:
                return questions
    return questions


if __name__ == "__main__":
    main()
