from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

from common import load_config, read_json, resolve_path, write_json
from hytopomem.eval.oracle_metrics import evaluate_candidate_pool, summarize_oracle
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.schema import MemoryGraph, NodeType, RelationType
from hytopomem.retrieval.bm25_retriever import BM25Retriever
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker, RerankCandidate, reranked_paths
from hytopomem.retrieval.topdown_semantic_retriever import merge_route_metadata


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


def default_model() -> str:
    return DEFAULT_LOCAL_CE if Path(DEFAULT_LOCAL_CE).exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_multiview_v3_4_ab.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--candidate-b", default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_B_true_bu_euhyp_td_hyp_paths.json")
    parser.add_argument("--candidate-d", default="outputs/dual_geometry/full_graph_v3_3_episode_true_union_D_true_bu_euhyp_td_euhyp_paths.json")
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--temporal-topn", type=int, default=100)
    parser.add_argument("--union-topn", type=int, default=150)
    parser.add_argument("--ce-model", default=default_model())
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-prefix", default="outputs/dual_geometry/full_graph_v3_4_ab_temporal_union")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = helpers.flatten_questions(
        read_json(resolve_path(args.questions or config["data"]["processed_path"])),
        args.limit,
    )
    temporal_index = TemporalIndex.from_graph(graph)
    bm25 = BM25Retriever(list(graph.iter_nodes(NodeType.FACT)))
    b_items = load_item_map(args.candidate_b)
    d_items = load_item_map(args.candidate_d)

    b_groups: dict[str, list[RerankCandidate]] = {}
    d_groups: dict[str, list[RerankCandidate]] = {}
    b_pools: dict[str, list[str]] = {}
    d_pools: dict[str, list[str]] = {}

    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        temporal_candidates = temporal_candidates_for_question(
            graph,
            bm25,
            temporal_index,
            qa["question"],
            seed_topn=args.seed_topn,
            temporal_topn=args.temporal_topn,
        )
        b_base = candidates_from_paths(graph, b_items.get(qid, {}).get("paths", []), route="B")
        d_base = candidates_from_paths(graph, d_items.get(qid, {}).get("paths", []), route="D")
        b_union = merge_candidates(b_base, temporal_candidates)
        d_union = merge_candidates(d_base, temporal_candidates)
        b_groups[qid] = b_union
        d_groups[qid] = d_union
        b_pools[qid] = [candidate.node.node_id for candidate in b_union]
        d_pools[qid] = [candidate.node.node_id for candidate in d_union]
        if index % 250 == 0 or index == len(questions):
            print(f"built temporal union candidate groups {index}/{len(questions)}", flush=True)

    reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
    score_map = score_candidate_groups(
        reranker,
        {item["question_id"]: item["question"] for item in questions},
        b_groups,
        d_groups,
    )

    outputs = {
        "B_bu_euhyp_td_hyp_temporal_union150": [],
        "D_bu_euhyp_td_euhyp_temporal_union150": [],
    }
    pools = {key: {} for key in outputs}
    for qa in questions:
        qid = qa["question_id"]
        b_paths = rank_candidates(
            qid,
            b_groups.get(qid, []),
            score_map=score_map,
            top_k=args.union_topn,
            retriever_name="v3_4_B_temporal_union_ce",
        )
        d_paths = rank_candidates(
            qid,
            d_groups.get(qid, []),
            score_map=score_map,
            top_k=args.union_topn,
            retriever_name="v3_4_D_temporal_union_ce",
        )
        outputs["B_bu_euhyp_td_hyp_temporal_union150"].append(output_item(qa, b_paths, args, "B_bu_euhyp_td_hyp_temporal_union150"))
        outputs["D_bu_euhyp_td_euhyp_temporal_union150"].append(output_item(qa, d_paths, args, "D_bu_euhyp_td_euhyp_temporal_union150"))
        pools["B_bu_euhyp_td_hyp_temporal_union150"][qid] = [evidence_node_id(path.model_dump(mode="json")) for path in b_paths]
        pools["D_bu_euhyp_td_euhyp_temporal_union150"][qid] = [evidence_node_id(path.model_dump(mode="json")) for path in d_paths]

    prefix = resolve_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for key, items in outputs.items():
        path = Path(f"{prefix}_{key}_paths.json")
        eval_path = Path(f"{prefix}_{key}_eval.json")
        write_json(items, path)
        payload = evaluation_payload(graph, items, pools[key], args.union_topn, method=key)
        write_json(payload, eval_path)
        print(f"{key} summary={payload['summary']}")
        print(f"{key} candidate_pool={payload['candidate_pool']}")
        print(f"wrote {path}")


class TemporalIndex:
    def __init__(self) -> None:
        self.fact_event: dict[str, str] = {}
        self.event_facts: dict[str, list[str]] = defaultdict(list)
        self.event_sessions: dict[str, list[str]] = defaultdict(list)
        self.session_events: dict[str, list[str]] = defaultdict(list)

    @classmethod
    def from_graph(cls, graph: MemoryGraph) -> "TemporalIndex":
        index = cls()
        for edge in graph.edges:
            if edge.relation != RelationType.IS_SPECIFIC_OF:
                continue
            src = graph.nodes.get(edge.src)
            dst = graph.nodes.get(edge.dst)
            if src is None or dst is None:
                continue
            if src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                if edge.metadata.get("hierarchy_v3") != "lexical_alias_event":
                    index.fact_event[edge.src] = edge.dst
                    index.event_facts[edge.dst].append(edge.src)
            elif edge.metadata.get("hierarchy_v3_4_temporal") == "event_session":
                index.event_sessions[edge.src].append(edge.dst)
                index.session_events[edge.dst].append(edge.src)
        return index


def temporal_candidates_for_question(
    graph: MemoryGraph,
    bm25: BM25Retriever,
    index: TemporalIndex,
    question: str,
    *,
    seed_topn: int,
    temporal_topn: int,
) -> list[RerankCandidate]:
    seed_hits = bm25.search(question, top_k=max(seed_topn, 50))
    rows: list[RerankCandidate] = []
    seen: set[str] = set()
    temporal_rank = 0
    for seed_rank, (seed, seed_score) in enumerate(seed_hits[:seed_topn], start=1):
        seed_event_id = index.fact_event.get(seed.node_id)
        if not seed_event_id:
            continue
        for session_id in index.event_sessions.get(seed_event_id, []):
            for event_id in index.session_events.get(session_id, []):
                for fact_id in index.event_facts.get(event_id, []):
                    if fact_id in seen or fact_id not in graph.nodes:
                        continue
                    seen.add(fact_id)
                    temporal_rank += 1
                    fact = graph.nodes[fact_id]
                    metadata = {
                        "candidate_source": "temporal_session",
                        "route_source": "temporal_session",
                        "temporal_rank": str(temporal_rank),
                        "temporal_seed_rank": str(seed_rank),
                        "temporal_seed_node_id": seed.node_id,
                        "temporal_seed_event_id": seed_event_id,
                        "temporal_session_id": session_id,
                        "event_node_id": event_id,
                        "seed_node_id": seed.node_id,
                        "is_seed": str(fact_id == seed.node_id),
                        "hop": "2",
                    }
                    rows.append(
                        RerankCandidate(
                            node=fact,
                            base_score=float(seed_score) / max(float(seed_rank), 1.0),
                            path_node_ids=[seed.node_id, seed_event_id, session_id, event_id, fact_id],
                            path_edge_ids=[],
                            metadata=metadata,
                        )
                    )
                    if len(rows) >= temporal_topn:
                        return rows
    return rows


def candidates_from_paths(graph: MemoryGraph, paths: list[dict], *, route: str) -> list[RerankCandidate]:
    candidates = []
    seen = set()
    for rank, path in enumerate(paths, start=1):
        node_id = evidence_node_id(path)
        if not node_id or node_id in seen or node_id not in graph.nodes:
            continue
        seen.add(node_id)
        metadata = dict(path.get("metadata", {}))
        metadata[f"{route.lower()}_route_rank"] = str(rank)
        candidates.append(
            RerankCandidate(
                node=graph.nodes[node_id],
                base_score=float(path.get("scores", {}).get("base", path.get("score", 0.0))),
                path_node_ids=list(path.get("node_ids", [])) or [node_id],
                path_edge_ids=list(path.get("edge_ids", [])),
                metadata=metadata,
            )
        )
    return candidates


def merge_candidates(*groups: list[RerankCandidate]) -> list[RerankCandidate]:
    merged: dict[str, RerankCandidate] = {}
    for group in groups:
        for candidate in group:
            node_id = candidate.node.node_id
            previous = merged.get(node_id)
            if previous is None:
                merged[node_id] = copy_candidate(candidate)
            else:
                previous.metadata = merge_route_metadata(previous.metadata or {}, candidate.metadata or {})
                previous.base_score = max(previous.base_score, candidate.base_score)
                if len(candidate.path_node_ids or []) > len(previous.path_node_ids or []):
                    previous.path_node_ids = candidate.path_node_ids
                    previous.path_edge_ids = candidate.path_edge_ids
    return list(merged.values())


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


def rank_candidates(
    question_id: str,
    candidates: list[RerankCandidate],
    *,
    score_map: dict[tuple[str, str], float],
    top_k: int,
    retriever_name: str,
):
    ranked = sorted(
        ((candidate, score_map.get((question_id, candidate.node.node_id), candidate.base_score)) for candidate in candidates),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return reranked_paths(question_id, ranked, retriever_name=retriever_name)


def output_item(qa: dict, paths: list, args, method: str) -> dict:
    return {
        "question_id": qa["question_id"],
        "question": qa["question"],
        "answer": qa.get("answer"),
        "category": qa.get("category"),
        "gold_evidence": qa.get("evidence", []),
        "paths": [path.model_dump(mode="json") for path in paths],
        "metadata": {
            "method": method,
            "graph": str(resolve_path(args.graph)),
            "candidate_b": str(resolve_path(args.candidate_b)),
            "candidate_d": str(resolve_path(args.candidate_d)),
            "seed_topn": args.seed_topn,
            "temporal_topn": args.temporal_topn,
            "union_topn": args.union_topn,
            "ce_model": args.ce_model,
        },
    }


def evaluation_payload(graph: MemoryGraph, items: list[dict], pools: dict[str, list[str]], k: int, *, method: str) -> dict:
    results = [
        evaluate_candidate_pool(
            graph,
            question_id=item["question_id"],
            gold_evidence=item.get("gold_evidence", []),
            candidate_node_ids=pools[item["question_id"]][:k],
        )
        for item in items
    ]
    return {
        "summary": summarize_oracle(results),
        "candidate_pool": summarize_oracle(results),
        "per_question": [result.__dict__ for result in results],
        "metadata": {"method": method, "k": k},
    }


def evidence_node_id(path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id:
        return metadata_id
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def copy_candidate(candidate: RerankCandidate) -> RerankCandidate:
    return RerankCandidate(
        node=candidate.node,
        base_score=candidate.base_score,
        path_node_ids=list(candidate.path_node_ids or []),
        path_edge_ids=list(candidate.path_edge_ids or []),
        metadata=dict(candidate.metadata or {}),
    )


def load_item_map(path: str) -> dict[str, dict]:
    return {str(item["question_id"]): item for item in read_json(resolve_path(path))}


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
