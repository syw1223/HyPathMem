from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from common import load_config, read_json, resolve_path, write_json
from hytopomem.memory.graph_store import JsonGraphStore
from hytopomem.memory.node_extractor import content_terms
from hytopomem.memory.schema import Edge, MemoryGraph, Node, NodeType, RelationType
from hytopomem.retrieval.cross_encoder_reranker import CrossEncoderReranker, RerankCandidate, reranked_paths
from hytopomem.retrieval.hyperbolic_topdown_retriever import (
    load_hyperbolic_router,
    lorentz_distance_numpy,
)
from hytopomem.retrieval.topdown_semantic_retriever import (
    SentenceTransformerEncoder,
    bucket_ids_by_conversation,
    conversation_id_from_question,
    default_embedder,
    merge_route_metadata,
)


DEFAULT_LOCAL_CE = "/home/sunyuwei/LightMem/LightMem/models/cross-encoder-ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class HypBottomConfig:
    seed_topn: int = 20
    preselect_topn: int = 100
    same_event_limit: int = 30
    topic_event_limit: int = 8
    topic_fact_limit_per_event: int = 12
    max_topic_degree: int = 160
    w_hyp: float = 1.0
    w_seed: float = 0.8
    w_event: float = 0.65
    w_topic: float = 0.35
    w_overlap: float = 0.45
    w_edge: float = 0.35
    w_degree: float = 0.25
    w_hop: float = 0.20
    restrict_conversation: bool = True
    hierarchy_version: str = "v3"


@dataclass
class HypBottomCandidate:
    node: Node
    score: float
    hyp_norm: float
    is_seed: bool
    source: str
    seed_node_id: str
    event_node_id: str
    topic_node_id: str
    hop: int
    edge_confidence: float
    event_degree: int
    topic_degree: int
    path_node_ids: list[str]
    seed_rank: int


class HyperbolicBottomUpRetriever:
    def __init__(
        self,
        graph: MemoryGraph,
        *,
        encoder: SentenceTransformerEncoder,
        checkpoint_path: Path,
        embedding_cache_path: Path,
        config: HypBottomConfig,
        device: str | None = None,
        batch_size: int = 1024,
    ):
        self.graph = graph
        self.encoder = encoder
        self.config = config
        self.device = torch.device(device if device and torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        checkpoint = load_hyperbolic_router(checkpoint_path, self.device)
        self.model = checkpoint.model
        self.model.eval()
        self.checkpoint_metadata = checkpoint.metadata
        self.hierarchy_key = f"hierarchy_{config.hierarchy_version}"
        self.facts = sorted(graph.iter_nodes(NodeType.FACT), key=lambda node: node.node_id)
        self.fact_ids = [node.node_id for node in self.facts]
        self.fact_index = {node_id: index for index, node_id in enumerate(self.fact_ids)}
        self.fact_to_event = self._fact_to_event()
        self.event_to_topic = self._event_to_topic()
        self.event_to_facts = self._event_to_facts()
        self.topic_to_events = self._topic_to_events()
        self.facts_by_conversation = bucket_ids_by_conversation(self.fact_ids)
        fact_embeddings = self._load_fact_embeddings(embedding_cache_path)
        self.fact_points = self._map_embeddings(fact_embeddings)
        self._outgoing: dict[str, list[Edge]] = {}
        for edge in graph.edges:
            self._outgoing.setdefault(edge.src, []).append(edge)

    def candidates(self, question_id: str, query: str) -> list[RerankCandidate]:
        query_vector = self.encoder.encode([query])[0]
        return self.candidates_from_vector(question_id, query, query_vector)

    def candidates_from_vector(self, question_id: str, query: str, query_vector: np.ndarray) -> list[RerankCandidate]:
        query_point = self._map_embeddings(np.asarray([query_vector], dtype=np.float32))[0]
        seed_hits = self._top_facts(question_id, query_point, self.config.seed_topn)
        hyp_scores = self._normalized_hyp_scores(question_id, query_point, max(self.config.preselect_topn * 4, 300))
        ranked = self._ranked_candidates_from_seed_hits(query, seed_hits, hyp_scores)
        return [self._to_rerank_candidate(candidate) for candidate in ranked]

    def _top_facts(self, question_id: str, query_point: np.ndarray, top_k: int) -> list[tuple[Node, float, int]]:
        fact_ids = self._candidate_fact_ids(question_id)
        scored = top_by_lorentz_fact_score(fact_ids, self.fact_index, self.fact_points, query_point, top_k)
        return [(self.graph.nodes[fact_id], score, rank) for rank, (fact_id, score) in enumerate(scored, start=1)]

    def _normalized_hyp_scores(self, question_id: str, query_point: np.ndarray, top_k: int) -> dict[str, float]:
        fact_ids = self._candidate_fact_ids(question_id)
        scored = top_by_lorentz_fact_score(fact_ids, self.fact_index, self.fact_points, query_point, top_k)
        if not scored:
            return {}
        values = [score for _fact_id, score in scored]
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return {fact_id: 1.0 for fact_id, _score in scored}
        return {fact_id: (score - lo) / (hi - lo) for fact_id, score in scored}

    def _ranked_candidates_from_seed_hits(
        self,
        query: str,
        seed_hits: list[tuple[Node, float, int]],
        hyp_scores: dict[str, float],
    ) -> list[HypBottomCandidate]:
        query_terms = set(content_terms(query))
        candidates: dict[str, HypBottomCandidate] = {}
        for seed, _seed_score, seed_rank in seed_hits:
            event_id = self.fact_to_event.get(seed.node_id, "")
            topic_id = self.event_to_topic.get(event_id, "") if event_id else ""
            self._upsert(
                candidates,
                self._make_candidate(
                    node=seed,
                    query_terms=query_terms,
                    hyp_norm=hyp_scores.get(seed.node_id, 0.0),
                    is_seed=True,
                    source="seed",
                    seed_node_id=seed.node_id,
                    event_node_id=event_id,
                    topic_node_id=topic_id,
                    hop=0,
                    edge_confidence=1.0,
                    path_node_ids=[seed.node_id],
                    seed_rank=seed_rank,
                ),
            )
            if event_id:
                for fact_id in self.event_to_facts.get(event_id, [])[: self.config.same_event_limit]:
                    if fact_id == seed.node_id:
                        continue
                    fact = self.graph.nodes.get(fact_id)
                    if fact is not None and fact.type == NodeType.FACT:
                        self._upsert(
                            candidates,
                            self._make_candidate(
                                node=fact,
                                query_terms=query_terms,
                                hyp_norm=hyp_scores.get(fact.node_id, 0.0),
                                is_seed=False,
                                source="same_event",
                                seed_node_id=seed.node_id,
                                event_node_id=event_id,
                                topic_node_id=topic_id,
                                hop=2,
                                edge_confidence=self._fact_event_confidence(fact.node_id, event_id),
                                path_node_ids=[seed.node_id, event_id, fact.node_id],
                                seed_rank=seed_rank,
                            ),
                        )
            if topic_id:
                topic_degree = self._topic_degree(topic_id)
                if topic_degree > self.config.max_topic_degree:
                    continue
                topic_events = self._rank_topic_events(topic_id, seed.node_id, query_terms)
                for event_id2 in topic_events[: self.config.topic_event_limit]:
                    fact_ids = self.event_to_facts.get(event_id2, [])[: self.config.topic_fact_limit_per_event]
                    for fact_id in fact_ids:
                        if fact_id == seed.node_id:
                            continue
                        fact = self.graph.nodes.get(fact_id)
                        if fact is None or fact.type != NodeType.FACT:
                            continue
                        source = "same_event" if event_id2 == event_id else "same_topic"
                        self._upsert(
                            candidates,
                            self._make_candidate(
                                node=fact,
                                query_terms=query_terms,
                                hyp_norm=hyp_scores.get(fact_id, 0.0),
                                is_seed=False,
                                source=source,
                                seed_node_id=seed.node_id,
                                event_node_id=event_id2,
                                topic_node_id=topic_id,
                                hop=2 if source == "same_event" else 4,
                                edge_confidence=self._fact_event_confidence(fact_id, event_id2),
                                path_node_ids=[seed.node_id, event_id, topic_id, event_id2, fact_id],
                                seed_rank=seed_rank,
                            ),
                        )
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[: self.config.preselect_topn]

    def _make_candidate(
        self,
        *,
        node: Node,
        query_terms: set[str],
        hyp_norm: float,
        is_seed: bool,
        source: str,
        seed_node_id: str,
        event_node_id: str,
        topic_node_id: str,
        hop: int,
        edge_confidence: float,
        path_node_ids: list[str],
        seed_rank: int,
    ) -> HypBottomCandidate:
        c = self.config
        event_degree = len(self.event_to_facts.get(event_node_id, [])) if event_node_id else 0
        topic_degree = self._topic_degree(topic_node_id) if topic_node_id else 0
        overlap = overlap_score(query_terms, node.text)
        source_score = c.w_event if source == "same_event" else c.w_topic if source == "same_topic" else 0.0
        score = (
            c.w_hyp * hyp_norm
            + c.w_seed * float(is_seed)
            + source_score
            + c.w_overlap * overlap
            + c.w_edge * edge_confidence
            - c.w_degree * math.log1p(max(event_degree, 1))
            - c.w_hop * hop
        )
        return HypBottomCandidate(
            node=node,
            score=score,
            hyp_norm=hyp_norm,
            is_seed=is_seed,
            source=source,
            seed_node_id=seed_node_id,
            event_node_id=event_node_id,
            topic_node_id=topic_node_id,
            hop=hop,
            edge_confidence=edge_confidence,
            event_degree=event_degree,
            topic_degree=topic_degree,
            path_node_ids=[node_id for node_id in path_node_ids if node_id],
            seed_rank=seed_rank,
        )

    def _to_rerank_candidate(self, candidate: HypBottomCandidate) -> RerankCandidate:
        metadata = {
            "candidate_source": candidate.source,
            "route_source": "bottom_up+hyp_bottom",
            "is_seed": str(candidate.is_seed),
            "seed_node_id": candidate.seed_node_id,
            "event_node_id": candidate.event_node_id,
            "topic_node_id": candidate.topic_node_id,
            "bm25_norm": "0.000000",
            "hyp_bottom_score": f"{candidate.hyp_norm:.6f}",
            "hyp_bottom_seed_rank": str(candidate.seed_rank),
            "bottom_up_rank": str(candidate.seed_rank),
            "hop": str(candidate.hop),
            "edge_confidence": f"{candidate.edge_confidence:.6f}",
            "event_degree": str(candidate.event_degree),
            "topic_degree": str(candidate.topic_degree),
            "hierarchy_v2_mode": "event_topic",
            "retriever": "hyperbolic_bottom_up",
        }
        return RerankCandidate(
            node=candidate.node,
            base_score=candidate.score,
            path_node_ids=candidate.path_node_ids,
            metadata=metadata,
        )

    def _candidate_fact_ids(self, question_id: str) -> list[str]:
        if not self.config.restrict_conversation:
            return self.fact_ids
        conv_id = conversation_id_from_question(question_id)
        return self.facts_by_conversation.get(conv_id, [])

    def _load_fact_embeddings(self, cache_path: Path) -> np.ndarray:
        payload = np.load(cache_path, allow_pickle=False)
        node_ids = [str(item) for item in payload["node_ids"]]
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        missing = [node_id for node_id in self.fact_ids if node_id not in by_id]
        if missing:
            raise ValueError(f"embedding cache missing {len(missing)} fact nodes; first={missing[:3]}")
        return embeddings[[by_id[node_id] for node_id in self.fact_ids]]

    def _map_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(embeddings), self.batch_size):
                batch = torch.tensor(embeddings[start : start + self.batch_size], dtype=torch.float32, device=self.device)
                outputs.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.empty((0, 0), dtype=np.float32)

    def _fact_to_event(self) -> dict[str, str]:
        mapping = {}
        for edge in self.graph.edges:
            if not self._is_hierarchy_edge(edge, "fact_event"):
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.FACT and dst.type == NodeType.EVENT:
                mapping[edge.src] = edge.dst
        return mapping

    def _event_to_topic(self) -> dict[str, str]:
        if self.config.hierarchy_version == "v3_3":
            event_to_episode = {}
            episode_to_topic = {}
            for edge in self.graph.edges:
                src = self.graph.nodes.get(edge.src)
                dst = self.graph.nodes.get(edge.dst)
                if src is None or dst is None:
                    continue
                if self._is_hierarchy_edge(edge, "event_episode") and src.type == NodeType.EVENT and dst.type == NodeType.EVENT:
                    event_to_episode[edge.src] = edge.dst
                elif self._is_hierarchy_edge(edge, "episode_topic") and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC:
                    episode_to_topic[edge.src] = edge.dst
            return {
                event_id: episode_to_topic[episode_id]
                for event_id, episode_id in event_to_episode.items()
                if episode_id in episode_to_topic
            }
        mapping = {}
        for edge in self.graph.edges:
            if not self._is_hierarchy_edge(edge, "event_topic"):
                continue
            src = self.graph.nodes.get(edge.src)
            dst = self.graph.nodes.get(edge.dst)
            if src is not None and dst is not None and src.type == NodeType.EVENT and dst.type == NodeType.TOPIC:
                mapping[edge.src] = edge.dst
        return mapping

    def _event_to_facts(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for fact_id, event_id in self.fact_to_event.items():
            mapping.setdefault(event_id, []).append(fact_id)
        return mapping

    def _topic_to_events(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for event_id, topic_id in self.event_to_topic.items():
            mapping.setdefault(topic_id, []).append(event_id)
        return mapping

    def _topic_degree(self, topic_id: str) -> int:
        return sum(len(self.event_to_facts.get(event_id, [])) for event_id in self.topic_to_events.get(topic_id, []))

    def _fact_event_confidence(self, fact_id: str, event_id: str) -> float:
        for edge in self._outgoing.get(fact_id, []):
            if edge.dst == event_id and self._is_hierarchy_edge(edge, "fact_event"):
                return edge.confidence
        return 0.0

    def _rank_topic_events(self, topic_id: str, seed_id: str, query_terms: set[str]) -> list[str]:
        seed_event = self.fact_to_event.get(seed_id, "")
        return sorted(
            self.topic_to_events.get(topic_id, []),
            key=lambda event_id: (
                event_id == seed_event,
                overlap_score(query_terms, self.graph.nodes[event_id].text) if event_id in self.graph.nodes else 0.0,
                len(self.event_to_facts.get(event_id, [])),
            ),
            reverse=True,
        )

    def _upsert(self, candidates: dict[str, HypBottomCandidate], candidate: HypBottomCandidate) -> None:
        existing = candidates.get(candidate.node.node_id)
        if existing is None or candidate.score > existing.score:
            candidates[candidate.node.node_id] = candidate

    def _is_hierarchy_edge(self, edge: Edge, role: str) -> bool:
        return edge.relation == RelationType.IS_SPECIFIC_OF and edge.metadata.get(self.hierarchy_key) == role


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--graph", default="outputs/graphs/locomo_graph_semantic_hierarchy_v3_2_gpt4o_semantic.json")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--checkpoint", default="outputs/models/graph_v2_lorentz_router/minilm_structure_router_v3_2_gpt4o_hardneg.pt")
    parser.add_argument("--embedder", default=default_embedder())
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-cache", default="outputs/embeddings/graph_v3_2_gpt4o_minilm_fact_event_topic.npz")
    parser.add_argument("--router-device", default="cuda")
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--hierarchy-version", choices=["v3", "v3_3"], default="v3")
    parser.add_argument("--td-hyp-paths", default="outputs/topdown/full_graph_v3_2_gpt4o_hyp_retrained_both_hardneg_ce_selector_top20base_paths.json")
    parser.add_argument("--seed-topn", type=int, default=20)
    parser.add_argument("--preselect-topn", type=int, default=100)
    parser.add_argument("--same-event-limit", type=int, default=30)
    parser.add_argument("--topic-event-limit", type=int, default=8)
    parser.add_argument("--topic-fact-limit-per-event", type=int, default=12)
    parser.add_argument("--candidate-topn", type=int, default=100)
    parser.add_argument("--ce-model", default=DEFAULT_LOCAL_CE)
    parser.add_argument("--ce-device", default=None)
    parser.add_argument("--ce-batch-size", type=int, default=128)
    parser.add_argument("--skip-ce", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-prefix", default="outputs/dual_geometry/full_graph_v3_2_pure_hyp")
    args = parser.parse_args()

    helpers = load_topdown_helpers()
    config = load_config(args.config)
    graph = JsonGraphStore().load(resolve_path(args.graph))
    questions = helpers.flatten_questions(read_json(resolve_path(args.questions or config["data"]["processed_path"])), args.limit)
    td_hyp_items = load_item_map(args.td_hyp_paths)

    encoder = SentenceTransformerEncoder(args.embedder, device=args.embedding_device, batch_size=args.embedding_batch_size)
    retriever = HyperbolicBottomUpRetriever(
        graph,
        encoder=encoder,
        checkpoint_path=resolve_path(args.checkpoint),
        embedding_cache_path=resolve_path(args.embedding_cache),
        config=HypBottomConfig(
            seed_topn=args.seed_topn,
            preselect_topn=args.preselect_topn,
            same_event_limit=args.same_event_limit,
            topic_event_limit=args.topic_event_limit,
            topic_fact_limit_per_event=args.topic_fact_limit_per_event,
            hierarchy_version=args.hierarchy_version,
        ),
        device=args.router_device,
        batch_size=args.router_batch_size,
    )
    query_vectors = encoder.encode([item["question"] for item in questions])

    bu_groups: dict[str, list[RerankCandidate]] = {}
    union_groups: dict[str, list[RerankCandidate]] = {}
    bu_pools: dict[str, list[str]] = {}
    union_pools: dict[str, list[str]] = {}
    for index, (qa, query_vector) in enumerate(zip(questions, query_vectors), start=1):
        qid = qa["question_id"]
        bu_candidates = retriever.candidates_from_vector(qid, qa["question"], query_vector)
        td_candidates = candidates_from_paths(graph, td_hyp_items.get(qid, {}).get("paths", []), route="hyp")
        union_candidates = merge_candidates(bu_candidates, td_candidates, args.candidate_topn)
        bu_groups[qid] = bu_candidates[: args.candidate_topn]
        union_groups[qid] = union_candidates
        bu_pools[qid] = [candidate.node.node_id for candidate in bu_groups[qid]]
        union_pools[qid] = [candidate.node.node_id for candidate in union_candidates]
        if index % 500 == 0 or index == len(questions):
            print(f"generated pure-hyp bottom-up {index}/{len(questions)}", flush=True)

    score_map: dict[tuple[str, str], float] = {}
    if not args.skip_ce:
        reranker = CrossEncoderReranker(args.ce_model, device=args.ce_device, batch_size=args.ce_batch_size)
        score_map = score_candidate_groups(reranker, {item["question_id"]: item["question"] for item in questions}, bu_groups, union_groups)

    bu_outputs = []
    union_outputs = []
    for index, qa in enumerate(questions, start=1):
        qid = qa["question_id"]
        bu_paths = rank_candidates(qid, bu_groups.get(qid, []), score_map=score_map, top_k=args.candidate_topn, retriever_name="hyp_bottom_up")
        union_paths = rank_candidates(qid, union_groups.get(qid, []), score_map=score_map, top_k=args.candidate_topn, retriever_name="hyp_bottom_up_union_hyp_topdown")
        bu_outputs.append(output_item(qa, bu_paths, args, "BU-Hyp"))
        union_outputs.append(output_item(qa, union_paths, args, "BU-Hyp + TD-Hyp"))
        if index % 500 == 0 or index == len(questions):
            print(f"ranked pure-hyp bottom-up {index}/{len(questions)}", flush=True)

    prefix = resolve_path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, outputs, pools, method in [
        ("bu_hyp", bu_outputs, bu_pools, "BU-Hyp"),
        ("bu_hyp_td_hyp", union_outputs, union_pools, "BU-Hyp + TD-Hyp"),
    ]:
        path = Path(f"{prefix}_{suffix}_paths.json")
        eval_path = Path(f"{prefix}_{suffix}_eval.json")
        write_json(outputs, path)
        payload = helpers.evaluation_payload(graph, outputs, pools, args.candidate_topn, method=method)
        write_json(payload, eval_path)
        print(f"{method} summary={payload['summary']}")
        print(f"{method} candidate_pool={payload['candidate_pool']}")
        print(f"wrote {path}")


def top_by_lorentz_fact_score(
    node_ids: list[str],
    index: dict[str, int],
    points: np.ndarray,
    query_point: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    valid_node_ids = [node_id for node_id in node_ids if node_id in index]
    if not valid_node_ids or top_k <= 0:
        return []
    indices = np.asarray([index[node_id] for node_id in valid_node_ids], dtype=np.int64)
    scores = -lorentz_distance_numpy(points[indices], query_point)
    limit = min(top_k, len(scores))
    if limit >= len(scores):
        local_order = np.argsort(-scores)
    else:
        local_order = np.argpartition(-scores, limit - 1)[:limit]
        local_order = local_order[np.argsort(-scores[local_order])]
    return [(valid_node_ids[int(pos)], float(scores[pos])) for pos in local_order]


def overlap_score(query_terms: set[str], text: str) -> float:
    node_terms = set(content_terms(text))
    if not query_terms or not node_terms:
        return 0.0
    return min(1.0, len(query_terms & node_terms) / max(len(query_terms), 1))


def candidates_from_paths(graph: MemoryGraph, paths: list[dict], *, route: str) -> list[RerankCandidate]:
    candidates = []
    seen = set()
    for rank, path in enumerate(paths, start=1):
        node_id = evidence_node_id(path)
        if not node_id or node_id in seen or node_id not in graph.nodes:
            continue
        seen.add(node_id)
        metadata = dict(path.get("metadata", {}))
        metadata[f"{route}_route_rank"] = str(rank)
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


def merge_candidates(left: list[RerankCandidate], right: list[RerankCandidate], topn: int) -> list[RerankCandidate]:
    merged: dict[str, RerankCandidate] = {}
    for candidate in [*left, *right]:
        node_id = candidate.node.node_id
        previous = merged.get(node_id)
        if previous is None:
            merged[node_id] = copy_candidate(candidate)
        else:
            previous.metadata = merge_route_metadata(previous.metadata or {}, candidate.metadata or {})
            previous.base_score = max(previous.base_score, candidate.base_score)
    rows = list(merged.values())
    rows.sort(key=lambda candidate: candidate.base_score, reverse=True)
    return rows[:topn]


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
    if not score_map:
        ranked = [(candidate, candidate.base_score) for candidate in candidates[:top_k]]
    else:
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
            "checkpoint": str(resolve_path(args.checkpoint)),
            "embedding_cache": str(resolve_path(args.embedding_cache)),
            "seed_topn": args.seed_topn,
            "preselect_topn": args.preselect_topn,
            "candidate_topn": args.candidate_topn,
            "ce_model": None if args.skip_ce else args.ce_model,
        },
    }


def evidence_node_id(path: dict) -> str:
    metadata_id = str(path.get("metadata", {}).get("evidence_node_id", ""))
    if metadata_id:
        return metadata_id
    node_ids = path.get("node_ids", [])
    return str(node_ids[-1]) if node_ids else ""


def load_item_map(path: str) -> dict[str, dict]:
    return {str(item["question_id"]): item for item in read_json(resolve_path(path))}


def copy_candidate(candidate: RerankCandidate) -> RerankCandidate:
    return RerankCandidate(
        node=candidate.node,
        base_score=candidate.base_score,
        path_node_ids=list(candidate.path_node_ids or []),
        path_edge_ids=list(candidate.path_edge_ids or []),
        metadata=dict(candidate.metadata or {}),
    )


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
