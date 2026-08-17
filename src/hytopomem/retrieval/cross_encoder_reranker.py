from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from hytopomem.memory.schema import EvidencePath, Node


@dataclass
class RerankCandidate:
    node: Node
    base_score: float = 0.0
    path_node_ids: list[str] | None = None
    path_edge_ids: list[str] | None = None
    metadata: dict | None = None


class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, device: str | None = None, batch_size: int = 32):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for cross-encoder reranking") from exc

        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = CrossEncoder(model_name_or_path, **kwargs)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[RerankCandidate], top_k: int) -> list[tuple[RerankCandidate, float]]:
        if not candidates:
            return []
        pairs = [(query, candidate.node.text) for candidate in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, [float(score) for score in scores]),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]


def reranked_paths(
    query_id: str,
    ranked: Iterable[tuple[RerankCandidate, float]],
    *,
    retriever_name: str,
) -> list[EvidencePath]:
    paths = []
    for candidate, ce_score in ranked:
        node_ids = candidate.path_node_ids or [candidate.node.node_id]
        edge_ids = candidate.path_edge_ids or []
        metadata = dict(candidate.metadata or {})
        metadata.update(
            {
                "retriever": retriever_name,
                "evidence_node_id": candidate.node.node_id,
                "evidence_node_type": candidate.node.type.value,
            }
        )
        paths.append(
            EvidencePath(
                query_id=query_id,
                anchor_id=None,
                node_ids=node_ids,
                edge_ids=edge_ids,
                score=ce_score,
                scores={"cross_encoder": ce_score, "base": candidate.base_score},
                metadata=metadata,
            )
        )
    return paths

