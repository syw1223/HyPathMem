from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hytopomem.geometry.lorentz import LorentzManifold
from hytopomem.memory.schema import Node, NodeStatus, NodeType
from hytopomem.models.text_encoder import HashTextEncoder


_TYPE_OFFSETS = {
    NodeType.RAW: 0.05,
    NodeType.FACT: 0.2,
    NodeType.EVENT: 0.12,
    NodeType.ANCHOR: -0.15,
}

_STATUS_OFFSETS = {
    NodeStatus.ACTIVE: 0.0,
    NodeStatus.OUTDATED: -0.1,
    NodeStatus.EXCEPTION: 0.08,
    NodeStatus.DISPUTED: -0.05,
}


@dataclass
class HyperbolicMapper:
    """Inductive text-to-Lorentz mapper for the MVP.

    This deterministic mapper keeps the code path compatible with a trainable
    encoder: text/type/status/confidence go in, Lorentz points come out.
    """

    dim: int = 32
    curvature: float = 1.0
    text_encoder: HashTextEncoder | None = None

    def __post_init__(self) -> None:
        if self.dim < 2:
            raise ValueError("dim must include time-like coordinate and be >= 2")
        if self.text_encoder is None:
            self.text_encoder = HashTextEncoder(dim=self.dim - 1)
        self.manifold = LorentzManifold(curvature=self.curvature)

    def encode_nodes(self, nodes: list[Node]) -> dict[str, np.ndarray]:
        return {node.node_id: self.encode_node(node) for node in nodes}

    def encode_node(self, node: Node) -> np.ndarray:
        tangent = self.text_encoder.encode_one(node.text)
        scale = 0.45 + _TYPE_OFFSETS[node.type] + _STATUS_OFFSETS[node.status]
        scale += 0.2 * max(0.0, min(1.0, node.confidence))
        if node.type == NodeType.ANCHOR:
            scale *= 0.75
        return self.manifold.expmap0(tangent * scale)

    def encode_query(self, query: str) -> np.ndarray:
        tangent = self.text_encoder.encode_one(query)
        return self.manifold.expmap0(tangent * 0.55)

