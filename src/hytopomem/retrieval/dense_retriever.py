from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from hytopomem.memory.schema import Node
from hytopomem.models.text_encoder import HashTextEncoder


@dataclass
class DenseRetriever:
    nodes: Sequence[Node]
    encoder: HashTextEncoder

    def __post_init__(self) -> None:
        self.matrix = self.encoder.encode([node.text for node in self.nodes])

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Node, float]]:
        if not self.nodes:
            return []
        q = self.encoder.encode_one(query)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [(self.nodes[int(idx)], float(scores[int(idx)])) for idx in order if scores[int(idx)] > 0]

