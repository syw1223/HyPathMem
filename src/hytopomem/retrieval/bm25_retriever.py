from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from hytopomem.memory.schema import Node


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass
class BM25Retriever:
    nodes: Sequence[Node]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.docs = [tokenize(node.text) for node in self.nodes]
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avg_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.doc_freq = defaultdict(int)
        for doc in self.docs:
            for token in set(doc):
                self.doc_freq[token] += 1

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Node, float]]:
        q_terms = tokenize(query)
        scores = []
        total_docs = max(len(self.nodes), 1)
        for node, doc, doc_len in zip(self.nodes, self.docs, self.doc_lens):
            counts = Counter(doc)
            score = 0.0
            for term in q_terms:
                if term not in counts:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = counts[term]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-6))
                score += idf * (tf * (self.k1 + 1) / denom)
            if score > 0:
                scores.append((node, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

