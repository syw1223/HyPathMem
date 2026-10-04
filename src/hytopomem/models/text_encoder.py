from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


@dataclass
class HashTextEncoder:
    dim: int = 128
    lowercase: bool = True

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = [self.encode_one(text) for text in texts]
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack(vectors)

    def encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float64)
        tokens = self._tokens(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 8) % 2 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def _tokens(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return _TOKEN_RE.findall(text)

