from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, TypeVar

from hytopomem.memory.node_extractor import content_terms


T = TypeVar("T")


def lexical_redundancy(left: str, right: str) -> float:
    left_terms = set(content_terms(left))
    right_terms = set(content_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


@dataclass(frozen=True)
class MMRItem:
    item_id: str
    text: str
    score: float


def select_mmr(
    items: Sequence[T],
    *,
    score_fn: Callable[[T], float],
    text_fn: Callable[[T], str],
    id_fn: Callable[[T], str],
    k: int,
    lambda_div: float = 0.3,
) -> List[T]:
    if k <= 0:
        return []
    remaining = list(items)
    selected: List[T] = []
    while remaining and len(selected) < k:
        best_item = None
        best_score = float("-inf")
        for item in remaining:
            relevance = score_fn(item)
            redundancy = 0.0
            if selected:
                redundancy = max(lexical_redundancy(text_fn(item), text_fn(chosen)) for chosen in selected)
            mmr_score = relevance - lambda_div * redundancy
            if mmr_score > best_score:
                best_item = item
                best_score = mmr_score
        if best_item is None:
            break
        selected.append(best_item)
        remaining = [item for item in remaining if id_fn(item) != id_fn(best_item)]
    return selected

