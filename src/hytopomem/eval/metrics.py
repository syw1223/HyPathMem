from __future__ import annotations

from typing import Iterable, Sequence


def evidence_recall_at_k(predicted_support_ids: Sequence[str], gold_evidence: Iterable[str], k: int) -> float:
    gold = set(gold_evidence)
    if not gold:
        return 0.0
    pred = set(predicted_support_ids[:k])
    return len(pred & gold) / len(gold)


def exact_match(prediction: str, answer: str) -> float:
    return float(prediction.strip().lower() == answer.strip().lower())

