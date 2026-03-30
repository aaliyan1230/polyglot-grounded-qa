from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg: float


def compute_recall_at_k(found_relevant: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(found_relevant / total_relevant, 1.0)
