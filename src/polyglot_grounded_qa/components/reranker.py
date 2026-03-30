from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


@dataclass(slots=True)
class BaselineReranker:
    """Simple score sorter to keep interfaces stable."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        _ = query
        ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ordered[:k]
