from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


@dataclass(slots=True)
class BaselineRetriever:
    """Minimal retriever stub for notebook and test plumbing."""

    corpus: list[RetrievedChunk]

    def retrieve(self, query: str, language: str, k: int) -> list[RetrievedChunk]:
        _ = language
        query_terms = set(query.lower().split())
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in self.corpus:
            overlap = len(query_terms.intersection(chunk.text.lower().split()))
            score = float(overlap) + chunk.score
            scored.append((score, chunk.model_copy(update={"score": score})))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
