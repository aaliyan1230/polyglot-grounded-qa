from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


@dataclass(slots=True)
class FaissBm25Adapter:
    """Placeholder adapter for local FAISS + BM25 hybrid retrieval."""

    def build_index(self, chunks: list[RetrievedChunk]) -> None:
        _ = chunks

    def query(self, query: str, language: str, k: int) -> list[RetrievedChunk]:
        _ = query, language, k
        return []
