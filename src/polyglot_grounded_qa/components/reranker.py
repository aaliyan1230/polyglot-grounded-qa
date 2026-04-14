from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


def _route_preference(chunk: RetrievedChunk) -> int:
    routing_decision = str(chunk.metadata.get("routing_decision", "balanced"))
    evidence_type = str(chunk.metadata.get("evidence_type", "text"))
    if routing_decision == "graph-first":
        return 1 if evidence_type == "graph" else 0
    if routing_decision == "text-first":
        return 1 if evidence_type == "text" else 0
    return 0


@dataclass(slots=True)
class BaselineReranker:
    """Simple score sorter to keep interfaces stable."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        _ = query
        ordered = sorted(
            chunks,
            key=lambda chunk: (_route_preference(chunk), chunk.score),
            reverse=True,
        )
        return ordered[:k]
