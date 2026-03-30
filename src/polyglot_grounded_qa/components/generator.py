from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import Citation, GroundedAnswer, RetrievedChunk


@dataclass(slots=True)
class BaselineGenerator:
    """Deterministic placeholder generator with citation wiring."""

    def generate(self, query: str, chunks: list[RetrievedChunk], language: str) -> GroundedAnswer:
        _ = language
        if not chunks:
            return GroundedAnswer(answer="I do not have enough evidence.", abstained=True)

        top = chunks[0]
        answer = f"Based on the retrieved evidence, relevant context for '{query}' is: {top.text}"
        citation = Citation(doc_id=top.doc_id, chunk_id=top.chunk_id)
        return GroundedAnswer(
            answer=answer, citations=[citation], metadata={"generator": "baseline"}
        )
