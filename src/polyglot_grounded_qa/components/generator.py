from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from polyglot_grounded_qa.schemas.contracts import Citation, GroundedAnswer, RetrievedChunk


class TextCompletionAdapter(Protocol):
    def complete(self, prompt: str) -> str: ...


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


@dataclass(slots=True)
class AdapterBackedGenerator:
    """Generator that uses an adapter completion API and preserves citation bindings."""

    adapter: TextCompletionAdapter
    name: str

    def generate(self, query: str, chunks: list[RetrievedChunk], language: str) -> GroundedAnswer:
        _ = language
        if not chunks:
            return GroundedAnswer(answer="I do not have enough evidence.", abstained=True)

        top = chunks[0]
        citation = Citation(doc_id=top.doc_id, chunk_id=top.chunk_id)
        prompt = (
            "Answer the question using only the provided evidence.\n"
            f"Question: {query}\n"
            f"Evidence ({top.chunk_id}): {top.text}\n"
            "Return one concise answer sentence."
        )
        generated = self.adapter.complete(prompt).strip()
        if not generated:
            generated = (
                f"Based on the retrieved evidence, relevant context for '{query}' is: {top.text}"
            )

        return GroundedAnswer(
            answer=generated,
            citations=[citation],
            metadata={"generator": self.name},
        )
