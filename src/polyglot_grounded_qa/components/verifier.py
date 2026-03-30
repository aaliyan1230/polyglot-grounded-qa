from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.contracts import Claim, RetrievedChunk


@dataclass(slots=True)
class BaselineVerifier:
    """Naive verifier for interface stability during early experiments."""

    def verify(
        self, claims: list[Claim], chunks: list[RetrievedChunk], language: str
    ) -> list[Claim]:
        _ = language
        if not chunks:
            return [c.model_copy(update={"supported": False, "confidence": 0.0}) for c in claims]
        return [c.model_copy(update={"supported": True, "confidence": 0.6}) for c in claims]
