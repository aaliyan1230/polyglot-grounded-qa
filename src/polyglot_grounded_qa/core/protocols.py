from __future__ import annotations

from typing import Protocol

from polyglot_grounded_qa.schemas.contracts import Claim, GroundedAnswer, RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, language: str, k: int) -> list[RetrievedChunk]: ...


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]: ...


class Generator(Protocol):
    def generate(
        self, query: str, chunks: list[RetrievedChunk], language: str
    ) -> GroundedAnswer: ...


class Verifier(Protocol):
    def verify(
        self, claims: list[Claim], chunks: list[RetrievedChunk], language: str
    ) -> list[Claim]: ...


class AbstentionPolicy(Protocol):
    def should_abstain(self, answer: GroundedAnswer) -> bool: ...
