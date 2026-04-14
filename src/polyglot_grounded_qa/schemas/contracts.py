from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    span: str | None = None


class Claim(BaseModel):
    text: str = Field(min_length=1)
    supported: bool | None = None
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float | None = None


class KnowledgeGraphTriple(BaseModel):
    subject: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)


class KnowledgeGraphPath(BaseModel):
    path_id: str = Field(min_length=1)
    triples: list[KnowledgeGraphTriple] = Field(min_length=1)
    score: float = Field(ge=0.0)
    languages: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def render_text(self) -> str:
        return " | ".join(
            f"{triple.subject} {triple.relation} {triple.object}" for triple in self.triples
        )

    def to_retrieved_chunk(self, score: float | None = None) -> RetrievedChunk:
        chunk_score = self.score if score is None else score
        return RetrievedChunk(
            doc_id=f"kg::{self.path_id}",
            chunk_id=self.path_id,
            text=self.render_text(),
            score=chunk_score,
            metadata={
                **self.metadata,
                "evidence_type": "graph",
                "languages": self.languages,
                "graph_path_score": chunk_score,
                "path_length": len(self.triples),
            },
        )


class GroundedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    abstained: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class PipelineState:
    query: str
    language: str
    chunks: list[RetrievedChunk]
    answer: GroundedAnswer | None = None
