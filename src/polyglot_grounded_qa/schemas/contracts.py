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
