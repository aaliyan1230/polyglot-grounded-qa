from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PromptTemplates(BaseModel):
    system: str = "You answer using only cited evidence."
    user: str = "Question: {query}"
    citation_instruction: str = "Cite supporting chunk IDs."


class ThresholdConfig(BaseModel):
    abstain_min_support: float = Field(ge=0.0, le=1.0, default=0.55)
    abstain_max_conflict_ratio: float = Field(ge=0.0, le=1.0, default=0.3)
    verifier_min_confidence: float = Field(ge=0.0, le=1.0, default=0.65)


class LanguageConfig(BaseModel):
    tag: str
    parent: str | None = None
    display_name: str
    tokenizer: str = "default"
    normalizer: str = "default"
    prompts: PromptTemplates = Field(default_factory=PromptTemplates)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    model_overrides: dict[str, str] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    backend: str = "faiss_bm25"
    top_k_dense: int = Field(default=30, ge=1)
    top_k_sparse: int = Field(default=30, ge=1)
    top_k_rerank: int = Field(default=15, ge=1)
    fusion: str = "weighted_rrf"
    dense_weight: float = Field(default=0.7, ge=0.0)
    sparse_weight: float = Field(default=0.3, ge=0.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "RetrievalConfig":
        total = self.dense_weight + self.sparse_weight
        if total <= 0:
            msg = "dense_weight + sparse_weight must be > 0"
            raise ValueError(msg)
        return self


class PipelineConfig(BaseModel):
    default_language: str = "base"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    run_name: str = "local-run"


class PathsConfig(BaseModel):
    project_root: Path
    configs_dir: Path
    data_dir: Path
    artifacts_dir: Path


class AppConfig(BaseModel):
    paths: PathsConfig
    pipeline: PipelineConfig
    languages: dict[str, LanguageConfig]
    models: dict[str, Any] = Field(default_factory=dict)
