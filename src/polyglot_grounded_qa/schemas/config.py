from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PromptTemplates(BaseModel):
    system: str = "You answer using only cited evidence."
    user: str = "Question: {query}"
    citation_instruction: str = "Cite supporting chunk IDs."


class ThresholdConfig(BaseModel):
    abstain_min_support: float = Field(ge=0.0, le=1.0, default=0.55)
    abstain_max_conflict_ratio: float = Field(ge=0.0, le=1.0, default=0.3)
    verifier_min_confidence: float = Field(ge=0.0, le=1.0, default=0.65)
    graph_min_path_score: float = Field(ge=0.0, le=1.0, default=0.4)
    graph_min_path_count: int = Field(ge=0, default=1)
    hybrid_min_text_evidence: int = Field(ge=0, default=1)
    hybrid_min_text_support: float = Field(ge=0.0, le=1.0, default=0.50)


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
    mode: Literal["text", "graph", "hybrid"] = "text"
    hybrid_policy: Literal["naive", "filtered", "routed"] = "naive"
    backend: str = "faiss_bm25"
    top_k_dense: int = Field(default=30, ge=1)
    top_k_sparse: int = Field(default=30, ge=1)
    top_k_rerank: int = Field(default=15, ge=1)
    graph_top_k: int = Field(default=8, ge=1)
    fusion: str = "weighted_rrf"
    dense_weight: float = Field(default=0.7, ge=0.0)
    sparse_weight: float = Field(default=0.3, ge=0.0)
    text_weight: float = Field(default=0.65, ge=0.0)
    graph_weight: float = Field(default=0.35, ge=0.0)
    graph_max_hops: int = Field(default=2, ge=1)
    graph_min_path_score: float = Field(default=0.35, ge=0.0, le=1.0)
    graph_min_quality_score: float = Field(default=0.45, ge=0.0, le=1.0)
    entity_link_min_score: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> RetrievalConfig:
        total = self.dense_weight + self.sparse_weight
        if total <= 0:
            msg = "dense_weight + sparse_weight must be > 0"
            raise ValueError(msg)
        hybrid_total = self.text_weight + self.graph_weight
        if self.mode == "hybrid" and hybrid_total <= 0:
            msg = "text_weight + graph_weight must be > 0 when retrieval mode is hybrid"
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
