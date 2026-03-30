from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa.components.abstain import ThresholdAbstentionPolicy
from polyglot_grounded_qa.components.generator import BaselineGenerator
from polyglot_grounded_qa.components.reranker import BaselineReranker
from polyglot_grounded_qa.components.retriever import BaselineRetriever
from polyglot_grounded_qa.components.verifier import BaselineVerifier
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.pipeline import GroundedQAPipeline
from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


def create_default_pipeline(project_root: str) -> GroundedQAPipeline:
    cfg = load_app_config(project_root=Path(project_root))
    default_language = cfg.languages[cfg.pipeline.default_language]

    seed_corpus = [
        RetrievedChunk(
            doc_id="seed-doc-1",
            chunk_id="seed-chunk-1",
            text="Polyglot grounded QA uses retrieval with citations and verification.",
            score=0.1,
        ),
        RetrievedChunk(
            doc_id="seed-doc-2",
            chunk_id="seed-chunk-2",
            text="Language packs inherit base behavior and only override locale-specific rules.",
            score=0.1,
        ),
    ]

    return GroundedQAPipeline(
        retriever=BaselineRetriever(corpus=seed_corpus),
        reranker=BaselineReranker(),
        generator=BaselineGenerator(),
        verifier=BaselineVerifier(),
        abstention=ThresholdAbstentionPolicy(default_language.thresholds),
        top_k_retrieve=cfg.pipeline.retrieval.top_k_dense,
        top_k_rerank=cfg.pipeline.retrieval.top_k_rerank,
    )
