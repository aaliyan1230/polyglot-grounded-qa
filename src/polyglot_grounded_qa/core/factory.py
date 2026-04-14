from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from polyglot_grounded_qa.adapters.llm.google_adapter import GoogleGenAIAdapter
from polyglot_grounded_qa.adapters.llm.ollama_adapter import OllamaAdapter
from polyglot_grounded_qa.components.abstain import GraphAwareAbstentionPolicy
from polyglot_grounded_qa.components.generator import AdapterBackedGenerator, BaselineGenerator
from polyglot_grounded_qa.components.reranker import BaselineReranker
from polyglot_grounded_qa.components.retriever import (
    BaselineRetriever,
    HybridRetriever,
    SeedKnowledgeGraphRetriever,
)
from polyglot_grounded_qa.components.verifier import BaselineVerifier
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.pipeline import GroundedQAPipeline
from polyglot_grounded_qa.core.seed_data import get_seed_corpus, get_seed_graph_paths
from polyglot_grounded_qa.schemas.config import RetrievalConfig


def _select_generator(models_cfg: dict[str, Any]) -> BaselineGenerator | AdapterBackedGenerator:
    generation_cfg = models_cfg.get("generation", {}) if isinstance(models_cfg, dict) else {}
    backend = os.getenv("PGQA_GENERATOR_BACKEND", generation_cfg.get("backend", "baseline"))

    if backend == "google":
        model_name = generation_cfg.get("google_model", "gemini-2.5-flash")
        adapter = GoogleGenAIAdapter(model=model_name)
        return AdapterBackedGenerator(adapter=adapter, name="google")

    if backend == "ollama":
        model_name = generation_cfg.get("ollama_model", generation_cfg.get("local_default", ""))
        host = os.getenv("PGQA_OLLAMA_HOST", generation_cfg.get("ollama_host", "http://localhost:11434"))
        adapter = OllamaAdapter(model=model_name, host=host)
        return AdapterBackedGenerator(adapter=adapter, name="ollama")

    return BaselineGenerator()


def create_default_pipeline(
    project_root: str, retrieval_mode: str | None = None
) -> GroundedQAPipeline:
    cfg = load_app_config(project_root=Path(project_root))
    default_language = cfg.languages[cfg.pipeline.default_language]
    seed_corpus = get_seed_corpus()
    seed_graph_paths = get_seed_graph_paths()
    generator = _select_generator(cfg.models)
    requested_mode = retrieval_mode or os.getenv(
        "PGQA_RETRIEVAL_MODE", cfg.pipeline.retrieval.mode
    )
    retrieval_cfg = RetrievalConfig.model_validate(
        {**cfg.pipeline.retrieval.model_dump(), "mode": requested_mode}
    )
    text_retriever = BaselineRetriever(corpus=seed_corpus)
    graph_retriever = SeedKnowledgeGraphRetriever(
        paths=seed_graph_paths,
        min_path_score=retrieval_cfg.graph_min_path_score,
        entity_link_min_score=retrieval_cfg.entity_link_min_score,
    )

    if retrieval_cfg.mode == "graph":
        top_k_retrieve = retrieval_cfg.graph_top_k
    elif retrieval_cfg.mode == "hybrid":
        top_k_retrieve = max(retrieval_cfg.top_k_dense, retrieval_cfg.graph_top_k)
    else:
        top_k_retrieve = retrieval_cfg.top_k_dense

    return GroundedQAPipeline(
        retriever=HybridRetriever(
            text_retriever=text_retriever,
            graph_retriever=graph_retriever,
            retrieval_cfg=retrieval_cfg,
        ),
        reranker=BaselineReranker(),
        generator=generator,
        verifier=BaselineVerifier(),
        abstention=GraphAwareAbstentionPolicy(default_language.thresholds),
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=retrieval_cfg.top_k_rerank,
    )
