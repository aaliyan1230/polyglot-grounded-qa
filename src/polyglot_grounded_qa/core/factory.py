from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from polyglot_grounded_qa.adapters.llm.google_adapter import GoogleGenAIAdapter
from polyglot_grounded_qa.adapters.llm.ollama_adapter import OllamaAdapter
from polyglot_grounded_qa.components.abstain import ThresholdAbstentionPolicy
from polyglot_grounded_qa.components.generator import AdapterBackedGenerator, BaselineGenerator
from polyglot_grounded_qa.components.reranker import BaselineReranker
from polyglot_grounded_qa.components.retriever import BaselineRetriever
from polyglot_grounded_qa.components.verifier import BaselineVerifier
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.pipeline import GroundedQAPipeline
from polyglot_grounded_qa.core.seed_data import get_seed_corpus


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


def create_default_pipeline(project_root: str) -> GroundedQAPipeline:
    cfg = load_app_config(project_root=Path(project_root))
    default_language = cfg.languages[cfg.pipeline.default_language]
    seed_corpus = get_seed_corpus()
    generator = _select_generator(cfg.models)

    return GroundedQAPipeline(
        retriever=BaselineRetriever(corpus=seed_corpus),
        reranker=BaselineReranker(),
        generator=generator,
        verifier=BaselineVerifier(),
        abstention=ThresholdAbstentionPolicy(default_language.thresholds),
        top_k_retrieve=cfg.pipeline.retrieval.top_k_dense,
        top_k_rerank=cfg.pipeline.retrieval.top_k_rerank,
    )
