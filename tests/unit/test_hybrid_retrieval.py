from __future__ import annotations

from polyglot_grounded_qa.components.retriever import (
    BaselineRetriever,
    HybridRetriever,
    SeedKnowledgeGraphRetriever,
)
from polyglot_grounded_qa.core.seed_data import get_seed_corpus, get_seed_graph_paths
from polyglot_grounded_qa.schemas.config import RetrievalConfig


def test_graph_retriever_returns_graph_chunks() -> None:
    retriever = SeedKnowledgeGraphRetriever(
        paths=get_seed_graph_paths(),
        min_path_score=0.35,
        entity_link_min_score=0.2,
    )

    chunks = retriever.retrieve(
        query="Why are citations required in grounded QA?",
        language="base",
        k=3,
    )

    assert chunks
    assert all(chunk.metadata.get("evidence_type") == "graph" for chunk in chunks)


def test_graph_retriever_reports_no_link_for_unknown_query() -> None:
    retriever = SeedKnowledgeGraphRetriever(
        paths=get_seed_graph_paths(),
        min_path_score=0.35,
        entity_link_min_score=0.2,
    )

    diagnostics = retriever.analyze_query(
        query="Explain nebula thermodynamics.",
        language="base",
        k=3,
    )

    assert diagnostics.failure_bucket == "no-link"
    assert diagnostics.returned_path_count == 0


def test_hybrid_retriever_fuses_text_and_graph_results() -> None:
    cfg = RetrievalConfig.model_validate(
        {
            "mode": "hybrid",
            "top_k_dense": 3,
            "top_k_sparse": 3,
            "top_k_rerank": 3,
            "graph_top_k": 3,
            "text_weight": 0.65,
            "graph_weight": 0.35,
        }
    )
    retriever = HybridRetriever(
        text_retriever=BaselineRetriever(corpus=get_seed_corpus()),
        graph_retriever=SeedKnowledgeGraphRetriever(
            paths=get_seed_graph_paths(),
            min_path_score=0.35,
            entity_link_min_score=0.2,
        ),
        retrieval_cfg=cfg,
    )

    chunks = retriever.retrieve(
        query="How does hybrid retrieval improve grounded QA?",
        language="base",
        k=5,
    )

    evidence_types = {chunk.metadata.get("evidence_type") for chunk in chunks}
    assert "text" in evidence_types
    assert "graph" in evidence_types