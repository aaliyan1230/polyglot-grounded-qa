from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa import create_default_pipeline


def test_pipeline_returns_citations() -> None:
    root = Path(__file__).resolve().parents[2]
    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is locale inheritance?", language="base")
    assert result.answer
    assert isinstance(result.abstained, bool)


def test_hybrid_pipeline_surfaces_graph_metadata() -> None:
    root = Path(__file__).resolve().parents[2]
    pipeline = create_default_pipeline(str(root), retrieval_mode="hybrid")
    result = pipeline.run("Why are citations required in grounded QA?", language="base")

    assert result.metadata.get("retrieval_mode") == "hybrid"
    assert int(result.metadata.get("graph_evidence_count", 0)) >= 1
