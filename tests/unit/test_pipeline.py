from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa import create_default_pipeline


def test_pipeline_returns_citations() -> None:
    root = Path(__file__).resolve().parents[2]
    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is locale inheritance?", language="base")
    assert result.answer
    assert isinstance(result.abstained, bool)
