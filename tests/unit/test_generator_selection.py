from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa import create_default_pipeline


def test_default_backend_is_baseline(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("PGQA_GENERATOR_BACKEND", raising=False)

    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is grounded QA?", language="base")

    assert result.metadata.get("generator") == "baseline"


def test_unknown_backend_falls_back_to_baseline(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PGQA_GENERATOR_BACKEND", "unknown")

    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is grounded QA?", language="base")

    assert result.metadata.get("generator") == "baseline"


def test_google_backend_without_key_falls_back_content(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PGQA_GENERATOR_BACKEND", "google")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is grounded QA?", language="base")

    assert result.metadata.get("generator") == "google"
    assert "Based on the retrieved evidence" in result.answer


def test_ollama_backend_without_runtime_falls_back_content(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("PGQA_GENERATOR_BACKEND", "ollama")

    pipeline = create_default_pipeline(str(root))
    result = pipeline.run("What is grounded QA?", language="base")

    assert result.metadata.get("generator") == "ollama"
    assert "Based on the retrieved evidence" in result.answer
