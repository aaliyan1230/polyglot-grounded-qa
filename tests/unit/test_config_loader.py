from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa.core.config_loader import load_app_config


def test_load_app_config_language_inheritance() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_app_config(root)

    assert "base" in cfg.languages
    assert "es-MX" in cfg.languages
    assert cfg.languages["es-MX"].parent == "es"
    assert cfg.languages["es-MX"].prompts.system
    assert cfg.pipeline.retrieval.mode == "text"
    assert cfg.pipeline.retrieval.hybrid_policy == "naive"
    assert cfg.pipeline.retrieval.graph_top_k >= 1
