from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.langpacks.registry import build_pack


def test_language_pack_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_app_config(root)

    for tag, lang_cfg in cfg.languages.items():
        pack = build_pack(lang_cfg)
        normalized = pack.normalize("  hello   world  ")
        tokens = pack.tokenize(normalized)
        assert normalized
        assert isinstance(tokens, list)
        assert pack.config.tag == tag
