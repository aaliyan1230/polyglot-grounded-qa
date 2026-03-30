from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from polyglot_grounded_qa.schemas.config import (
    AppConfig,
    LanguageConfig,
    PathsConfig,
    PipelineConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_language_config(tag: str, raw_by_tag: dict[str, dict[str, Any]]) -> LanguageConfig:
    if tag not in raw_by_tag:
        msg = f"Language config '{tag}' not found"
        raise KeyError(msg)

    raw = raw_by_tag[tag]
    parent = raw.get("parent")

    if parent:
        parent_cfg = _resolve_language_config(parent, raw_by_tag).model_dump()
        merged = _deep_merge(parent_cfg, raw)
    else:
        merged = raw

    merged["tag"] = tag
    return LanguageConfig.model_validate(merged)


def load_app_config(project_root: Path) -> AppConfig:
    configs_dir = project_root / "configs"
    pipeline_cfg = PipelineConfig.model_validate(
        _read_yaml(configs_dir / "pipeline" / "default.yaml")
    )
    models_cfg = _read_yaml(configs_dir / "models" / "default.yaml")

    language_dir = configs_dir / "languages"
    raw_language_files = {
        path.stem.replace("-", "_"): _read_yaml(path) for path in language_dir.glob("*.yaml")
    }
    languages = {
        tag: _resolve_language_config(tag, raw_language_files) for tag in raw_language_files
    }

    app = AppConfig(
        paths=PathsConfig(
            project_root=project_root,
            configs_dir=configs_dir,
            data_dir=project_root / "data",
            artifacts_dir=project_root / "artifacts",
        ),
        pipeline=pipeline_cfg,
        languages=languages,
        models=models_cfg,
    )
    return app
