from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from polyglot_grounded_qa.schemas.config import AppConfig


def build_run_metadata(cfg: AppConfig, language: str) -> dict[str, Any]:
    """Build stable metadata fields for eval/ablation artifacts."""
    config_json = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    return {
        "run_name": cfg.pipeline.run_name,
        "language": language,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
    }