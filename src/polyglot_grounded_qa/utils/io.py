from __future__ import annotations

from pathlib import Path

import polars as pl


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.write_parquet(path)
