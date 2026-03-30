from __future__ import annotations

from pathlib import Path


def test_notebook_count() -> None:
    root = Path(__file__).resolve().parents[2]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 8
