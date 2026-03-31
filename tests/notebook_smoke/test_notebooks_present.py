from __future__ import annotations

from pathlib import Path


def test_notebook_count() -> None:
    root = Path(__file__).resolve().parents[2]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) >= 8


def test_notebook_sequence_is_present() -> None:
    root = Path(__file__).resolve().parents[2]
    notebooks = sorted(path.name for path in (root / "notebooks").glob("*.ipynb"))
    expected = [
        "00_project_story.ipynb",
        "10_data_ingestion.ipynb",
        "20_index_and_retrieval.ipynb",
        "30_citation_generation.ipynb",
        "40_verification_abstention.ipynb",
        "50_eval_harness.ipynb",
        "60_language_packs.ipynb",
        "70_ablations.ipynb",
        "80_final_results.ipynb",
    ]
    for notebook in expected:
        assert notebook in notebooks
