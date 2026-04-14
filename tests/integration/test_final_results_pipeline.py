from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl


def test_materialize_final_results_and_contract_core() -> None:
    root = Path(__file__).resolve().parents[2]

    subprocess.run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root, check=True)
    subprocess.run(["uv", "run", "python", "scripts/run_ablation.py"], cwd=root, check=True)
    subprocess.run(["uv", "run", "python", "scripts/analyze_kg_coverage.py"], cwd=root, check=True)
    subprocess.run(["uv", "run", "python", "scripts/analyze_kg_path_quality.py"], cwd=root, check=True)

    subprocess.run(
        ["uv", "run", "python", "scripts/materialize_final_results.py"],
        cwd=root,
        check=True,
    )

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/check_final_artifacts_contract.py",
            "--core-only",
        ],
        cwd=root,
        check=True,
    )

    assert (root / "artifacts/tables/final_eval_overview.parquet").exists()
    assert (root / "artifacts/tables/final_ablation_summary.parquet").exists()
    assert (root / "artifacts/tables/final_hybrid_summary.parquet").exists()
    assert (root / "artifacts/tables/final_hybrid_takeaways.md").exists()
    assert (root / "artifacts/tables/final_repro_diagnostics.parquet").exists()
    assert (root / "artifacts/tables/final_reader_takeaways.md").exists()

    hybrid_summary = pl.read_parquet(root / "artifacts/tables/final_hybrid_summary.parquet")
    assert {
        "language",
        "kg_path_yield_rate",
        "supporting_path_rate",
        "high_leakage_rate",
        "delta_hybrid_filtered_support_minus_hybrid",
        "delta_hybrid_routed_support_minus_hybrid",
        "delta_hybrid_routed_graph_top_rate_minus_hybrid",
        "hybrid_filtered_graph_filter_fallback_rate",
        "hybrid_routed_graph_first_adherence_rate",
        "hybrid_routed_text_first_adherence_rate",
    }.issubset(set(hybrid_summary.columns))
