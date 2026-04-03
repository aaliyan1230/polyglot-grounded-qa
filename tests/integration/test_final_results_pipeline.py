from __future__ import annotations

import subprocess
from pathlib import Path


def test_materialize_final_results_and_contract_core() -> None:
    root = Path(__file__).resolve().parents[2]

    subprocess.run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root, check=True)
    subprocess.run(["uv", "run", "python", "scripts/run_ablation.py"], cwd=root, check=True)

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
    assert (root / "artifacts/tables/final_repro_diagnostics.parquet").exists()
    assert (root / "artifacts/tables/final_reader_takeaways.md").exists()
