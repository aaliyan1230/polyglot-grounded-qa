from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl


def test_run_pipeline_script() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline.py",
        "What is grounded QA?",
        "--language",
        "base",
    ]
    subprocess.run(cmd, cwd=root, check=True)


def test_build_index_script_writes_seed_artifact() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/build_index.py"], cwd=root, check=True)

    index_path = root / "artifacts" / "indexes" / "seed_chunks.parquet"
    assert index_path.exists()

    df = pl.read_parquet(index_path)
    assert len(df) >= 2
    assert {"doc_id", "chunk_id", "text", "score"}.issubset(set(df.columns))


def test_run_eval_script_writes_expected_columns() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root, check=True)

    eval_path = root / "artifacts" / "runs" / "eval_results.parquet"
    assert eval_path.exists()

    df = pl.read_parquet(eval_path)
    expected = {
        "run_name",
        "language",
        "timestamp_utc",
        "config_hash",
        "query",
        "answer",
        "abstained",
        "citation_count",
    }
    assert expected.issubset(set(df.columns))


def test_run_eval_script_is_reproducible_on_key_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root, check=True)
    first = pl.read_parquet(root / "artifacts" / "runs" / "eval_results.parquet")

    subprocess.run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root, check=True)
    second = pl.read_parquet(root / "artifacts" / "runs" / "eval_results.parquet")

    assert set(first.columns) == set(second.columns)
    assert len(first) == len(second)

    stable_cols = [
        "run_name",
        "language",
        "config_hash",
        "query",
        "answer",
        "abstained",
        "citation_count",
    ]
    first_stable = first.select(stable_cols).sort("query")
    second_stable = second.select(stable_cols).sort("query")
    assert first_stable.to_dicts() == second_stable.to_dicts()


def test_run_ablation_script_writes_expected_columns() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/run_ablation.py"], cwd=root, check=True)

    ablation_path = root / "artifacts" / "tables" / "ablation_results.parquet"
    assert ablation_path.exists()

    df = pl.read_parquet(ablation_path)
    expected = {
        "run_name",
        "language",
        "timestamp_utc",
        "config_hash",
        "variant",
        "answer",
        "abstained",
        "citation_count",
    }
    assert expected.issubset(set(df.columns))
