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
        "retrieval_mode",
        "hybrid_policy",
        "routing_decision",
        "top_evidence_type",
        "top_chunk_id",
        "text_evidence_count",
        "graph_evidence_count",
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
        "retrieval_mode",
        "hybrid_policy",
        "routing_decision",
        "top_evidence_type",
        "top_chunk_id",
        "text_evidence_count",
        "graph_evidence_count",
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
        "query",
        "query_category",
        "answer",
        "abstained",
        "citation_count",
        "retrieval_mode",
        "hybrid_policy",
        "routing_decision",
        "top_evidence_type",
        "top_chunk_id",
        "text_evidence_count",
        "graph_evidence_count",
    }
    assert expected.issubset(set(df.columns))
    assert {"text-only", "kg-only", "hybrid", "hybrid-path-filtered", "hybrid-routed"}.issubset(
        set(df.get_column("variant").to_list())
    )
    assert {"base", "es", "fr", "tr"}.issubset(set(df.get_column("language").to_list()))


def test_build_kg_cache_script_writes_seed_artifact() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["uv", "run", "python", "scripts/build_kg_cache.py", "--offline", "--refresh"],
        cwd=root,
        check=True,
    )

    index_path = root / "artifacts" / "indexes" / "kg_seed_paths.parquet"
    assert index_path.exists()

    df = pl.read_parquet(index_path)
    assert len(df) >= 4
    assert {
        "path_id",
        "language",
        "path_length",
        "score",
        "path_text",
        "source",
        "triples_json",
    }.issubset(set(df.columns))


def test_analyze_kg_coverage_script_writes_contract_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/analyze_kg_coverage.py"], cwd=root, check=True)

    summary_path = root / "artifacts" / "tables" / "kg_coverage_summary.parquet"
    by_language_path = root / "artifacts" / "tables" / "kg_coverage_by_language.parquet"
    report_path = root / "artifacts" / "tables" / "kg_coverage_report.md"

    assert summary_path.exists()
    assert by_language_path.exists()
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Coverage snapshot" in report_text
    assert "## Failure buckets" in report_text

    summary_df = pl.read_parquet(summary_path)
    assert {
        "failure_bucket",
        "queries",
        "avg_linked_entity_count",
        "avg_returned_path_count",
        "avg_max_path_score",
    }.issubset(set(summary_df.columns))

    by_language_df = pl.read_parquet(by_language_path)
    assert {
        "language",
        "queries",
        "path_yield_rate",
        "avg_linked_entity_count",
    }.issubset(set(by_language_df.columns))


def test_analyze_kg_path_quality_script_writes_contract_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["uv", "run", "python", "scripts/analyze_kg_path_quality.py"], cwd=root, check=True)

    summary_path = root / "artifacts" / "tables" / "kg_path_quality_summary.parquet"
    by_language_path = root / "artifacts" / "tables" / "kg_path_quality_by_language.parquet"
    rows_path = root / "artifacts" / "tables" / "kg_path_quality_rows.parquet"
    report_path = root / "artifacts" / "tables" / "kg_path_quality_report.md"

    assert summary_path.exists()
    assert by_language_path.exists()
    assert rows_path.exists()
    assert report_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Summary" in report_text
    assert "## Sample audited paths" in report_text

    summary_df = pl.read_parquet(summary_path)
    assert {
        "quality_label",
        "leakage_risk",
        "rows",
        "avg_graph_path_score",
    }.issubset(set(summary_df.columns))

    by_language_df = pl.read_parquet(by_language_path)
    assert {
        "language",
        "supporting_path_rate",
        "explicit_restatement_rate",
        "high_leakage_rate",
    }.issubset(set(by_language_df.columns))


def test_analyze_sft_dataset_script_writes_contract_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/analyze_sft_dataset.py",
            "--input",
            "data/benchmarks/finetune/sft_dataset_merged.jsonl",
        ],
        cwd=root,
        check=True,
    )

    report_path = root / "artifacts" / "tables" / "sft_dataset_quality_report.md"
    by_language_path = root / "artifacts" / "tables" / "sft_dataset_quality_by_language.parquet"
    by_split_path = root / "artifacts" / "tables" / "sft_dataset_quality_by_split.parquet"

    assert report_path.exists()
    assert by_language_path.exists()
    assert by_split_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Contract checks" in report_text
    assert "## By split" in report_text

    by_language_df = pl.read_parquet(by_language_path)
    assert {"language", "rows", "abstain_rate", "citation_validity_rate"}.issubset(
        set(by_language_df.columns)
    )


def test_run_finetune_eval_script_writes_diagnostics_outputs() -> None:
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["uv", "run", "python", "scripts/run_finetune_eval.py", "--variant", "baseline-pipeline"],
        cwd=root,
        check=True,
    )

    summary_path = root / "artifacts" / "tables" / "finetune_eval_summary.parquet"
    by_language_path = root / "artifacts" / "tables" / "finetune_eval_by_language.parquet"
    by_label_type_path = root / "artifacts" / "tables" / "finetune_eval_by_label_type.parquet"
    report_path = root / "artifacts" / "tables" / "finetune_eval_diagnostics.md"

    assert summary_path.exists()
    assert by_language_path.exists()
    assert by_label_type_path.exists()
    assert report_path.exists()

    summary_df = pl.read_parquet(summary_path)
    assert {
        "variant",
        "abstain_accuracy",
        "abstain_precision",
        "abstain_recall",
        "false_abstain_rate",
        "missed_abstain_rate",
        "grounded_trust_score",
    }.issubset(set(summary_df.columns))
