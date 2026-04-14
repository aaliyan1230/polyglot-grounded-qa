from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

CORE_ARTIFACT_COLUMNS: dict[str, set[str]] = {
    "artifacts/tables/final_eval_overview.parquet": {
        "run_name",
        "language",
        "config_hash",
        "retrieval_mode",
        "question_count",
        "abstained_count",
        "avg_citations",
        "abstain_rate",
        "avg_text_evidence_count",
        "avg_graph_evidence_count",
        "avg_graph_support_score",
    },
    "artifacts/tables/final_ablation_summary.parquet": {
        "run_name",
        "language",
        "variant",
        "retrieval_mode",
        "sample_count",
        "abstained_count",
        "avg_citations",
        "abstain_rate",
        "avg_text_evidence_count",
        "avg_graph_evidence_count",
        "avg_graph_support_score",
    },
    "artifacts/tables/final_hybrid_summary.parquet": {
        "language",
        "text_avg_graph_evidence_count",
        "kg_only_avg_graph_evidence_count",
        "hybrid_avg_graph_evidence_count",
        "hybrid_avg_graph_support_score",
        "kg_path_yield_rate",
        "supporting_path_rate",
        "high_leakage_rate",
        "hybrid_filtered_graph_filter_fallback_rate",
        "hybrid_routed_graph_first_adherence_rate",
        "hybrid_routed_text_first_adherence_rate",
        "delta_hybrid_routed_graph_top_rate_minus_hybrid",
    },
    "artifacts/tables/final_repro_diagnostics.parquet": {"metric", "value"},
}

FINETUNE_ARTIFACT_COLUMNS: dict[str, set[str]] = {
    "artifacts/tables/final_finetune_eval_summary.parquet": {
        "variant",
        "timestamp_utc",
        "rows",
        "abstain_accuracy",
        "avg_citation_precision",
        "avg_citation_recall",
        "avg_answer_token_f1",
        "grounded_trust_score",
    },
    "artifacts/tables/final_finetune_eval_deltas.parquet": {
        "variant",
        "timestamp_utc",
        "rows",
        "delta_abstain_accuracy",
        "delta_avg_citation_precision",
        "delta_avg_citation_recall",
        "delta_avg_answer_token_f1",
        "delta_grounded_trust_score",
    },
    "artifacts/tables/finetune_variant_leaderboard.parquet": {
        "variant",
        "is_practical_variant",
        "passes_grounding_gate",
        "delta_grounded_trust_score",
        "delta_avg_answer_token_f1",
    },
}

TEXT_ARTIFACTS = [
    "artifacts/tables/final_reader_takeaways.md",
    "artifacts/tables/final_hybrid_takeaways.md",
]


PROMOTION_GATE_SPEC = {
    "delta_avg_citation_precision": "> 0.0",
    "delta_avg_citation_recall": "> 0.0",
    "delta_grounded_trust_score": "> 0.0",
    "delta_avg_answer_token_f1": ">= 0.0",
    "delta_abstain_accuracy": ">= 0.0",
}


def _check_parquet_columns(path: Path, required: set[str]) -> str | None:
    if not path.exists():
        return f"missing file: {path}"
    df = pl.read_parquet(path)
    missing_cols = required.difference(set(df.columns))
    if missing_cols:
        return f"missing columns in {path}: {sorted(missing_cols)}"
    return None


def _check_leaderboard_gate(path: Path) -> str | None:
    if not path.exists():
        return f"missing file: {path}"

    df = pl.read_parquet(path)
    needed = set(PROMOTION_GATE_SPEC.keys())
    missing = needed.difference(set(df.columns))
    if missing:
        return f"cannot validate gate in {path}: missing columns {sorted(missing)}"

    expected_gate = (
        pl.col("is_practical_variant")
        &
        (pl.col("delta_avg_citation_precision") > 0.0)
        & (pl.col("delta_avg_citation_recall") > 0.0)
        & (pl.col("delta_grounded_trust_score") > 0.0)
        & (pl.col("delta_avg_answer_token_f1") >= 0.0)
        & (pl.col("delta_abstain_accuracy") >= 0.0)
    )
    mismatched = df.filter(pl.col("passes_grounding_gate") != expected_gate)
    if mismatched.height > 0:
        return (
            "leaderboard gate flag mismatch: passes_grounding_gate does not match promotion-gate rules "
            f"for {mismatched.height} row(s) in {path}"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final result artifact contracts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Validate only non-finetune final artifacts.",
    )
    parser.add_argument(
        "--require-finetune",
        action="store_true",
        help="Require all finetune final artifacts to be present and valid.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    errors: list[str] = []

    for rel_path, required_cols in CORE_ARTIFACT_COLUMNS.items():
        err = _check_parquet_columns(root / rel_path, required_cols)
        if err:
            errors.append(err)

    for rel_path in TEXT_ARTIFACTS:
        p = root / rel_path
        if not p.exists():
            errors.append(f"missing file: {p}")

    finetune_present = all((root / rel).exists() for rel in FINETUNE_ARTIFACT_COLUMNS)
    should_check_finetune = (not args.core_only) and (args.require_finetune or finetune_present)

    if args.require_finetune and not finetune_present:
        missing = [rel for rel in FINETUNE_ARTIFACT_COLUMNS if not (root / rel).exists()]
        errors.append(f"finetune artifacts required but missing: {missing}")

    if should_check_finetune:
        for rel_path, required_cols in FINETUNE_ARTIFACT_COLUMNS.items():
            err = _check_parquet_columns(root / rel_path, required_cols)
            if err:
                errors.append(err)

        gate_err = _check_leaderboard_gate(root / "artifacts/tables/finetune_variant_leaderboard.parquet")
        if gate_err:
            errors.append(gate_err)

    if errors:
        print("Final artifact contract check FAILED:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)

    print("Final artifact contract check PASSED")
    print(f"Project root: {root}")
    if should_check_finetune:
        print("Validated: core + finetune artifact contracts")
    else:
        print("Validated: core artifact contracts")


if __name__ == "__main__":
    main()
