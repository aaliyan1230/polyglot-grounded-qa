from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def _ensure_grounded_trust_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "abstain_accuracy",
        "avg_citation_precision",
        "avg_citation_recall",
        "avg_answer_token_f1",
    ]
    if all(col in out.columns for col in required):
        computed = (
            0.2 * out["abstain_accuracy"].astype(float)
            + 0.3 * out["avg_citation_precision"].astype(float)
            + 0.3 * out["avg_citation_recall"].astype(float)
            + 0.2 * out["avg_answer_token_f1"].astype(float)
        )
        if "grounded_trust_score" not in out.columns:
            out["grounded_trust_score"] = computed
        else:
            out["grounded_trust_score"] = out["grounded_trust_score"].fillna(computed)
    return out


def _write_reader_takeaways(output_dir: Path) -> None:
    deltas_path = output_dir / "final_finetune_eval_deltas.parquet"
    leaderboard_path = output_dir / "finetune_variant_leaderboard.parquet"

    takeaway_lines = [
        "## 7) Reader takeaway summary",
        "",
        "This section converts the latest delta table into plain-language findings.",
        "",
    ]

    if deltas_path.exists():
        latest_deltas = pd.read_parquet(deltas_path)

        sort_col = (
            "delta_grounded_trust_score"
            if "delta_grounded_trust_score" in latest_deltas.columns
            else "delta_avg_answer_token_f1"
        )
        latest_deltas = latest_deltas.sort_values(sort_col, ascending=False)

        if latest_deltas.empty:
            takeaway_lines.append("- No comparison variants are available yet.")
        else:
            best_overall = latest_deltas.iloc[0]
            best_overall_delta = float(best_overall.get(sort_col, 0.0))
            takeaway_lines.append(
                f"- Highest {sort_col} (including diagnostics): **{best_overall['variant']}** "
                f"($\\Delta$ = {best_overall_delta:.4f})."
            )

            practical_variant = None
            if leaderboard_path.exists():
                leaderboard = pd.read_parquet(leaderboard_path)
                gated = leaderboard[
                    (leaderboard["is_practical_variant"] == True)  # noqa: E712
                    & (leaderboard["passes_grounding_gate"] == True)  # noqa: E712
                ]
                if not gated.empty:
                    practical_variant = str(gated.iloc[0]["variant"])

            if practical_variant is not None:
                best_practical = latest_deltas[latest_deltas["variant"] == practical_variant].head(1)
                if not best_practical.empty:
                    best_practical = best_practical.iloc[0]
                    trust_delta = float(best_practical.get("delta_grounded_trust_score", 0.0))
                    f1_delta = float(best_practical.get("delta_avg_answer_token_f1", 0.0))
                    abstain_delta = float(best_practical.get("delta_abstain_accuracy", 0.0))
                    cit_p_delta = float(best_practical.get("delta_avg_citation_precision", 0.0))
                    cit_r_delta = float(best_practical.get("delta_avg_citation_recall", 0.0))
                    takeaway_lines.append(
                        f"- Recommended practical variant today (leaderboard gate-pass): **{best_practical['variant']}** "
                        f"($\\Delta$ trust = {trust_delta:.4f}, $\\Delta$ F1 = {f1_delta:.4f}, "
                        f"$\\Delta$ abstain = {abstain_delta:.4f}, "
                        f"$\\Delta$ citation precision = {cit_p_delta:.4f}, "
                        f"$\\Delta$ citation recall = {cit_r_delta:.4f})."
                    )
            else:
                non_oracle = latest_deltas[
                    ~latest_deltas["variant"].astype(str).str.contains("oracle", case=False, na=False)
                ]
                if not non_oracle.empty:
                    best_non_oracle = non_oracle.iloc[0]
                    takeaway_lines.append(
                        f"- Best non-oracle variant available: **{best_non_oracle['variant']}**."
                    )

            meaningful = latest_deltas[
                (latest_deltas.get("delta_avg_citation_precision", 0) > 0)
                & (latest_deltas.get("delta_avg_citation_recall", 0) > 0)
                & (latest_deltas.get("delta_avg_answer_token_f1", 0) > 0)
            ]
            if meaningful.empty:
                takeaway_lines.append(
                    "- No variant currently improves citation precision, citation recall, and answer F1 simultaneously."
                )
            else:
                names = ", ".join(meaningful["variant"].tolist())
                takeaway_lines.append(
                    f"- Variants with simultaneous gains on grounding and answer quality: {names}."
                )
    else:
        takeaway_lines.append("- Delta artifact not found. Run finetune evaluation first.")

    takeaway_lines.extend(
        [
            "",
            "### Where we stand",
            "- The evaluation loop is reproducible end-to-end and tracks a composite grounded trust score.",
            "- Practical ranking is artifact-backed via finetune_variant_leaderboard outputs.",
            "- Next milestone is to add and compare a real trained-adapter row under this same trust-first rubric.",
        ]
    )

    takeaway_text = "\n".join(takeaway_lines)
    out_path = output_dir / "final_reader_takeaways.md"
    out_path.write_text(takeaway_text + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize final result tables from eval/ablation/finetune artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument(
        "--require-finetune",
        action="store_true",
        help="Fail if finetune artifacts are missing.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    eval_path = root / "artifacts" / "runs" / "eval_results.parquet"
    ablation_path = root / "artifacts" / "tables" / "ablation_results.parquet"
    output_dir = root / "artifacts" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists() or not ablation_path.exists():
        raise FileNotFoundError(
            "Missing eval/ablation artifacts. Run scripts/run_eval.py and scripts/run_ablation.py first."
        )

    con = duckdb.connect()

    eval_overview = con.execute(
        f"""
        SELECT
            run_name,
            language,
            config_hash,
            COUNT(*) AS question_count,
            SUM(CASE WHEN abstained THEN 1 ELSE 0 END) AS abstained_count,
            AVG(citation_count)::DOUBLE AS avg_citations
        FROM read_parquet('{eval_path}')
        GROUP BY run_name, language, config_hash
        ORDER BY run_name, language
        """
    ).df()
    eval_overview["abstain_rate"] = (
        eval_overview["abstained_count"] / eval_overview["question_count"]
    ).fillna(0.0)

    ablation_summary = con.execute(
        f"""
        SELECT
            run_name,
            language,
            variant,
            COUNT(*) AS sample_count,
            SUM(CASE WHEN abstained THEN 1 ELSE 0 END) AS abstained_count,
            AVG(citation_count)::DOUBLE AS avg_citations
        FROM read_parquet('{ablation_path}')
        GROUP BY run_name, language, variant
        ORDER BY run_name, language, variant
        """
    ).df()
    ablation_summary["abstain_rate"] = (
        ablation_summary["abstained_count"] / ablation_summary["sample_count"]
    ).fillna(0.0)

    pivot = ablation_summary.pivot(
        index=["run_name", "language"],
        columns="variant",
        values=["abstain_rate", "avg_citations"],
    )
    pivot.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).strip("_")
        for col in pivot.columns.to_flat_index()
    ]
    if (
        "abstain_rate_baseline" in pivot.columns
        and "abstain_rate_no_rerank" in pivot.columns
    ):
        pivot["delta_abstain_rate_no_rerank_minus_baseline"] = (
            pivot["abstain_rate_no_rerank"] - pivot["abstain_rate_baseline"]
        )
    if "avg_citations_baseline" in pivot.columns and "avg_citations_no_rerank" in pivot.columns:
        pivot["delta_avg_citations_no_rerank_minus_baseline"] = (
            pivot["avg_citations_no_rerank"] - pivot["avg_citations_baseline"]
        )

    eval_hashes = set(
        con.execute(f"SELECT DISTINCT config_hash FROM read_parquet('{eval_path}')")
        .fetchdf()["config_hash"]
        .tolist()
    )
    ablation_hashes = set(
        con.execute(f"SELECT DISTINCT config_hash FROM read_parquet('{ablation_path}')")
        .fetchdf()["config_hash"]
        .tolist()
    )
    shared_hashes = eval_hashes.intersection(ablation_hashes)

    diagnostics = pd.DataFrame(
        [
            {"metric": "eval_config_hash_count", "value": str(len(eval_hashes))},
            {"metric": "ablation_config_hash_count", "value": str(len(ablation_hashes))},
            {"metric": "shared_config_hash_count", "value": str(len(shared_hashes))},
            {"metric": "config_hash_aligned", "value": str(len(shared_hashes) > 0)},
        ]
    )

    eval_overview.to_parquet(output_dir / "final_eval_overview.parquet", index=False)
    ablation_summary.to_parquet(output_dir / "final_ablation_summary.parquet", index=False)
    pivot.to_parquet(output_dir / "final_ablation_deltas.parquet", index=False)
    diagnostics.to_parquet(output_dir / "final_repro_diagnostics.parquet", index=False)

    finetune_summary_path = output_dir / "finetune_eval_summary.parquet"
    finetune_by_language_path = output_dir / "finetune_eval_by_language.parquet"

    if finetune_summary_path.exists():
        finetune_summary = con.execute(
            f"SELECT * FROM read_parquet('{finetune_summary_path}') ORDER BY timestamp_utc DESC"
        ).df()
        finetune_summary = _ensure_grounded_trust_score(finetune_summary)
        finetune_summary.to_parquet(output_dir / "final_finetune_eval_summary.parquet", index=False)

        if finetune_by_language_path.exists():
            finetune_by_language = con.execute(
                f"SELECT * FROM read_parquet('{finetune_by_language_path}') ORDER BY language"
            ).df()
            finetune_by_language = _ensure_grounded_trust_score(finetune_by_language)
            finetune_by_language.to_parquet(
                output_dir / "final_finetune_eval_by_language.parquet", index=False
            )

        history = finetune_summary.copy()
        baseline_rows = history[history["variant"] == "baseline-pipeline"]
        comparison_rows = history[history["variant"] != "baseline-pipeline"]

        if not baseline_rows.empty and not comparison_rows.empty:
            baseline = baseline_rows.iloc[0]
            latest_by_variant = (
                comparison_rows.sort_values("timestamp_utc", ascending=False)
                .groupby("variant")
                .head(1)
            )

            metric_cols = [
                "abstain_accuracy",
                "avg_citation_precision",
                "avg_citation_recall",
                "avg_answer_token_f1",
                "grounded_trust_score",
            ]
            metric_cols = [
                m for m in metric_cols if m in latest_by_variant.columns and m in baseline.index
            ]

            deltas = latest_by_variant.copy()
            for metric in metric_cols:
                deltas[f"delta_{metric}"] = deltas[metric] - float(baseline[metric])

            cols = [
                "variant",
                "timestamp_utc",
                "rows",
                "abstain_accuracy",
                "avg_citation_precision",
                "avg_citation_recall",
                "avg_answer_token_f1",
                "grounded_trust_score",
                "delta_abstain_accuracy",
                "delta_avg_citation_precision",
                "delta_avg_citation_recall",
                "delta_avg_answer_token_f1",
                "delta_grounded_trust_score",
            ]
            cols = [c for c in cols if c in deltas.columns]
            if "delta_grounded_trust_score" in cols:
                deltas = deltas[cols].sort_values("delta_grounded_trust_score", ascending=False)
            else:
                deltas = deltas[cols]
            deltas.to_parquet(output_dir / "final_finetune_eval_deltas.parquet", index=False)
    elif args.require_finetune:
        raise FileNotFoundError(
            f"Missing required finetune summary artifact: {finetune_summary_path}"
        )

    _write_reader_takeaways(output_dir=output_dir)

    print("Wrote final result artifacts to", output_dir)


if __name__ == "__main__":
    main()
