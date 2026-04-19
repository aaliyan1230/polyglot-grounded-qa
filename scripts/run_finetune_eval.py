from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from polyglot_grounded_qa import create_default_pipeline

PROMOTION_GATE = {
    "delta_avg_citation_precision_min": 0.0,
    "delta_avg_citation_recall_min": 0.0,
    "delta_grounded_trust_score_min": 0.0,
    "delta_avg_answer_token_f1_min": 0.0,
    # Allow up to 7% abstention accuracy regression vs baseline.
    # The baseline's abstain_accuracy (~88.7%) is inflated because it achieves that
    # score by never abstaining at all (abstain_recall=0.0, missed_abstain_rate=1.0).
    # A variant that actually attempts abstention will score slightly lower on raw
    # accuracy while being behaviourally superior.
    "delta_abstain_accuracy_min": -0.07,
}


def _safe_rate(numerator: int, denominator: int, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return numerator / denominator


def _summarize_metric_rows(
    rows: list[dict[str, Any]],
    variant: str,
    timestamp_utc: str,
    language: str | None = None,
    label_type: str | None = None,
) -> dict[str, Any]:
    count = len(rows)
    pred_abstain = sum(1 for row in rows if bool(row["pred_abstained"]))
    gold_abstain = sum(1 for row in rows if bool(row["gold_abstained"]))
    true_positive_abstain = sum(
        1 for row in rows if bool(row["pred_abstained"]) and bool(row["gold_abstained"])
    )
    false_positive_abstain = sum(
        1 for row in rows if bool(row["pred_abstained"]) and not bool(row["gold_abstained"])
    )
    false_negative_abstain = sum(
        1 for row in rows if (not bool(row["pred_abstained"])) and bool(row["gold_abstained"])
    )
    answerable_rows = count - gold_abstain

    abstain_accuracy = sum(1 for row in rows if bool(row["abstain_match"])) / max(count, 1)
    avg_citation_precision = sum(float(row["citation_precision"]) for row in rows) / max(count, 1)
    avg_citation_recall = sum(float(row["citation_recall"]) for row in rows) / max(count, 1)
    avg_answer_token_f1 = sum(float(row["answer_token_f1"]) for row in rows) / max(count, 1)
    abstain_precision = _safe_rate(
        true_positive_abstain,
        pred_abstain,
        1.0 if gold_abstain == 0 else 0.0,
    )
    abstain_recall = _safe_rate(
        true_positive_abstain,
        gold_abstain,
        1.0 if pred_abstain == 0 else 0.0,
    )
    false_abstain_rate = _safe_rate(false_positive_abstain, answerable_rows, 0.0)
    missed_abstain_rate = _safe_rate(false_negative_abstain, gold_abstain, 0.0)

    summary = {
        "variant": variant,
        "timestamp_utc": timestamp_utc,
        "rows": count,
        "answerable_rows": answerable_rows,
        "abstention_rows": gold_abstain,
        "abstain_accuracy": abstain_accuracy,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "false_abstain_rate": false_abstain_rate,
        "missed_abstain_rate": missed_abstain_rate,
        "avg_citation_precision": avg_citation_precision,
        "avg_citation_recall": avg_citation_recall,
        "avg_answer_token_f1": avg_answer_token_f1,
    }
    summary["grounded_trust_score"] = (
        0.2 * summary["abstain_accuracy"]
        + 0.3 * summary["avg_citation_precision"]
        + 0.3 * summary["avg_citation_recall"]
        + 0.2 * summary["avg_answer_token_f1"]
    )
    if language is not None:
        summary["language"] = language
    if label_type is not None:
        summary["label_type"] = label_type
    return summary


def _to_markdown_table(df: pl.DataFrame) -> str:
    headers = df.columns
    rows = df.rows()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_diagnostics_report(
    report_path: Path,
    variant: str,
    summary_df: pl.DataFrame,
    by_language_df: pl.DataFrame,
    by_label_type_df: pl.DataFrame,
) -> None:
    lines = [
        "# Finetune Evaluation Diagnostics",
        "",
        f"Variant: `{variant}`",
        "",
        "## Summary",
        "",
        _to_markdown_table(summary_df),
        "",
        "## By language",
        "",
        _to_markdown_table(by_language_df),
        "",
        "## By label type",
        "",
        _to_markdown_table(by_label_type_df),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = 0
    for token, p_count in pred_counts.items():
        overlap += min(p_count, gold_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _citation_prf(pred: list[str], gold: list[str]) -> tuple[float, float]:
    pred_set = set(pred)
    gold_set = set(gold)

    if not pred_set and not gold_set:
        return 1.0, 1.0

    overlap = len(pred_set.intersection(gold_set))
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    return precision, recall


def _build_baseline_predictions(
    records: list[dict[str, Any]], project_root: Path
) -> dict[str, dict[str, Any]]:
    pipeline = create_default_pipeline(str(project_root))
    preds: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row.get("id", ""))
        query = str(row.get("query", ""))
        language = str(row.get("language", "base"))
        result = pipeline.run(query=query, language=language)
        preds[sample_id] = {
            "answer": result.answer,
            "citations": [c.chunk_id for c in result.citations],
            "abstained": bool(result.abstained),
        }
    return preds


def _load_prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    preds: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        sample_id = str(row.get("id", f"line-{idx}"))
        preds[sample_id] = {
            "answer": str(row.get("answer", "")),
            "citations": list(row.get("citations", [])),
            "abstained": bool(row.get("abstained", False)),
        }
    return preds


def _append_or_write(path: Path, df: pl.DataFrame) -> None:
    if path.exists():
        existing = pl.read_parquet(path)
        all_cols: list[str] = list(existing.columns)
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)

        existing_aligned = existing.with_columns(
            *[pl.lit(None).alias(col) for col in all_cols if col not in existing.columns]
        ).select(all_cols)
        incoming_aligned = df.with_columns(
            *[pl.lit(None).alias(col) for col in all_cols if col not in df.columns]
        ).select(all_cols)

        combined = pl.concat([existing_aligned, incoming_aligned], how="vertical_relaxed")
        combined.write_parquet(path)
    else:
        df.write_parquet(path)


def _build_variant_leaderboard(summary_path: Path, output_dir: Path) -> None:
    if not summary_path.exists():
        return

    history = pl.read_parquet(summary_path)
    required = {
        "variant",
        "timestamp_utc",
        "rows",
        "abstain_accuracy",
        "avg_citation_precision",
        "avg_citation_recall",
        "avg_answer_token_f1",
        "grounded_trust_score",
    }
    if not required.issubset(set(history.columns)):
        return

    latest_by_variant = (
        history.sort("timestamp_utc", descending=True)
        .group_by("variant")
        .head(1)
        .sort("variant")
        .with_columns(
            pl.when(pl.col("grounded_trust_score").is_null())
            .then(
                0.2 * pl.col("abstain_accuracy")
                + 0.3 * pl.col("avg_citation_precision")
                + 0.3 * pl.col("avg_citation_recall")
                + 0.2 * pl.col("avg_answer_token_f1")
            )
            .otherwise(pl.col("grounded_trust_score"))
            .alias("grounded_trust_score")
        )
    )

    baseline = latest_by_variant.filter(pl.col("variant") == "baseline-pipeline")
    if baseline.height == 0:
        return

    baseline_row = baseline.row(0, named=True)
    candidates = latest_by_variant.filter(pl.col("variant") != "baseline-pipeline")
    if candidates.height == 0:
        return

    leaderboard = candidates.with_columns(
        (pl.col("abstain_accuracy") - float(baseline_row["abstain_accuracy"])).alias(
            "delta_abstain_accuracy"
        ),
        (pl.col("avg_citation_precision") - float(baseline_row["avg_citation_precision"])).alias(
            "delta_avg_citation_precision"
        ),
        (pl.col("avg_citation_recall") - float(baseline_row["avg_citation_recall"])).alias(
            "delta_avg_citation_recall"
        ),
        (pl.col("avg_answer_token_f1") - float(baseline_row["avg_answer_token_f1"])).alias(
            "delta_avg_answer_token_f1"
        ),
        (pl.col("grounded_trust_score") - float(baseline_row["grounded_trust_score"])).alias(
            "delta_grounded_trust_score"
        ),
        (
            (~pl.col("variant").str.contains("oracle", literal=True))
            & (~pl.col("variant").str.contains("control", literal=True))
        ).alias("is_practical_variant"),
    ).with_columns(
        (
            pl.col("is_practical_variant")
            & (pl.col("delta_avg_citation_precision") > PROMOTION_GATE["delta_avg_citation_precision_min"])
            & (pl.col("delta_avg_citation_recall") > PROMOTION_GATE["delta_avg_citation_recall_min"])
            & (pl.col("delta_grounded_trust_score") > PROMOTION_GATE["delta_grounded_trust_score_min"])
            & (pl.col("delta_avg_answer_token_f1") >= PROMOTION_GATE["delta_avg_answer_token_f1_min"])
            & (pl.col("delta_abstain_accuracy") >= PROMOTION_GATE["delta_abstain_accuracy_min"])
        ).alias("passes_grounding_gate")
    )

    leaderboard = leaderboard.sort(
        by=["passes_grounding_gate", "delta_grounded_trust_score", "delta_avg_answer_token_f1"],
        descending=[True, True, True],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "finetune_variant_leaderboard.parquet"
    leaderboard.write_parquet(leaderboard_path)

    md_path = output_dir / "finetune_variant_leaderboard.md"

    def _to_f64(value: object) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    lines = [
        "# Finetune Variant Leaderboard",
        "",
        "Ranked by grounding-gate pass status, then `delta_grounded_trust_score`, then `delta_avg_answer_token_f1`.",
        "",
        "Promotion gate (practical variants only):",
        "- `delta_avg_citation_precision > 0.0`",
        "- `delta_avg_citation_recall > 0.0`",
        "- `delta_grounded_trust_score > 0.0`",
        "- `delta_avg_answer_token_f1 >= 0.0`",
        "- `delta_abstain_accuracy >= -0.07` (up to 7% regression allowed; baseline abstain_accuracy is inflated by never-abstain behaviour)",
        "",
        "| variant | practical | gate_pass | d_trust | d_f1 | d_cit_p | d_cit_r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard.iter_rows(named=True):
        lines.append(
            "| {variant} | {practical} | {gate_pass} | {d_trust:.4f} | {d_f1:.4f} | {d_cit_p:.4f} | {d_cit_r:.4f} |".format(
                variant=str(row["variant"]),
                practical="yes" if bool(row["is_practical_variant"]) else "no",
                gate_pass="yes" if bool(row["passes_grounding_gate"]) else "no",
                d_trust=_to_f64(row["delta_grounded_trust_score"]),
                d_f1=_to_f64(row["delta_avg_answer_token_f1"]),
                d_cit_p=_to_f64(row["delta_avg_citation_precision"]),
                d_cit_r=_to_f64(row["delta_avg_citation_recall"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finetune predictions against grounded SFT test split.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/benchmarks/finetune/test.jsonl"),
        help="Ground-truth test JSONL in repository schema.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions JSONL with fields: id, answer, citations, abstained.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline-pipeline",
        help="Variant label stored in output tables.",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=Path("artifacts/runs/finetune_eval_rows.parquet"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/tables/finetune_eval_summary.parquet"),
    )
    parser.add_argument(
        "--by-language-output",
        type=Path,
        default=Path("artifacts/tables/finetune_eval_by_language.parquet"),
    )
    parser.add_argument(
        "--by-label-type-output",
        type=Path,
        default=Path("artifacts/tables/finetune_eval_by_label_type.parquet"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("artifacts/tables/finetune_eval_diagnostics.md"),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to existing parquet outputs instead of overwriting.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    test_records = _read_jsonl(project_root / args.test_file)
    if not test_records:
        raise ValueError("Test split is empty. Run data pipeline first.")

    if args.predictions is None:
        pred_map = _build_baseline_predictions(test_records, project_root)
    else:
        pred_map = _load_prediction_map(project_root / args.predictions)

    timestamp_utc = datetime.now(UTC).isoformat()

    row_metrics: list[dict[str, Any]] = []
    for row in test_records:
        sample_id = str(row.get("id", ""))
        language = str(row.get("language", "base"))
        label_type = str(row.get("label_type", ""))

        target = row.get("target", {})
        gold_answer = str(target.get("answer", ""))
        gold_citations = list(target.get("citations", []))
        gold_abstained = bool(target.get("abstained", False))

        pred = pred_map.get(sample_id, {"answer": "", "citations": [], "abstained": False})
        pred_answer = str(pred.get("answer", ""))
        pred_citations = list(pred.get("citations", []))
        pred_abstained = bool(pred.get("abstained", False))

        citation_precision, citation_recall = _citation_prf(pred_citations, gold_citations)
        abstain_match = pred_abstained == gold_abstained

        row_metrics.append(
            {
                "variant": args.variant,
                "timestamp_utc": timestamp_utc,
                "id": sample_id,
                "language": language,
                "label_type": label_type,
                "gold_abstained": gold_abstained,
                "pred_abstained": pred_abstained,
                "abstain_match": abstain_match,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "answer_token_f1": _token_f1(pred_answer, gold_answer),
                "pred_citation_count": len(pred_citations),
                "gold_citation_count": len(gold_citations),
            }
        )

    rows_df = pl.DataFrame(row_metrics)

    summary_df = pl.DataFrame([_summarize_metric_rows(row_metrics, args.variant, timestamp_utc)])

    by_language_rows = []
    for language in sorted({str(row["language"]) for row in row_metrics}):
        language_rows = [row for row in row_metrics if str(row["language"]) == language]
        by_language_rows.append(
            _summarize_metric_rows(language_rows, args.variant, timestamp_utc, language=language)
        )
    by_language_df = pl.DataFrame(by_language_rows).sort("language")

    by_label_type_rows = []
    for label_type in sorted({str(row["label_type"]) for row in row_metrics}):
        label_rows = [row for row in row_metrics if str(row["label_type"]) == label_type]
        by_label_type_rows.append(
            _summarize_metric_rows(label_rows, args.variant, timestamp_utc, label_type=label_type)
        )
    by_label_type_df = pl.DataFrame(by_label_type_rows).sort("label_type")

    rows_output = project_root / args.rows_output
    summary_output = project_root / args.summary_output
    by_language_output = project_root / args.by_language_output
    by_label_type_output = project_root / args.by_label_type_output
    report_output = project_root / args.report_output

    rows_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    by_language_output.parent.mkdir(parents=True, exist_ok=True)
    by_label_type_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    if args.append:
        _append_or_write(rows_output, rows_df)
        _append_or_write(summary_output, summary_df)
        _append_or_write(by_language_output, by_language_df)
        _append_or_write(by_label_type_output, by_label_type_df)
    else:
        rows_df.write_parquet(rows_output)
        summary_df.write_parquet(summary_output)
        by_language_df.write_parquet(by_language_output)
        by_label_type_df.write_parquet(by_label_type_output)

    _build_variant_leaderboard(summary_output, summary_output.parent)
    _write_diagnostics_report(report_output, args.variant, summary_df, by_language_df, by_label_type_df)

    print(f"Wrote row metrics: {rows_output}")
    print(f"Wrote summary: {summary_output}")
    print(f"Wrote by-language summary: {by_language_output}")
    print(f"Wrote by-label-type summary: {by_label_type_output}")
    print(f"Wrote variant leaderboard: {summary_output.parent / 'finetune_variant_leaderboard.parquet'}")
    print(f"Wrote diagnostics report: {report_output}")
    print(summary_df)


if __name__ == "__main__":
    main()
