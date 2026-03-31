from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from polyglot_grounded_qa import create_default_pipeline


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
        combined = pl.concat([existing, df], how="vertical_relaxed")
        combined.write_parquet(path)
    else:
        df.write_parquet(path)


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

    summary_df = rows_df.select(
        pl.lit(args.variant).alias("variant"),
        pl.lit(timestamp_utc).alias("timestamp_utc"),
        pl.len().alias("rows"),
        pl.col("abstain_match").mean().alias("abstain_accuracy"),
        pl.col("citation_precision").mean().alias("avg_citation_precision"),
        pl.col("citation_recall").mean().alias("avg_citation_recall"),
        pl.col("answer_token_f1").mean().alias("avg_answer_token_f1"),
    )

    by_language_df = (
        rows_df.group_by("language")
        .agg(
            pl.len().alias("rows"),
            pl.col("abstain_match").mean().alias("abstain_accuracy"),
            pl.col("citation_precision").mean().alias("avg_citation_precision"),
            pl.col("citation_recall").mean().alias("avg_citation_recall"),
            pl.col("answer_token_f1").mean().alias("avg_answer_token_f1"),
        )
        .with_columns(pl.lit(args.variant).alias("variant"), pl.lit(timestamp_utc).alias("timestamp_utc"))
        .select(
            "variant",
            "timestamp_utc",
            "language",
            "rows",
            "abstain_accuracy",
            "avg_citation_precision",
            "avg_citation_recall",
            "avg_answer_token_f1",
        )
        .sort("language")
    )

    rows_output = project_root / args.rows_output
    summary_output = project_root / args.summary_output
    by_language_output = project_root / args.by_language_output

    rows_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    by_language_output.parent.mkdir(parents=True, exist_ok=True)

    if args.append:
        _append_or_write(rows_output, rows_df)
        _append_or_write(summary_output, summary_df)
        _append_or_write(by_language_output, by_language_df)
    else:
        rows_df.write_parquet(rows_output)
        summary_df.write_parquet(summary_output)
        by_language_df.write_parquet(by_language_output)

    print(f"Wrote row metrics: {rows_output}")
    print(f"Wrote summary: {summary_output}")
    print(f"Wrote by-language summary: {by_language_output}")
    print(summary_df)


if __name__ == "__main__":
    main()
