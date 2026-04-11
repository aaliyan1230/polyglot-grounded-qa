from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


def _to_markdown_table(df: pl.DataFrame) -> str:
    headers = df.columns
    rows = df.rows()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _flatten_rows(rows: list[dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        retrieved = row.get("retrieved_chunks", [])
        retrieved_list = retrieved if isinstance(retrieved, list) else []
        retrieved_ids = {
            str(chunk.get("chunk_id", ""))
            for chunk in retrieved_list
            if isinstance(chunk, dict) and chunk.get("chunk_id") is not None
        }
        target = row.get("target", {})
        citations = target.get("citations", [])
        citations_list = citations if isinstance(citations, list) else []
        abstained = bool(target.get("abstained", False))
        label_type = str(row.get("label_type", ""))
        expected_abstained = label_type == "insufficient_evidence"
        citation_valid = all(str(citation) in retrieved_ids for citation in citations_list)
        flat_rows.append(
            {
                "id": row.get("id", ""),
                "split": split or "merged",
                "language": row.get("language", ""),
                "label_type": label_type,
                "source": row.get("source", ""),
                "abstained": abstained,
                "retrieved_count": len(retrieved_list),
                "citation_count": len(citations_list),
                "query_length": len(str(row.get("query", ""))),
                "answer_length": len(str(target.get("answer", ""))),
                "citation_valid": citation_valid,
                "label_matches_target": abstained == expected_abstained,
                "answerable_without_citations": (not abstained) and len(citations_list) == 0,
                "abstention_with_citations": abstained and len(citations_list) > 0,
                "hard_negative": abstained and len(retrieved_list) > 0,
                "evidence_free_abstention": abstained and len(retrieved_list) == 0,
            }
        )
    return flat_rows


def _aggregate(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    return (
        df.group_by(group_cols)
        .agg(
            pl.len().alias("rows"),
            pl.col("abstained").cast(pl.Float64).mean().alias("abstain_rate"),
            pl.col("citation_valid").cast(pl.Float64).mean().alias("citation_validity_rate"),
            pl.col("label_matches_target").cast(pl.Float64).mean().alias("label_consistency_rate"),
            pl.col("hard_negative").cast(pl.Float64).mean().alias("hard_negative_rate"),
            pl.col("answerable_without_citations")
            .cast(pl.Float64)
            .mean()
            .alias("answerable_without_citations_rate"),
            pl.col("abstention_with_citations")
            .cast(pl.Float64)
            .mean()
            .alias("abstention_with_citations_rate"),
            pl.col("retrieved_count").mean().alias("avg_retrieved_count"),
            pl.col("citation_count").mean().alias("avg_citation_count"),
            pl.col("query_length").mean().alias("avg_query_length"),
            pl.col("answer_length").mean().alias("avg_answer_length"),
            (pl.len() - pl.col("abstained").sum()).alias("answerable_rows"),
            pl.col("abstained").sum().alias("abstention_rows"),
        )
        .sort(group_cols)
    )


def _compute_split_overlap(split_rows: dict[str, list[dict[str, Any]]]) -> pl.DataFrame:
    split_keys: dict[str, set[tuple[str, str, str]]] = {}
    for split_name, rows in split_rows.items():
        split_keys[split_name] = {
            (
                str(row.get("language", "")),
                str(row.get("query", "")),
                str(row.get("label_type", "")),
            )
            for row in rows
        }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    records: list[dict[str, Any]] = []
    for left, right in pairs:
        left_keys = split_keys.get(left, set())
        right_keys = split_keys.get(right, set())
        overlap = len(left_keys.intersection(right_keys))
        records.append({"split_pair": f"{left}-{right}", "overlap_keys": overlap})
    return pl.DataFrame(records)


def _build_warnings(
    overall: dict[str, Any],
    by_language: pl.DataFrame,
    split_overlap: pl.DataFrame,
    min_abstain_ratio: float,
    min_language_rows: int,
) -> list[str]:
    warnings: list[str] = []
    if float(overall.get("abstain_rate", 0.0)) < min_abstain_ratio:
        warnings.append(
            f"Overall abstain rate {float(overall.get('abstain_rate', 0.0)):.3f} is below target {min_abstain_ratio:.3f}."
        )
    if float(overall.get("hard_negative_rate", 0.0)) == 0.0:
        warnings.append("No hard-negative abstention examples were found with retrieved evidence present.")
    if float(overall.get("citation_validity_rate", 0.0)) < 1.0:
        warnings.append("Some target citation IDs do not match retrieved chunk IDs.")
    if float(overall.get("label_consistency_rate", 0.0)) < 1.0:
        warnings.append("Some rows have label_type values that disagree with target.abstained.")

    for row in by_language.iter_rows(named=True):
        language = str(row["language"])
        rows = int(row["rows"])
        abstain_rate = float(row["abstain_rate"])
        if rows < min_language_rows:
            warnings.append(
                f"Language {language} has only {rows} rows; target minimum is {min_language_rows}."
            )
        if abstain_rate < min_abstain_ratio:
            warnings.append(
                f"Language {language} abstain rate {abstain_rate:.3f} is below target {min_abstain_ratio:.3f}."
            )

    for row in split_overlap.iter_rows(named=True):
        if int(row["overlap_keys"]) > 0:
            warnings.append(
                f"Split overlap detected for {row['split_pair']}: {int(row['overlap_keys'])} shared keys."
            )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze grounded QA SFT dataset quality.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks/finetune/sft_dataset_merged.jsonl"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/tables/sft_dataset_quality"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/benchmarks/finetune"),
        help="Directory containing train/val/test splits for consistency checks.",
    )
    parser.add_argument(
        "--min-abstain-ratio",
        type=float,
        default=0.2,
        help="Target minimum abstention ratio used for warnings.",
    )
    parser.add_argument(
        "--min-language-rows",
        type=int,
        default=40,
        help="Target minimum rows per language used for warnings.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rows = _read_jsonl(project_root / args.input)
    if not rows:
        raise ValueError("Input dataset is empty")

    merged_df = pl.DataFrame(_flatten_rows(rows))

    overall = _aggregate(merged_df.with_columns(pl.lit("all").alias("scope")), ["scope"]).drop("scope")
    by_language = _aggregate(merged_df, ["language"])
    by_source = _aggregate(merged_df, ["source"]).sort("rows", descending=True)
    by_label_type = _aggregate(merged_df, ["label_type"])

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_tables: list[pl.DataFrame] = []
    for split_name in ["train", "val", "test"]:
        split_path = project_root / args.split_dir / f"{split_name}.jsonl"
        if split_path.exists():
            loaded = _read_jsonl(split_path)
            split_rows[split_name] = loaded
            split_tables.append(pl.DataFrame(_flatten_rows(loaded, split=split_name)))

    if split_tables:
        split_df = pl.concat(split_tables, how="vertical_relaxed")
        by_split = _aggregate(split_df, ["split"])
        by_split_language = _aggregate(split_df, ["split", "language"])
        split_overlap = _compute_split_overlap(split_rows)
    else:
        by_split = pl.DataFrame([])
        by_split_language = pl.DataFrame([])
        split_overlap = pl.DataFrame([])

    out_prefix = project_root / args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    overall.write_parquet(out_prefix.with_name(out_prefix.name + "_overall.parquet"))
    by_language.write_parquet(out_prefix.with_name(out_prefix.name + "_by_language.parquet"))
    by_source.write_parquet(out_prefix.with_name(out_prefix.name + "_by_source.parquet"))
    by_label_type.write_parquet(out_prefix.with_name(out_prefix.name + "_by_label_type.parquet"))
    if by_split.height > 0:
        by_split.write_parquet(out_prefix.with_name(out_prefix.name + "_by_split.parquet"))
        by_split_language.write_parquet(
            out_prefix.with_name(out_prefix.name + "_by_split_language.parquet")
        )
        split_overlap.write_parquet(out_prefix.with_name(out_prefix.name + "_split_overlap.parquet"))

    overall_row = overall.row(0, named=True)
    warnings = _build_warnings(
        overall=overall_row,
        by_language=by_language,
        split_overlap=split_overlap,
        min_abstain_ratio=args.min_abstain_ratio,
        min_language_rows=args.min_language_rows,
    )

    report_path = out_prefix.with_name(out_prefix.name + "_report.md")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# SFT Dataset Quality Report\n\n")
        f.write("## Contract checks\n\n")
        if warnings:
            for warning in warnings:
                f.write(f"- {warning}\n")
        else:
            f.write("- No contract warnings detected.\n")
        f.write("\n## Overall\n\n")
        f.write(_to_markdown_table(overall))
        f.write("\n\n## By language\n\n")
        f.write(_to_markdown_table(by_language))
        f.write("\n\n## By source\n\n")
        f.write(_to_markdown_table(by_source))
        f.write("\n\n## By label type\n\n")
        f.write(_to_markdown_table(by_label_type))
        if by_split.height > 0:
            f.write("\n\n## By split\n\n")
            f.write(_to_markdown_table(by_split))
            f.write("\n\n## Split by language\n\n")
            f.write(_to_markdown_table(by_split_language))
            f.write("\n\n## Split overlap\n\n")
            f.write(_to_markdown_table(split_overlap))
        f.write("\n")

    print(f"Rows analyzed: {len(rows)}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
