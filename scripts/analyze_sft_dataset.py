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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rows = _read_jsonl(project_root / args.input)
    if not rows:
        raise ValueError("Input dataset is empty")

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        retrieved = row.get("retrieved_chunks", [])
        target = row.get("target", {})
        citations = target.get("citations", [])
        flat_rows.append(
            {
                "id": row.get("id", ""),
                "language": row.get("language", ""),
                "label_type": row.get("label_type", ""),
                "source": row.get("source", ""),
                "abstained": bool(target.get("abstained", False)),
                "retrieved_count": len(retrieved) if isinstance(retrieved, list) else 0,
                "citation_count": len(citations) if isinstance(citations, list) else 0,
                "query_length": len(str(row.get("query", ""))),
                "answer_length": len(str(target.get("answer", ""))),
            }
        )

    df = pl.DataFrame(flat_rows)

    overall = df.select(
        pl.len().alias("rows"),
        pl.col("abstained").mean().alias("abstain_rate"),
        pl.col("citation_count").mean().alias("avg_citation_count"),
        pl.col("retrieved_count").mean().alias("avg_retrieved_count"),
        pl.col("query_length").mean().alias("avg_query_length"),
        pl.col("answer_length").mean().alias("avg_answer_length"),
    )

    by_language = (
        df.group_by("language")
        .agg(
            pl.len().alias("rows"),
            pl.col("abstained").mean().alias("abstain_rate"),
            pl.col("citation_count").mean().alias("avg_citation_count"),
            pl.col("retrieved_count").mean().alias("avg_retrieved_count"),
        )
        .sort("language")
    )

    by_source = (
        df.group_by("source")
        .agg(
            pl.len().alias("rows"),
            pl.col("abstained").mean().alias("abstain_rate"),
            pl.col("citation_count").mean().alias("avg_citation_count"),
        )
        .sort("rows", descending=True)
    )

    out_prefix = project_root / args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    overall.write_parquet(out_prefix.with_name(out_prefix.name + "_overall.parquet"))
    by_language.write_parquet(out_prefix.with_name(out_prefix.name + "_by_language.parquet"))
    by_source.write_parquet(out_prefix.with_name(out_prefix.name + "_by_source.parquet"))

    report_path = out_prefix.with_name(out_prefix.name + "_report.md")
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# SFT Dataset Quality Report\n\n")
        f.write("## Overall\n\n")
        f.write(_to_markdown_table(overall))
        f.write("\n\n## By language\n\n")
        f.write(_to_markdown_table(by_language))
        f.write("\n\n## By source\n\n")
        f.write(_to_markdown_table(by_source))
        f.write("\n")

    print(f"Rows analyzed: {len(df)}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
