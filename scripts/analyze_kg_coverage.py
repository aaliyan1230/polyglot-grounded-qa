from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa.components.retriever import SeedKnowledgeGraphRetriever
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths
from polyglot_grounded_qa.utils.io import ensure_parent_dir, write_parquet


COVERAGE_QUERIES: dict[str, list[str]] = {
    "base": [
        "What is grounded QA?",
        "Why are citations required in grounded QA?",
        "How does locale inheritance work?",
        "When should the system abstain?",
    ],
    "es": [
        "Que es grounded QA?",
        "Por que se requieren citas en grounded QA?",
        "Como funciona la herencia de locale?",
        "Como ayuda la recuperacion hibrida?",
    ],
    "fr": [
        "Qu est ce que le grounded QA?",
        "Pourquoi les citations sont elles requises?",
        "Comment fonctionne l heritage des langues?",
        "Quand faut il s abstenir?",
    ],
    "tr": [
        "Grounded QA nedir?",
        "Dil paketleri mirasi nasil calisir?",
        "Sistem ne zaman cekimser kalmali?",
        "Graf destekli alma nasil yardimci olur?",
    ],
}


def _to_markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "| empty |\n| --- |"

    headers = df.columns
    rows = df.iter_rows()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _render_report(
    summary_df: pl.DataFrame, by_language_df: pl.DataFrame, report_path: Path
) -> None:
    ensure_parent_dir(report_path)
    lines = [
        "# KG Coverage Report",
        "",
        "## Coverage snapshot",
        "",
        _to_markdown_table(by_language_df),
        "",
        "## Failure buckets",
        "",
        _to_markdown_table(summary_df),
        "",
        "## Interpretation",
        "",
        "This report is a CPU-only coverage audit for the seed KG cache. No Kaggle GPU is required for this step.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)
    graph_retriever = SeedKnowledgeGraphRetriever(
        paths=get_seed_graph_paths(),
        min_path_score=cfg.pipeline.retrieval.graph_min_path_score,
        entity_link_min_score=cfg.pipeline.retrieval.entity_link_min_score,
    )

    rows: list[dict[str, object]] = []
    for language in cfg.languages:
        for query in COVERAGE_QUERIES.get(language, COVERAGE_QUERIES["base"]):
            diagnostics = graph_retriever.analyze_query(
                query=query,
                language=language,
                k=cfg.pipeline.retrieval.graph_top_k,
            )
            rows.append(
                {
                    "language": diagnostics.language,
                    "query": diagnostics.query,
                    "linked_entity_count": diagnostics.linked_entity_count,
                    "candidate_path_count": diagnostics.candidate_path_count,
                    "returned_path_count": diagnostics.returned_path_count,
                    "max_path_score": diagnostics.max_path_score,
                    "failure_bucket": diagnostics.failure_bucket,
                }
            )

    df = pl.DataFrame(rows)
    summary_df = (
        df.group_by("failure_bucket")
        .agg(
            pl.len().alias("queries"),
            pl.col("linked_entity_count").mean().round(4).alias("avg_linked_entity_count"),
            pl.col("returned_path_count").mean().round(4).alias("avg_returned_path_count"),
            pl.col("max_path_score").mean().round(4).alias("avg_max_path_score"),
        )
        .sort("failure_bucket")
    )
    by_language_df = (
        df.group_by("language")
        .agg(
            pl.len().alias("queries"),
            pl.col("returned_path_count").gt(0).mean().round(4).alias("path_yield_rate"),
            pl.col("linked_entity_count").mean().round(4).alias("avg_linked_entity_count"),
            pl.col("returned_path_count").mean().round(4).alias("avg_returned_path_count"),
            pl.col("max_path_score").mean().round(4).alias("avg_max_path_score"),
        )
        .sort("language")
    )

    summary_path = project_root / "artifacts" / "tables" / "kg_coverage_summary.parquet"
    by_language_path = project_root / "artifacts" / "tables" / "kg_coverage_by_language.parquet"
    report_path = project_root / "artifacts" / "tables" / "kg_coverage_report.md"
    write_parquet(summary_df, summary_path)
    write_parquet(by_language_df, by_language_path)
    _render_report(summary_df=summary_df, by_language_df=by_language_df, report_path=report_path)
    print(f"Wrote KG coverage outputs to {summary_path.parent}")


if __name__ == "__main__":
    main()