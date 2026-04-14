from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa.components.retriever import SeedKnowledgeGraphRetriever
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths
from polyglot_grounded_qa.utils.io import ensure_parent_dir, write_parquet


AUDIT_QUERIES: dict[str, list[str]] = {
    "base": [
        "What is grounded QA?",
        "Why are citations required in grounded QA?",
        "How does locale inheritance work?",
        "When should the system abstain?",
        "How does hybrid retrieval improve grounded QA?",
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


def _tokenize(text: str) -> set[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return set(tokens)


def _classify_path(query: str, path_text: str, path_length: int, link_score: float) -> tuple[str, str, float]:
    query_terms = _tokenize(query)
    path_terms = _tokenize(path_text)
    if not query_terms or not path_terms:
        return "underspecified", "low", 0.0

    overlap = len(query_terms.intersection(path_terms))
    lexical_overlap = overlap / max(len(query_terms), 1)

    if lexical_overlap >= 0.8 and path_length <= 2:
        return "explicit-restatement", "high", round(lexical_overlap, 4)
    if lexical_overlap >= 0.35 or link_score >= 0.65:
        return "supporting-path", "low", round(lexical_overlap, 4)
    return "underspecified", "medium", round(lexical_overlap, 4)


def _to_markdown_table(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "| empty |\n| --- |"

    headers = df.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(
    summary_df: pl.DataFrame,
    by_language_df: pl.DataFrame,
    sample_df: pl.DataFrame,
    report_path: Path,
) -> None:
    ensure_parent_dir(report_path)
    sample_view = sample_df.sort(["leakage_risk", "graph_path_score"], descending=[True, True]).head(8)
    lines = [
        "# KG Path Quality Audit",
        "",
        "## Summary",
        "",
        _to_markdown_table(summary_df),
        "",
        "## By language",
        "",
        _to_markdown_table(by_language_df),
        "",
        "## Sample audited paths",
        "",
        _to_markdown_table(sample_view),
        "",
        "## Interpretation",
        "",
        "This is a heuristic leakage and support-quality audit over retrieved graph paths. It is CPU-only and intended to catch trivial answer-restating support before stronger claims are made.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)
    retriever = SeedKnowledgeGraphRetriever(
        paths=get_seed_graph_paths(),
        min_path_score=cfg.pipeline.retrieval.graph_min_path_score,
        entity_link_min_score=cfg.pipeline.retrieval.entity_link_min_score,
    )

    rows: list[dict[str, object]] = []
    for language in cfg.languages:
        queries = AUDIT_QUERIES.get(language, AUDIT_QUERIES["base"])
        for query in queries:
            chunks = retriever.retrieve(query=query, language=language, k=cfg.pipeline.retrieval.graph_top_k)
            if not chunks:
                rows.append(
                    {
                        "language": language,
                        "query": query,
                        "path_id": None,
                        "path_length": 0,
                        "graph_path_score": 0.0,
                        "graph_link_score": 0.0,
                        "lexical_overlap": 0.0,
                        "quality_label": "no-supported-path",
                        "leakage_risk": "none",
                    }
                )
                continue

            for chunk in chunks:
                quality_label, leakage_risk, lexical_overlap = _classify_path(
                    query=query,
                    path_text=chunk.text,
                    path_length=int(chunk.metadata.get("path_length", 0)),
                    link_score=float(chunk.metadata.get("graph_link_score", 0.0)),
                )
                rows.append(
                    {
                        "language": language,
                        "query": query,
                        "path_id": chunk.chunk_id,
                        "path_length": int(chunk.metadata.get("path_length", 0)),
                        "graph_path_score": float(chunk.metadata.get("graph_path_score", chunk.score)),
                        "graph_link_score": float(chunk.metadata.get("graph_link_score", 0.0)),
                        "lexical_overlap": lexical_overlap,
                        "quality_label": quality_label,
                        "leakage_risk": leakage_risk,
                    }
                )

    df = pl.DataFrame(rows)
    summary_df = (
        df.group_by(["quality_label", "leakage_risk"])
        .agg(
            pl.len().alias("rows"),
            pl.col("graph_path_score").mean().round(4).alias("avg_graph_path_score"),
            pl.col("graph_link_score").mean().round(4).alias("avg_graph_link_score"),
            pl.col("lexical_overlap").mean().round(4).alias("avg_lexical_overlap"),
        )
        .sort(["leakage_risk", "quality_label"])
    )
    by_language_df = (
        df.group_by("language")
        .agg(
            pl.len().alias("rows"),
            pl.col("quality_label").eq("supporting-path").mean().round(4).alias("supporting_path_rate"),
            pl.col("quality_label").eq("explicit-restatement").mean().round(4).alias("explicit_restatement_rate"),
            pl.col("leakage_risk").eq("high").mean().round(4).alias("high_leakage_rate"),
            pl.col("graph_path_score").mean().round(4).alias("avg_graph_path_score"),
        )
        .sort("language")
    )

    summary_path = project_root / "artifacts" / "tables" / "kg_path_quality_summary.parquet"
    by_language_path = project_root / "artifacts" / "tables" / "kg_path_quality_by_language.parquet"
    rows_path = project_root / "artifacts" / "tables" / "kg_path_quality_rows.parquet"
    report_path = project_root / "artifacts" / "tables" / "kg_path_quality_report.md"

    write_parquet(summary_df, summary_path)
    write_parquet(by_language_df, by_language_path)
    write_parquet(df, rows_path)
    _write_report(summary_df=summary_df, by_language_df=by_language_df, sample_df=df, report_path=report_path)
    print(f"Wrote KG path quality outputs to {summary_path.parent}")


if __name__ == "__main__":
    main()