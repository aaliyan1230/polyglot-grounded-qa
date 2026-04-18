from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from polyglot_grounded_qa.components.abstain import GraphAwareAbstentionPolicy, ThresholdAbstentionPolicy
from polyglot_grounded_qa.components.retriever import SeedKnowledgeGraphRetriever
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.core.kg_cache import load_graph_paths
from polyglot_grounded_qa.schemas.contracts import Citation, Claim, GroundedAnswer, RetrievedChunk
from polyglot_grounded_qa.utils.io import ensure_parent_dir, write_parquet


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _estimate_text_support(query: str, chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    overlaps: list[float] = []
    for chunk in chunks:
        chunk_tokens = set(_tokenize(chunk.text))
        if not chunk_tokens:
            overlaps.append(0.0)
            continue
        overlaps.append(len(query_tokens.intersection(chunk_tokens)) / len(query_tokens))

    lexical_overlap = max(overlaps, default=0.0)
    density_bonus = min(len(chunks), 3) * 0.1
    return round(min(1.0, 0.25 + density_bonus + (0.65 * lexical_overlap)), 4)


def _graph_quality_score(chunk: RetrievedChunk) -> float:
    link_score = float(chunk.metadata.get("graph_link_score", 0.0))
    path_score = min(float(chunk.metadata.get("graph_path_score", chunk.score)) / 1.5, 1.0)
    path_length = int(chunk.metadata.get("path_length", 1))
    source = str(chunk.metadata.get("source", "seed"))
    source_score = 1.0 if source == "wikidata" else 0.85
    length_score = max(0.5, 1.0 - (0.15 * max(path_length - 1, 0)))
    return round(
        (0.45 * link_score) + (0.3 * path_score) + (0.15 * source_score) + (0.1 * length_score),
        4,
    )


def _to_retrieved_chunks(raw_chunks: list[dict[str, Any]]) -> list[RetrievedChunk]:
    rows: list[RetrievedChunk] = []
    for index, raw_chunk in enumerate(raw_chunks):
        rows.append(
            RetrievedChunk(
                doc_id=str(raw_chunk.get("doc_id", f"doc-{index}")),
                chunk_id=str(raw_chunk.get("chunk_id", f"chunk-{index}")),
                text=str(raw_chunk.get("text", "")),
                score=0.1,
                metadata={"evidence_type": "text"},
            )
        )
    return rows


def _build_answer(
    query: str,
    retrieval_mode: str,
    text_chunks: list[RetrievedChunk],
    graph_chunks: list[RetrievedChunk],
    text_support_score: float,
    graph_support_score: float,
    graph_quality_score: float,
) -> GroundedAnswer:
    citations: list[Citation] = []
    for chunk in text_chunks[:1]:
        citations.append(Citation(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id))
    for chunk in graph_chunks[:1]:
        citations.append(Citation(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id))

    if retrieval_mode == "text":
        confidence = text_support_score
    elif retrieval_mode == "graph":
        confidence = graph_support_score
    else:
        confidence = max(text_support_score, graph_support_score)

    claims = [
        Claim(
            text=f"Abstention support estimate for: {query}",
            supported=confidence > 0.0,
            confidence=confidence,
        )
    ]
    return GroundedAnswer(
        answer="I do not have enough evidence." if not citations else f"Evidence collected for: {query}",
        citations=citations,
        claims=claims,
        metadata={
            "retrieval_mode": retrieval_mode,
            "text_evidence_count": len(text_chunks),
            "graph_evidence_count": len(graph_chunks),
            "text_support_score": text_support_score,
            "graph_support_score": graph_support_score,
            "graph_quality_score": graph_quality_score,
            "top_evidence_type": "graph" if graph_chunks else ("text" if text_chunks else "none"),
        },
    )


def _score_variant_rows(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    total = len(rows)
    pred_abstain = sum(1 for row in rows if bool(row["pred_abstained"]))
    gold_abstain = sum(1 for row in rows if bool(row["gold_abstained"]))
    true_positive = sum(
        1 for row in rows if bool(row["pred_abstained"]) and bool(row["gold_abstained"])
    )
    false_positive = sum(
        1 for row in rows if bool(row["pred_abstained"]) and not bool(row["gold_abstained"])
    )
    false_negative = sum(
        1 for row in rows if (not bool(row["pred_abstained"])) and bool(row["gold_abstained"])
    )
    answerable_rows = total - gold_abstain

    abstain_accuracy = sum(1 for row in rows if bool(row["pred_abstained"]) == bool(row["gold_abstained"])) / max(total, 1)
    abstain_precision = true_positive / pred_abstain if pred_abstain else (1.0 if gold_abstain == 0 else 0.0)
    abstain_recall = true_positive / gold_abstain if gold_abstain else (1.0 if pred_abstain == 0 else 0.0)
    false_abstain_rate = false_positive / max(answerable_rows, 1)
    missed_abstain_rate = false_negative / max(gold_abstain, 1) if gold_abstain else 0.0
    return {
        "variant": variant,
        "rows": total,
        "abstain_accuracy": abstain_accuracy,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "false_abstain_rate": false_abstain_rate,
        "missed_abstain_rate": missed_abstain_rate,
        "avg_text_support_score": sum(float(row["text_support_score"]) for row in rows) / max(total, 1),
        "avg_graph_support_score": sum(float(row["graph_support_score"]) for row in rows) / max(total, 1),
        "avg_graph_quality_score": sum(float(row["graph_quality_score"]) for row in rows) / max(total, 1),
        "graph_supported_rate": sum(1 for row in rows if float(row["graph_support_score"]) > 0.0) / max(total, 1),
    }


def _to_markdown_table(df: pl.DataFrame) -> str:
    headers = df.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_report(summary_df: pl.DataFrame, by_language_df: pl.DataFrame, report_path: Path) -> None:
    ensure_parent_dir(report_path)
    lines = [
        "# Hybrid Abstention Summary",
        "",
        "## Overall",
        "",
        _to_markdown_table(summary_df),
        "",
        "## By language",
        "",
        _to_markdown_table(by_language_df),
        "",
        "## Interpretation",
        "",
        "This benchmark-backed abstention summary compares text-only, hybrid-threshold, and hybrid-graph-aware policies against existing abstention labels. It is CPU-only and aligned with the repo's trust-first evaluation direction.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate graph-aware abstention policies against benchmark abstention labels.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/benchmarks/finetune/test.jsonl"),
        help="Benchmark file with abstention labels.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cfg = load_app_config(project_root=project_root)
    records = _read_jsonl(project_root / args.test_file)
    graph_retriever = SeedKnowledgeGraphRetriever(
        paths=load_graph_paths(project_root),
        min_path_score=cfg.pipeline.retrieval.graph_min_path_score,
        entity_link_min_score=cfg.pipeline.retrieval.entity_link_min_score,
    )

    threshold_policy = ThresholdAbstentionPolicy(cfg.languages[cfg.pipeline.default_language].thresholds)
    graph_aware_policy = GraphAwareAbstentionPolicy(cfg.languages[cfg.pipeline.default_language].thresholds)

    scored_rows: list[dict[str, Any]] = []
    for record in records:
        query = str(record.get("query", ""))
        language = str(record.get("language", "base"))
        gold_abstained = bool(record.get("target", {}).get("abstained", False))
        text_chunks = _to_retrieved_chunks(list(record.get("retrieved_chunks", [])))
        graph_chunks = graph_retriever.retrieve(query=query, language=language, k=cfg.pipeline.retrieval.graph_top_k)
        graph_support_score = max(
            [float(chunk.metadata.get("graph_path_score", chunk.score)) for chunk in graph_chunks] or [0.0]
        )
        graph_quality_score = max([_graph_quality_score(chunk) for chunk in graph_chunks] or [0.0])
        text_support_score = _estimate_text_support(query=query, chunks=text_chunks)

        variants = {
            "text-only-abstain": (
                _build_answer(
                    query=query,
                    retrieval_mode="text",
                    text_chunks=text_chunks,
                    graph_chunks=[],
                    text_support_score=text_support_score,
                    graph_support_score=0.0,
                    graph_quality_score=0.0,
                ),
                threshold_policy,
            ),
            "hybrid-threshold-abstain": (
                _build_answer(
                    query=query,
                    retrieval_mode="hybrid",
                    text_chunks=text_chunks,
                    graph_chunks=graph_chunks,
                    text_support_score=text_support_score,
                    graph_support_score=graph_support_score,
                    graph_quality_score=graph_quality_score,
                ),
                threshold_policy,
            ),
            "hybrid-graph-aware-abstain": (
                _build_answer(
                    query=query,
                    retrieval_mode="hybrid",
                    text_chunks=text_chunks,
                    graph_chunks=graph_chunks,
                    text_support_score=text_support_score,
                    graph_support_score=graph_support_score,
                    graph_quality_score=graph_quality_score,
                ),
                graph_aware_policy,
            ),
        }

        for variant, (answer, policy) in variants.items():
            scored_rows.append(
                {
                    "variant": variant,
                    "language": language,
                    "label_type": str(record.get("label_type", "unknown")),
                    "gold_abstained": gold_abstained,
                    "pred_abstained": bool(policy.should_abstain(answer)),
                    "text_support_score": text_support_score,
                    "graph_support_score": graph_support_score,
                    "graph_quality_score": graph_quality_score,
                }
            )

    rows_df = pl.DataFrame(scored_rows)
    summary_rows = []
    for variant in rows_df.get_column("variant").unique().to_list():
        variant_rows = [row for row in scored_rows if row["variant"] == variant]
        summary_rows.append(_score_variant_rows(variant_rows, variant))
    summary_df = pl.DataFrame(summary_rows).sort("abstain_accuracy", descending=True)

    by_language_rows = []
    for variant in rows_df.get_column("variant").unique().to_list():
        for language in rows_df.get_column("language").unique().to_list():
            subset = [
                row
                for row in scored_rows
                if row["variant"] == variant and row["language"] == language
            ]
            if not subset:
                continue
            by_language_rows.append({"language": language, **_score_variant_rows(subset, variant)})
    by_language_df = pl.DataFrame(by_language_rows).sort(["language", "abstain_accuracy"], descending=[False, True])

    output_dir = project_root / "artifacts" / "tables"
    write_parquet(rows_df, output_dir / "hybrid_abstention_rows.parquet")
    write_parquet(summary_df, output_dir / "hybrid_abstention_summary.parquet")
    write_parquet(by_language_df, output_dir / "hybrid_abstention_by_language.parquet")
    _write_report(summary_df=summary_df, by_language_df=by_language_df, report_path=output_dir / "hybrid_abstention_report.md")
    print(f"Wrote hybrid abstention outputs to {output_dir}")


if __name__ == "__main__":
    main()