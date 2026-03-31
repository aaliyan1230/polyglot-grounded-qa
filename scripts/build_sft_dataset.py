from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from polyglot_grounded_qa import create_default_pipeline
from polyglot_grounded_qa.core.config_loader import load_app_config


def _load_queries(eval_path: Path) -> list[str]:
    if not eval_path.exists():
        return []
    df = pl.read_parquet(eval_path)
    if "query" not in df.columns:
        return []
    return sorted({str(q) for q in df["query"].to_list() if q is not None})


def _build_example(
    sample_id: str,
    language: str,
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: str,
    citations: list[str],
    abstained: bool,
    label_type: str,
    source: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "language": language,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "target": {
            "answer": answer,
            "citations": citations,
            "abstained": abstained,
            "reason": reason,
        },
        "label_type": label_type,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-pass SFT JSONL for grounded QA.")
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("artifacts/runs/eval_results.parquet"),
        help="Parquet with eval queries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/finetune/sft_dataset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="",
        help="Comma-separated language tags. Default uses all configured languages.",
    )
    parser.add_argument(
        "--synthetic-negatives-per-query",
        type=int,
        default=1,
        help="How many abstention-only synthetic negatives to add per query-language pair.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)

    if args.languages.strip():
        languages = [tag.strip() for tag in args.languages.split(",") if tag.strip()]
    else:
        languages = sorted(cfg.languages.keys())

    queries = _load_queries(project_root / args.eval_path)
    if not queries:
        queries = [
            "What is grounded QA?",
            "How does language-pack inheritance work?",
            "Why are citations required?",
        ]

    pipeline = create_default_pipeline(str(project_root))
    rows: list[dict[str, Any]] = []
    counter = 0

    for language in languages:
        for query in queries:
            retrieved = pipeline.retriever.retrieve(
                query=query,
                language=language,
                k=pipeline.top_k_retrieve,
            )
            reranked = pipeline.reranker.rerank(
                query=query,
                chunks=retrieved,
                k=pipeline.top_k_rerank,
            )
            answer = pipeline.run(query=query, language=language)

            retrieved_chunks = [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                }
                for chunk in reranked
            ]
            citations = [citation.chunk_id for citation in answer.citations]
            label_type = "insufficient_evidence" if answer.abstained else "answerable"

            rows.append(
                _build_example(
                    sample_id=f"sft-{counter:06d}",
                    language=language,
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    answer=answer.answer,
                    citations=citations,
                    abstained=answer.abstained,
                    label_type=label_type,
                    source="eval+pipeline",
                )
            )
            counter += 1

            for neg_idx in range(args.synthetic_negatives_per_query):
                rows.append(
                    _build_example(
                        sample_id=f"sft-{counter:06d}",
                        language=language,
                        query=f"{query} [insufficient-evidence-{neg_idx}]",
                        retrieved_chunks=[],
                        answer="I do not have enough evidence.",
                        citations=[],
                        abstained=True,
                        label_type="insufficient_evidence",
                        source="synthetic-negative",
                        reason="insufficient_evidence",
                    )
                )
                counter += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (project_root / args.output).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    abstained_count = sum(1 for row in rows if row["target"]["abstained"])
    print(f"Wrote {len(rows)} samples to {project_root / args.output}")
    print(f"Abstention ratio: {abstained_count / max(len(rows), 1):.3f}")
    print(f"Languages: {languages}")


if __name__ == "__main__":
    main()
