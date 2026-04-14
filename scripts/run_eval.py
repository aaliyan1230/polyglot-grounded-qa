from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from polyglot_grounded_qa import create_default_pipeline
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.utils.io import write_parquet
from polyglot_grounded_qa.utils.run_metadata import build_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for the configured retrieval mode.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["text", "graph", "hybrid"],
        default=None,
        help="Override retrieval mode for this evaluation run.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)
    language = cfg.pipeline.default_language
    metadata = build_run_metadata(cfg=cfg, language=language)
    pipeline = create_default_pipeline(str(project_root), retrieval_mode=args.retrieval_mode)

    queries = [
        "What is language-pack architecture?",
        "Why are citations required in grounded QA?",
    ]
    rows: list[dict[str, object]] = []
    for query in queries:
        result = pipeline.run(query=query, language=language)
        rows.append(
            {
                **metadata,
                "query": query,
                "answer": result.answer,
                "abstained": result.abstained,
                "citation_count": len(result.citations),
                "retrieval_mode": result.metadata.get("retrieval_mode", "text"),
                "text_evidence_count": result.metadata.get("text_evidence_count", 0),
                "graph_evidence_count": result.metadata.get("graph_evidence_count", 0),
                "graph_support_score": result.metadata.get("graph_support_score", 0.0),
            }
        )

    df = pl.DataFrame(rows)
    output_path = project_root / "artifacts" / "runs" / "eval_results.parquet"
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} eval rows to {output_path}")


if __name__ == "__main__":
    main()
