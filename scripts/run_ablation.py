from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa import create_default_pipeline
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.utils.io import write_parquet
from polyglot_grounded_qa.utils.run_metadata import build_run_metadata


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)
    language = cfg.pipeline.default_language
    metadata = build_run_metadata(cfg=cfg, language=language)

    settings = [
        {"name": "text-only", "mode": "text"},
        {"name": "kg-only", "mode": "graph"},
        {"name": "hybrid", "mode": "hybrid", "policy": "naive"},
        {"name": "hybrid-path-filtered", "mode": "hybrid", "policy": "filtered"},
        {"name": "hybrid-routed", "mode": "hybrid", "policy": "routed"},
    ]
    rows: list[dict[str, object]] = []
    query = "How does locale inheritance work?"
    for setting in settings:
        pipeline = create_default_pipeline(
            str(project_root),
            retrieval_mode=str(setting["mode"]),
            hybrid_policy=str(setting.get("policy")) if setting.get("policy") is not None else None,
        )
        result = pipeline.run(query=query, language=language)
        rows.append(
            {
                **metadata,
                "variant": setting["name"],
                "answer": result.answer,
                "abstained": result.abstained,
                "citation_count": len(result.citations),
                "retrieval_mode": result.metadata.get("retrieval_mode", "text"),
                "hybrid_policy": result.metadata.get("hybrid_policy", "naive"),
                "routing_decision": result.metadata.get("routing_decision", "static"),
                "text_evidence_count": result.metadata.get("text_evidence_count", 0),
                "graph_evidence_count": result.metadata.get("graph_evidence_count", 0),
                "graph_support_score": result.metadata.get("graph_support_score", 0.0),
                "graph_quality_score": result.metadata.get("graph_quality_score", 0.0),
            }
        )

    df = pl.DataFrame(rows)
    output_path = project_root / "artifacts" / "tables" / "ablation_results.parquet"
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} ablation rows to {output_path}")


if __name__ == "__main__":
    main()
