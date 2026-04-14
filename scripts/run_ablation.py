from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa import create_default_pipeline
from polyglot_grounded_qa.core.config_loader import load_app_config
from polyglot_grounded_qa.utils.io import write_parquet
from polyglot_grounded_qa.utils.run_metadata import build_run_metadata


ABLATION_QUERY_MATRIX: dict[str, list[dict[str, str]]] = {
    "base": [
        {"query": "What is grounded QA?", "category": "entity-factual"},
        {"query": "How does locale inheritance work?", "category": "mechanism"},
        {"query": "Why are citations required in grounded QA?", "category": "justification"},
        {"query": "When should the system abstain?", "category": "abstention"},
    ],
    "es": [
        {"query": "Que es grounded QA?", "category": "entity-factual"},
        {"query": "Como funciona la herencia de locale?", "category": "mechanism"},
        {"query": "Por que se requieren citas en grounded QA?", "category": "justification"},
        {"query": "Cuando debe abstenerse el sistema?", "category": "abstention"},
    ],
    "fr": [
        {"query": "Qu est ce que le grounded QA?", "category": "entity-factual"},
        {"query": "Comment fonctionne l heritage des langues?", "category": "mechanism"},
        {"query": "Pourquoi les citations sont elles requises?", "category": "justification"},
        {"query": "Quand faut il s abstenir?", "category": "abstention"},
    ],
    "tr": [
        {"query": "Grounded QA nedir?", "category": "entity-factual"},
        {"query": "Dil paketleri mirasi nasil calisir?", "category": "mechanism"},
        {"query": "Grounded QA icin neden atif gerekir?", "category": "justification"},
        {"query": "Sistem ne zaman cekimser kalmali?", "category": "abstention"},
    ],
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)

    settings = [
        {"name": "text-only", "mode": "text"},
        {"name": "kg-only", "mode": "graph"},
        {"name": "hybrid", "mode": "hybrid", "policy": "naive"},
        {
            "name": "hybrid-path-filtered",
            "mode": "hybrid",
            "policy": "filtered",
            "overrides": {"graph_min_quality_score": 0.78},
        },
        {"name": "hybrid-routed", "mode": "hybrid", "policy": "routed"},
    ]
    rows: list[dict[str, object]] = []
    languages = [tag for tag in cfg.languages if tag in ABLATION_QUERY_MATRIX]
    for ablation_language in languages:
        language_metadata = build_run_metadata(cfg=cfg, language=ablation_language)
        for setting in settings:
            pipeline = create_default_pipeline(
                str(project_root),
                retrieval_mode=str(setting["mode"]),
                hybrid_policy=str(setting.get("policy")) if setting.get("policy") is not None else None,
                retrieval_overrides=setting.get("overrides"),
            )
            for query_spec in ABLATION_QUERY_MATRIX[ablation_language]:
                result = pipeline.run(query=query_spec["query"], language=ablation_language)
                rows.append(
                    {
                        **language_metadata,
                        "variant": setting["name"],
                        "query": query_spec["query"],
                        "query_category": query_spec["category"],
                        "answer": result.answer,
                        "abstained": result.abstained,
                        "citation_count": len(result.citations),
                        "retrieval_mode": result.metadata.get("retrieval_mode", "text"),
                        "hybrid_policy": result.metadata.get("hybrid_policy", "naive"),
                        "routing_decision": result.metadata.get("routing_decision", "static"),
                        "top_evidence_type": result.metadata.get("top_evidence_type", "none"),
                        "top_chunk_id": result.metadata.get("top_chunk_id", ""),
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
