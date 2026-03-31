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
    pipeline = create_default_pipeline(str(project_root))

    settings = [
        {"name": "baseline", "k": 10},
        {"name": "no_rerank", "k": 1},
    ]
    rows: list[dict[str, object]] = []
    query = "How does locale inheritance work?"
    for setting in settings:
        pipeline.top_k_rerank = int(setting["k"])
        result = pipeline.run(query=query, language=language)
        rows.append(
            {
                **metadata,
                "variant": setting["name"],
                "answer": result.answer,
                "abstained": result.abstained,
                "citation_count": len(result.citations),
            }
        )

    df = pl.DataFrame(rows)
    output_path = project_root / "artifacts" / "tables" / "ablation_results.parquet"
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} ablation rows to {output_path}")


if __name__ == "__main__":
    main()
