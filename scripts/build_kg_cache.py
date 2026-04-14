from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths
from polyglot_grounded_qa.utils.io import write_parquet


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "artifacts" / "indexes" / "kg_seed_paths.parquet"

    rows: list[dict[str, object]] = []
    for path in get_seed_graph_paths():
        for language in path.languages:
            rows.append(
                {
                    "path_id": path.path_id,
                    "language": language,
                    "path_length": len(path.triples),
                    "score": path.score,
                    "path_text": path.render_text(),
                    "alias_count": len(path.metadata.get("aliases", [])),
                }
            )

    df = pl.DataFrame(rows)
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} KG cache rows to {output_path}")


if __name__ == "__main__":
    main()