from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa.core.seed_data import get_seed_corpus
from polyglot_grounded_qa.utils.io import write_parquet


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "artifacts" / "indexes" / "seed_chunks.parquet"

    chunks = get_seed_corpus()

    rows = [chunk.model_dump(exclude={"metadata"}) for chunk in chunks]
    df = pl.DataFrame(rows)
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} chunks to {output_path}")


if __name__ == "__main__":
    main()
