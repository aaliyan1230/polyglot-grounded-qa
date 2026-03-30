from __future__ import annotations

from pathlib import Path

import polars as pl

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk
from polyglot_grounded_qa.utils.io import write_parquet


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "artifacts" / "indexes" / "seed_chunks.parquet"

    chunks = [
        RetrievedChunk(
            doc_id="seed-doc-1",
            chunk_id="seed-chunk-1",
            text="Grounded QA answers with citations tied to retrieved evidence.",
            score=0.1,
        ),
        RetrievedChunk(
            doc_id="seed-doc-2",
            chunk_id="seed-chunk-2",
            text="Locale packs inherit from base packs and override only necessary behavior.",
            score=0.1,
        ),
    ]

    rows = [chunk.model_dump(exclude={"metadata"}) for chunk in chunks]
    df = pl.DataFrame(rows)
    write_parquet(df, output_path)
    print(f"Wrote {len(df)} chunks to {output_path}")


if __name__ == "__main__":
    main()
