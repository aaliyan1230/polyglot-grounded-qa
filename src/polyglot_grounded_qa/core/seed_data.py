from __future__ import annotations

from polyglot_grounded_qa.schemas.contracts import RetrievedChunk


def get_seed_corpus() -> list[RetrievedChunk]:
    """Return the canonical seed corpus used by baseline scripts and pipeline."""
    return [
        RetrievedChunk(
            doc_id="seed-doc-1",
            chunk_id="seed-chunk-1",
            text="Polyglot grounded QA uses retrieval with citations and verification.",
            score=0.1,
        ),
        RetrievedChunk(
            doc_id="seed-doc-2",
            chunk_id="seed-chunk-2",
            text="Language packs inherit base behavior and only override locale-specific rules.",
            score=0.1,
        ),
    ]