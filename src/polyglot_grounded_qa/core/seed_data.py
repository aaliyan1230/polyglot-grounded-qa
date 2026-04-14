from __future__ import annotations

from polyglot_grounded_qa.schemas.contracts import KnowledgeGraphPath, KnowledgeGraphTriple, RetrievedChunk


def get_seed_corpus() -> list[RetrievedChunk]:
    """Return the canonical seed corpus used by baseline scripts and pipeline."""
    return [
        RetrievedChunk(
            doc_id="seed-doc-1",
            chunk_id="seed-chunk-1",
            text="Polyglot grounded QA uses retrieval with citations and verification.",
            score=0.1,
            metadata={"evidence_type": "text", "languages": ["base", "es", "fr", "tr"]},
        ),
        RetrievedChunk(
            doc_id="seed-doc-2",
            chunk_id="seed-chunk-2",
            text="Language packs inherit base behavior and only override locale-specific rules.",
            score=0.1,
            metadata={"evidence_type": "text", "languages": ["base", "es", "fr", "tr"]},
        ),
        RetrievedChunk(
            doc_id="seed-doc-3",
            chunk_id="seed-chunk-3",
            text="Abstention is preferred when evidence is weak, contradictory, or uncited.",
            score=0.1,
            metadata={"evidence_type": "text", "languages": ["base", "es", "fr", "tr"]},
        ),
        RetrievedChunk(
            doc_id="seed-doc-4",
            chunk_id="seed-chunk-4",
            text="Hybrid retrieval combines text evidence with knowledge graph paths for grounded answers.",
            score=0.1,
            metadata={"evidence_type": "text", "languages": ["base", "es", "fr", "tr"]},
        ),
    ]


def get_seed_graph_paths() -> list[KnowledgeGraphPath]:
    """Return a small KG cache used for hybrid retrieval scaffolding and tests."""
    return [
        KnowledgeGraphPath(
            path_id="kg-path-grounded-qa",
            triples=[
                KnowledgeGraphTriple(
                    subject="Grounded QA",
                    relation="requires",
                    object="retrieval-backed evidence",
                ),
                KnowledgeGraphTriple(
                    subject="retrieval-backed evidence",
                    relation="supports",
                    object="citations and verification",
                ),
            ],
            score=0.82,
            languages=["base", "es", "fr", "tr"],
            metadata={
                "aliases": ["grounded qa", "retrieval", "citations", "verification"],
            },
        ),
        KnowledgeGraphPath(
            path_id="kg-path-language-packs",
            triples=[
                KnowledgeGraphTriple(
                    subject="Language packs",
                    relation="inherit",
                    object="base behavior",
                ),
                KnowledgeGraphTriple(
                    subject="Language packs",
                    relation="override",
                    object="locale-specific rules",
                ),
            ],
            score=0.78,
            languages=["base", "es", "fr", "tr"],
            metadata={
                "aliases": ["language packs", "locale inheritance", "base behavior"],
            },
        ),
        KnowledgeGraphPath(
            path_id="kg-path-abstention",
            triples=[
                KnowledgeGraphTriple(
                    subject="Abstention policy",
                    relation="triggers when",
                    object="evidence is weak",
                ),
                KnowledgeGraphTriple(
                    subject="Abstention policy",
                    relation="prefers",
                    object="no answer over unsupported answer",
                ),
            ],
            score=0.74,
            languages=["base", "fr", "tr"],
            metadata={
                "aliases": ["abstain", "abstention", "weak evidence", "unsupported answer"],
            },
        ),
        KnowledgeGraphPath(
            path_id="kg-path-hybrid-retrieval",
            triples=[
                KnowledgeGraphTriple(
                    subject="Hybrid retrieval",
                    relation="combines",
                    object="text retrieval",
                ),
                KnowledgeGraphTriple(
                    subject="Hybrid retrieval",
                    relation="adds",
                    object="knowledge graph support paths",
                ),
            ],
            score=0.76,
            languages=["base", "es", "fr"],
            metadata={
                "aliases": ["hybrid retrieval", "knowledge graph", "kg paths", "graph support"],
            },
        ),
    ]