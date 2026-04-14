from __future__ import annotations

import argparse
from pathlib import Path

from polyglot_grounded_qa import create_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grounded QA pipeline for a single query.")
    parser.add_argument("query", type=str, help="Question to answer")
    parser.add_argument("--language", default="base", help="Language pack tag")
    parser.add_argument(
        "--retrieval-mode",
        choices=["text", "graph", "hybrid"],
        default=None,
        help="Override retrieval mode for this run.",
    )
    parser.add_argument(
        "--hybrid-policy",
        choices=["naive", "filtered", "routed"],
        default=None,
        help="Override the hybrid retrieval policy for this run.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pipeline = create_default_pipeline(
        str(project_root),
        retrieval_mode=args.retrieval_mode,
        hybrid_policy=args.hybrid_policy,
    )
    result = pipeline.run(query=args.query, language=args.language)

    print(f"Answer: {result.answer}")
    print(f"Abstained: {result.abstained}")
    print(f"Retrieval mode: {result.metadata.get('retrieval_mode', 'text')}")
    print(f"Hybrid policy: {result.metadata.get('hybrid_policy', 'naive')}")
    print(f"Routing decision: {result.metadata.get('routing_decision', 'static')}")
    print(f"Text evidence: {result.metadata.get('text_evidence_count', 0)}")
    print(f"Graph evidence: {result.metadata.get('graph_evidence_count', 0)}")
    print(f"Citations: {[c.chunk_id for c in result.citations]}")


if __name__ == "__main__":
    main()
