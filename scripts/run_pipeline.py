from __future__ import annotations

import argparse
from pathlib import Path

from polyglot_grounded_qa import create_default_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grounded QA pipeline for a single query.")
    parser.add_argument("query", type=str, help="Question to answer")
    parser.add_argument("--language", default="base", help="Language pack tag")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pipeline = create_default_pipeline(str(project_root))
    result = pipeline.run(query=args.query, language=args.language)

    print(f"Answer: {result.answer}")
    print(f"Abstained: {result.abstained}")
    print(f"Citations: {[c.chunk_id for c in result.citations]}")


if __name__ == "__main__":
    main()
