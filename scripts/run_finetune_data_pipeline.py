from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _has_datasets() -> bool:
    return importlib.util.find_spec("datasets") is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run finetune data pipeline end-to-end.")
    parser.add_argument(
        "--with-public",
        action="store_true",
        help="Include public XQuAD ingestion when datasets package is available.",
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=80,
        help="Per-language sample cap for public ingestion.",
    )
    parser.add_argument(
        "--synthetic-negatives-per-language",
        type=int,
        default=20,
        help="Public synthetic abstention examples per language.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    _run([py, "scripts/build_sft_dataset.py"], cwd=root)

    if args.with_public:
        if _has_datasets():
            _run(
                [
                    py,
                    "scripts/ingest_public_qa.py",
                    "--max-per-language",
                    str(args.max_per_language),
                    "--synthetic-negatives-per-language",
                    str(args.synthetic_negatives_per_language),
                ],
                cwd=root,
            )
        else:
            print("Skipping public ingest: optional dependency 'datasets' is not installed.")

    _run([py, "scripts/merge_sft_datasets.py"], cwd=root)
    _run(
        [
            py,
            "scripts/split_sft_dataset.py",
            "--input",
            "data/benchmarks/finetune/sft_dataset_merged.jsonl",
        ],
        cwd=root,
    )
    _run([py, "scripts/format_sft_for_training.py"], cwd=root)
    _run(
        [
            py,
            "scripts/analyze_sft_dataset.py",
            "--input",
            "data/benchmarks/finetune/sft_dataset_merged.jsonl",
        ],
        cwd=root,
    )

    print("Finetune data pipeline completed.")


if __name__ == "__main__":
    main()
