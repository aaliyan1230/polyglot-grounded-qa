from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge internal and public SFT datasets.")
    parser.add_argument(
        "--internal",
        type=Path,
        default=Path("data/benchmarks/finetune/sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--public",
        type=Path,
        default=Path("data/benchmarks/finetune/public_sft_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/finetune/sft_dataset_merged.jsonl"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    internal_rows = _read_jsonl(project_root / args.internal)
    public_rows = _read_jsonl(project_root / args.public)

    merged = public_rows + internal_rows
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in merged:
        key = (
            str(row.get("language", "")),
            str(row.get("query", "")),
            str(row.get("label_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Internal rows: {len(internal_rows)}")
    print(f"Public rows:   {len(public_rows)}")
    print(f"Merged rows:   {len(deduped)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
