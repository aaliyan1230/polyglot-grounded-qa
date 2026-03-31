from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _split_bucket(
    rows: list[dict[str, Any]], train_ratio: float, val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(rows)
    train_n = round(n * train_ratio)
    val_n = round(n * val_ratio)

    # For small language buckets, keep at least one sample for val and test.
    if n >= 4 and val_n == 0:
        val_n = 1
    test_n = n - train_n - val_n
    if n >= 4 and test_n == 0:
        test_n = 1
        train_n = max(0, n - val_n - test_n)

    # Clamp to valid ranges after safeguards.
    train_n = max(0, min(train_n, n))
    val_n = max(0, min(val_n, n - train_n))

    train = rows[:train_n]
    val = rows[train_n : train_n + val_n]
    test = rows[train_n + val_n :]
    return train, val, test


def _abstain_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    abstained = sum(1 for row in rows if bool(row["target"].get("abstained", False)))
    return abstained / len(rows)


def _rebalance_abstain_ratio(
    donor: list[dict[str, Any]], recipient: list[dict[str, Any]], min_ratio: float
) -> None:
    if not recipient:
        return

    recipient_abstained = sum(
        1 for row in recipient if bool(row["target"].get("abstained", False))
    )
    needed = math.ceil(min_ratio * len(recipient)) - recipient_abstained
    if needed <= 0:
        return

    donor_abstained_idx = [
        idx for idx, row in enumerate(donor) if bool(row["target"].get("abstained", False))
    ]
    recipient_non_abstained_idx = [
        idx for idx, row in enumerate(recipient) if not bool(row["target"].get("abstained", False))
    ]

    swaps = min(needed, len(donor_abstained_idx), len(recipient_non_abstained_idx))
    for i in range(swaps):
        d_idx = donor_abstained_idx[i]
        r_idx = recipient_non_abstained_idx[i]
        donor[d_idx], recipient[r_idx] = recipient[r_idx], donor[d_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create language-aware splits for SFT JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks/finetune/sft_dataset.jsonl"),
        help="Input SFT JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks/finetune"),
        help="Output directory for train/val/test JSONL files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-abstain-ratio",
        type=float,
        default=0.2,
        help="Minimum abstained proportion required in each split.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require train > 0, val >= 0, and train + val < 1")

    project_root = Path(__file__).resolve().parents[1]
    rows = _load_jsonl(project_root / args.input)
    if not rows:
        raise ValueError("Input dataset is empty")

    rng = random.Random(args.seed)

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        language = str(row.get("language", "base"))
        by_language[language].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for _, bucket in by_language.items():
        rng.shuffle(bucket)
        train, val, test = _split_bucket(bucket, args.train_ratio, args.val_ratio)
        train_rows.extend(train)
        val_rows.extend(val)
        test_rows.extend(test)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    # Ensure val/test hit minimum abstention ratio when possible using train donors.
    _rebalance_abstain_ratio(train_rows, val_rows, args.min_abstain_ratio)
    _rebalance_abstain_ratio(train_rows, test_rows, args.min_abstain_ratio)

    train_abstain = _abstain_ratio(train_rows)
    val_abstain = _abstain_ratio(val_rows)
    test_abstain = _abstain_ratio(test_rows)

    for split_name, ratio in [
        ("train", train_abstain),
        ("val", val_abstain),
        ("test", test_abstain),
    ]:
        if ratio < args.min_abstain_ratio and split_name != "train":
            raise ValueError(
                f"{split_name} abstain ratio {ratio:.3f} is below required minimum {args.min_abstain_ratio:.3f}. "
                "Generate more abstention samples or lower --min-abstain-ratio."
            )

    output_dir = project_root / args.output_dir
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "val.jsonl", val_rows)
    _write_jsonl(output_dir / "test.jsonl", test_rows)

    print(f"Wrote train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    print(
        "Abstain ratios: "
        f"train={train_abstain:.3f}, val={val_abstain:.3f}, test={test_abstain:.3f}"
    )


if __name__ == "__main__":
    main()
