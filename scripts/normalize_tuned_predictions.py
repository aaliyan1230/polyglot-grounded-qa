from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    # Fast path: valid JSON as-is.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Fallback: extract first {...} block.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    block = match.group(0)
    try:
        obj = json.loads(block)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_record(row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    sample_id = str(row.get("id", f"line-{idx}"))

    # Case 1: already in required shape.
    if {"answer", "citations", "abstained"}.issubset(set(row.keys())):
        citations = row.get("citations", [])
        return {
            "id": sample_id,
            "answer": str(row.get("answer", "")),
            "citations": list(citations) if isinstance(citations, list) else [],
            "abstained": _to_bool(row.get("abstained", False)),
        }

    # Case 2: generated_text / prediction text containing JSON.
    candidate_text = ""
    for field in ["prediction", "generated_text", "output", "response", "assistant"]:
        if field in row:
            candidate_text = str(row[field])
            break

    obj = _extract_json_obj(candidate_text)
    if obj is None:
        return None

    citations = obj.get("citations", [])
    return {
        "id": sample_id,
        "answer": str(obj.get("answer", "")),
        "citations": list(citations) if isinstance(citations, list) else [],
        "abstained": _to_bool(obj.get("abstained", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize tuned model predictions for run_finetune_eval.")
    parser.add_argument("--input", type=Path, required=True, help="Raw predictions JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/tuned_predictions_normalized.jsonl"),
        help="Normalized predictions JSONL (id, answer, citations, abstained).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rows = _read_jsonl(project_root / args.input)

    normalized: list[dict[str, Any]] = []
    dropped = 0
    for idx, row in enumerate(rows):
        out = _normalize_record(row, idx)
        if out is None:
            dropped += 1
            continue
        normalized.append(out)

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Input rows: {len(rows)}")
    print(f"Normalized rows: {len(normalized)}")
    print(f"Dropped rows: {dropped}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
