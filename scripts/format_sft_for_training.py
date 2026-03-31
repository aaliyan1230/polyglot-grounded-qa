from __future__ import annotations

import argparse
import json
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


def _retrieval_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "[NO_RETRIEVED_EVIDENCE]"
    lines = []
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        text = str(chunk.get("text", "")).strip()
        lines.append(f"[{chunk_id}] {text}")
    return "\n".join(lines)


def _target_json(row: dict[str, Any]) -> str:
    target = row.get("target", {})
    payload = {
        "answer": target.get("answer", ""),
        "citations": target.get("citations", []),
        "abstained": bool(target.get("abstained", False)),
        "reason": target.get("reason", ""),
    }
    return json.dumps(payload, ensure_ascii=True)


def _instruction(row: dict[str, Any]) -> str:
    language = str(row.get("language", "base"))
    query = str(row.get("query", "")).strip()
    evidence = _retrieval_block(row.get("retrieved_chunks", []))
    return (
        "You are a grounded QA model. Use only evidence below.\n"
        "If evidence is insufficient, abstain.\n"
        "Return strict JSON with keys: answer, citations, abstained, reason.\n\n"
        f"Language: {language}\n"
        f"Query: {query}\n"
        f"Evidence:\n{evidence}"
    )


def _build_chat_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "language": row.get("language", "base"),
        "messages": [
            {"role": "system", "content": "Ground answers in evidence and abstain when unsure."},
            {"role": "user", "content": _instruction(row)},
            {"role": "assistant", "content": _target_json(row)},
        ],
        "source": row.get("source", ""),
        "label_type": row.get("label_type", ""),
    }


def _build_text_record(row: dict[str, Any]) -> dict[str, Any]:
    prompt = _instruction(row)
    completion = _target_json(row)
    return {
        "id": row.get("id", ""),
        "language": row.get("language", "base"),
        "text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}",
        "source": row.get("source", ""),
        "label_type": row.get("label_type", ""),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format SFT splits for training frameworks.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/benchmarks/finetune"),
        help="Directory containing train/val/test jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks/finetune/formatted"),
        help="Output directory for formatted datasets.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    in_dir = project_root / args.input_dir
    out_dir = project_root / args.output_dir

    for split in ["train", "val", "test"]:
        rows = _read_jsonl(in_dir / f"{split}.jsonl")
        chat_rows = [_build_chat_record(row) for row in rows]
        text_rows = [_build_text_record(row) for row in rows]

        _write_jsonl(out_dir / f"{split}.chat.jsonl", chat_rows)
        _write_jsonl(out_dir / f"{split}.text.jsonl", text_rows)

    print(f"Wrote formatted splits to {out_dir}")


if __name__ == "__main__":
    main()
