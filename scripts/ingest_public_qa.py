from __future__ import annotations

import argparse
import importlib
import json
import random
from pathlib import Path
from typing import Any


def _build_example(
    sample_id: str,
    language: str,
    query: str,
    chunk_id: str,
    context: str,
    answer: str,
    abstained: bool,
    label_type: str,
    source: str,
) -> dict[str, Any]:
    citations = [] if abstained else [chunk_id]
    return {
        "id": sample_id,
        "language": language,
        "query": query,
        "retrieved_chunks": [
            {
                "doc_id": f"{source}-doc",
                "chunk_id": chunk_id,
                "text": context,
            }
        ],
        "target": {
            "answer": answer,
            "citations": citations,
            "abstained": abstained,
            "reason": "insufficient_evidence" if abstained else "",
        },
        "label_type": label_type,
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest public multilingual QA into grounded-QA SFT schema."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/finetune/public_sft_dataset.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-per-language",
        type=int,
        default=150,
        help="Maximum answerable samples per language from XQuAD.",
    )
    parser.add_argument(
        "--synthetic-negatives-per-language",
        type=int,
        default=30,
        help="Synthetic abstention examples to add per language.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-per-language",
        type=int,
        default=40,
        help="Minimum rows per language after fallback bootstrapping.",
    )
    args = parser.parse_args()

    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = datasets_module.load_dataset
    except Exception as exc:  # pragma: no cover - runtime guidance
        raise RuntimeError(
            "Missing dependency 'datasets'. Run with: uv run --with datasets python scripts/ingest_public_qa.py"
        ) from exc

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Source candidates in priority order. Some datasets do not cover all languages.
    source_candidates = {
        "base": [("google/xquad", "xquad.en"), ("mlqa", "mlqa.en.en")],
        "es": [("google/xquad", "xquad.es"), ("mlqa", "mlqa.es.es")],
        "es-MX": [("google/xquad", "xquad.es"), ("mlqa", "mlqa.es.es")],
        "fr": [("mlqa", "mlqa.fr.fr")],
        "tr": [("google/xquad", "xquad.tr")],
    }

    rows: list[dict[str, Any]] = []
    counter = 0
    by_language_rows: dict[str, list[dict[str, Any]]] = {
        "base": [],
        "es": [],
        "es-MX": [],
        "fr": [],
        "tr": [],
    }

    for repo_lang, candidates in source_candidates.items():
        ds = None
        selected_source = ""
        for dataset_name, config_name in candidates:
            try:
                ds = load_dataset(dataset_name, config_name, split="validation")
                selected_source = f"{dataset_name}:{config_name}"
                break
            except Exception:
                continue

        if ds is None:
            print(f"Skipping {repo_lang}: no dataset source available from candidates={candidates}")
            continue

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[: args.max_per_language]

        for i in selected:
            sample = ds[int(i)]
            qid = str(sample.get("id", f"public-{repo_lang}-{i}"))
            query = str(sample["question"]).strip()
            context = str(sample["context"]).strip()
            answers = sample.get("answers", {})
            answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
            answer_text = str(answer_texts[0]).strip() if answer_texts else ""
            if not query or not context:
                continue

            abstained = answer_text == ""
            label_type = "insufficient_evidence" if abstained else "answerable"
            final_answer = answer_text if answer_text else "I do not have enough evidence."
            chunk_id = f"{qid}-chunk-0"

            rows.append(
                _build_example(
                    sample_id=f"public-{counter:07d}",
                    language=repo_lang,
                    query=query,
                    chunk_id=chunk_id,
                    context=context,
                    answer=final_answer,
                    abstained=abstained,
                    label_type=label_type,
                    source=selected_source,
                )
            )
            by_language_rows[repo_lang].append(rows[-1])
            counter += 1

        # Add explicit abstention supervision per language.
        for nidx in range(args.synthetic_negatives_per_language):
            rows.append(
                {
                    "id": f"public-{counter:07d}",
                    "language": repo_lang,
                    "query": f"Unanswerable verification probe {nidx} for {repo_lang}",
                    "retrieved_chunks": [],
                    "target": {
                        "answer": "I do not have enough evidence.",
                        "citations": [],
                        "abstained": True,
                        "reason": "insufficient_evidence",
                    },
                    "label_type": "insufficient_evidence",
                    "source": "synthetic-negative:public",
                }
            )
            by_language_rows[repo_lang].append(rows[-1])
            counter += 1

    # Bootstrap missing languages from base samples when public sources are sparse.
    base_pool = [row for row in by_language_rows.get("base", []) if row["label_type"] == "answerable"]
    for language, lang_rows in by_language_rows.items():
        if len(lang_rows) >= args.min_per_language:
            continue
        if not base_pool:
            continue

        needed = args.min_per_language - len(lang_rows)
        for idx in range(needed):
            template = base_pool[idx % len(base_pool)]
            copied = json.loads(json.dumps(template))
            copied["id"] = f"public-{counter:07d}"
            copied["language"] = language
            copied["query"] = f"{copied['query']} [bootstrap-{language}-{idx}]"
            copied["source"] = "bootstrap-from-base"
            rows.append(copied)
            by_language_rows[language].append(copied)
            counter += 1

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    abstained = sum(1 for row in rows if bool(row["target"].get("abstained", False)))
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Abstention ratio: {abstained / max(len(rows), 1):.3f}")


if __name__ == "__main__":
    main()
