from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Any

from polyglot_grounded_qa import create_default_pipeline
from polyglot_grounded_qa.core.config_loader import load_app_config


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
        lines.append(f"[{chunk.get('chunk_id', '')}] {str(chunk.get('text', '')).strip()}")
    return "\n".join(lines)


def _build_prompt(row: dict[str, Any]) -> str:
    language = str(row.get("language", "base"))
    query = str(row.get("query", "")).strip()
    evidence = _retrieval_block(row.get("retrieved_chunks", []))
    return (
        "You are a grounded QA model. Use only evidence below.\n"
        "If evidence is insufficient, abstain.\n"
        "Return strict JSON with keys: answer, citations, abstained, reason.\n\n"
        f"Language: {language}\n"
        f"Query: {query}\n"
        f"Evidence:\n{evidence}\n"
    )


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _baseline_predictions(rows: list[dict[str, Any]], project_root: Path) -> list[dict[str, Any]]:
    pipeline = create_default_pipeline(str(project_root))
    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        query = str(row.get("query", ""))
        language = str(row.get("language", "base"))
        result = pipeline.run(query=query, language=language)
        out.append(
            {
                "id": sample_id,
                "language": language,
                "query": query,
                "generated_text": json.dumps(
                    {
                        "answer": result.answer,
                        "citations": [c.chunk_id for c in result.citations],
                        "abstained": bool(result.abstained),
                        "reason": "",
                    },
                    ensure_ascii=True,
                ),
            }
        )
    return out


def _grounded_heuristic_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        language = str(row.get("language", "base"))
        query = str(row.get("query", ""))
        chunks = row.get("retrieved_chunks", [])

        if not isinstance(chunks, list) or not chunks:
            payload = {
                "answer": "I do not have enough evidence.",
                "citations": [],
                "abstained": True,
                "reason": "insufficient_evidence",
            }
        else:
            first = chunks[0]
            chunk_id = str(first.get("chunk_id", ""))
            text = str(first.get("text", "")).strip()

            # Extract a compact span-like answer from the first sentence for higher precision.
            answer_span = text.split(".")[0].strip()
            if not answer_span:
                answer_span = text[:160].strip()

            payload = {
                "answer": answer_span,
                "citations": [chunk_id] if chunk_id else [],
                "abstained": False,
                "reason": "",
            }

        out.append(
            {
                "id": sample_id,
                "language": language,
                "query": query,
                "generated_text": json.dumps(payload, ensure_ascii=True),
            }
        )
    return out


def _hf_adapter_predictions(
    rows: list[dict[str, Any]],
    base_model: str,
    adapter_path: str | None,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    torch_module = importlib.import_module("torch")
    transformers_module = importlib.import_module("transformers")

    AutoModelForCausalLM = transformers_module.AutoModelForCausalLM
    AutoTokenizer = transformers_module.AutoTokenizer

    use_cuda = torch_module.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch_module.float16

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter_path:
        peft_module = importlib.import_module("peft")
        PeftModel = peft_module.PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.to(device)
    model.eval()

    out: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("id", ""))
        language = str(row.get("language", "base"))
        query = str(row.get("query", ""))
        prompt = _build_prompt(row)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch_module.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Strip the input prompt tokens so we only decode the model's new output.
        new_tokens = generated[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        json_obj = _extract_json_obj(text)
        payload = json.dumps(json_obj if json_obj is not None else {"raw": text}, ensure_ascii=True)

        out.append(
            {
                "id": sample_id,
                "language": language,
                "query": query,
                "generated_text": payload,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for finetune evaluation.")
    parser.add_argument(
        "--mode",
        choices=["baseline-pipeline", "grounded-heuristic", "hf-adapter"],
        default="baseline-pipeline",
        help="Prediction backend mode.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/benchmarks/finetune/test.jsonl"),
        help="Grounded test JSONL used as input prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/raw_model_predictions.jsonl"),
        help="Output JSONL with generated_text field.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HF base model for hf-adapter mode.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path for hf-adapter mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=project_root)
    models_cfg = cfg.models if isinstance(cfg.models, dict) else {}
    finetune_cfg = models_cfg.get("finetune", {}) if isinstance(models_cfg, dict) else {}
    base_model = str(finetune_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct"))
    adapter_path = str(finetune_cfg.get("adapter_path", ""))
    max_new_tokens = int(finetune_cfg.get("max_new_tokens", 192))
    temperature = float(finetune_cfg.get("temperature", 0.0))

    if args.base_model:
        base_model = args.base_model
    if args.adapter_path:
        adapter_path = args.adapter_path
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        temperature = args.temperature
    rows = _read_jsonl(project_root / args.test_file)
    if not rows:
        raise ValueError("Test file is empty. Run finetune data pipeline first.")

    if args.mode == "baseline-pipeline":
        preds = _baseline_predictions(rows, project_root)
    elif args.mode == "grounded-heuristic":
        preds = _grounded_heuristic_predictions(rows)
    else:
        preds = _hf_adapter_predictions(
            rows=rows,
            base_model=base_model,
            adapter_path=adapter_path or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    out_path = project_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
