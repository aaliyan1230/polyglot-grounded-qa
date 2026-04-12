from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from polyglot_grounded_qa.core.config_loader import load_app_config


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate + normalize + evaluate a prompted base model or trained adapter in one command."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tuned-adapter-v1",
        help="Variant label appended to finetune eval outputs.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HF base model used with adapter for inference.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to saved adapter directory. Omit when running --no-adapter.",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base model with the same prompt/test set but without loading a LoRA adapter.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/benchmarks/finetune/test.jsonl"),
        help="Evaluation test split path.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("artifacts/runs/raw_model_predictions_adapter.jsonl"),
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=Path("artifacts/runs/tuned_predictions_adapter.jsonl"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to evaluation history instead of overwriting.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_app_config(project_root=root)
    models_cfg = cfg.models if isinstance(cfg.models, dict) else {}
    finetune_cfg = models_cfg.get("finetune", {}) if isinstance(models_cfg, dict) else {}
    base_model = str(finetune_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct"))
    adapter_path = str(finetune_cfg.get("adapter_path", ""))
    max_new_tokens = int(finetune_cfg.get("max_new_tokens", 192))
    temperature = float(finetune_cfg.get("temperature", 0.0))

    if args.base_model:
        base_model = args.base_model
    if args.adapter_path is not None:
        adapter_path = str(args.adapter_path)
    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        temperature = args.temperature

    if args.no_adapter:
        adapter_path = ""

    if not args.no_adapter and not adapter_path:
        raise ValueError(
            "Adapter path is empty. Set models.finetune.adapter_path in configs/models/default.yaml "
            "or pass --adapter-path, or use --no-adapter to evaluate the base model only."
        )

    adapter_abs = root / adapter_path if adapter_path else None
    if adapter_abs is not None and not adapter_abs.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_abs}\n"
            "Train first, then place adapter files there (or pass --adapter-path)."
        )

    generate_cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_tuned_predictions.py",
        "--mode",
        "hf-adapter",
        "--test-file",
        str(args.test_file),
        "--base-model",
        base_model,
        "--output",
        str(args.raw_output),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
    ]
    if adapter_path:
        generate_cmd.extend(["--adapter-path", adapter_path])

    _run(generate_cmd, cwd=root)

    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/normalize_tuned_predictions.py",
            "--input",
            str(args.raw_output),
            "--output",
            str(args.normalized_output),
        ],
        cwd=root,
    )

    eval_cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_finetune_eval.py",
        "--variant",
        args.variant,
        "--predictions",
        str(args.normalized_output),
        "--test-file",
        str(args.test_file),
    ]
    if args.append:
        eval_cmd.append("--append")

    _run(eval_cmd, cwd=root)
    print("Done. You can now rerun the final notebook delta + takeaway cells.")


if __name__ == "__main__":
    main()
