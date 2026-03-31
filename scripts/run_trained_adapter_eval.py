from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate + normalize + evaluate a trained adapter in one command."
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
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF base model used with adapter for inference.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("artifacts/runs/finetune_unsloth/lora_adapter"),
        help="Path to saved adapter directory.",
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
        default=192,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to evaluation history instead of overwriting.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    adapter_abs = root / args.adapter_path
    if not adapter_abs.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_abs}\n"
            "Train first, then place adapter files there (or pass --adapter-path)."
        )

    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/generate_tuned_predictions.py",
            "--mode",
            "hf-adapter",
            "--test-file",
            str(args.test_file),
            "--base-model",
            args.base_model,
            "--adapter-path",
            str(args.adapter_path),
            "--output",
            str(args.raw_output),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
        ],
        cwd=root,
    )

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
