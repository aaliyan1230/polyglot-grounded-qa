from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _maybe_run_prediction_variant(
    root: Path,
    variant: str,
    prediction_path: Path,
    append: bool,
) -> None:
    if not prediction_path.exists():
        print(f"Skipping {variant}: missing predictions file {prediction_path}")
        return

    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_finetune_eval.py",
        "--variant",
        variant,
        "--predictions",
        str(prediction_path),
    ]
    if append:
        cmd.append("--append")
    _run(cmd, cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command final-results refresh without manual notebook execution."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip scripts/run_eval.py.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip scripts/run_ablation.py.")
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip finetune summary refresh from prediction JSONLs.",
    )
    parser.add_argument(
        "--include-adapter-row",
        action="store_true",
        help=(
            "Run scripts/run_trained_adapter_eval.py to append a real tuned-adapter row. "
            "This is typically GPU/runtime-heavy."
        ),
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite finetune eval parquet files instead of appending history.",
    )
    parser.add_argument(
        "--require-finetune",
        action="store_true",
        help="Fail if finetune artifacts are missing at final materialization step.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    append = not args.no_append

    if not args.skip_eval:
        _run(["uv", "run", "python", "scripts/run_eval.py"], cwd=root)

    if not args.skip_ablation:
        _run(["uv", "run", "python", "scripts/run_ablation.py"], cwd=root)

    _run(["uv", "run", "python", "scripts/build_kg_cache.py"], cwd=root)
    _run(["uv", "run", "python", "scripts/analyze_kg_coverage.py"], cwd=root)
    _run(["uv", "run", "python", "scripts/analyze_kg_path_quality.py"], cwd=root)

    if not args.skip_finetune:
        base_cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_finetune_eval.py",
            "--variant",
            "baseline-pipeline",
        ]
        if append:
            base_cmd.append("--append")
        _run(base_cmd, cwd=root)

        _maybe_run_prediction_variant(
            root=root,
            variant="tuned-control-baseline",
            prediction_path=root / "artifacts/runs/tuned_predictions_control.jsonl",
            append=True,
        )
        _maybe_run_prediction_variant(
            root=root,
            variant="grounded-heuristic-v1",
            prediction_path=root / "artifacts/runs/tuned_predictions_heuristic.jsonl",
            append=True,
        )
        _maybe_run_prediction_variant(
            root=root,
            variant="oracle-upper-bound",
            prediction_path=root / "artifacts/runs/oracle_like_predictions.jsonl",
            append=True,
        )

    if args.include_adapter_row:
        adapter_cmd = [
            "uv",
            "run",
            "python",
            "scripts/run_trained_adapter_eval.py",
            "--variant",
            "tuned-adapter-v1",
        ]
        if append:
            adapter_cmd.append("--append")
        _run(adapter_cmd, cwd=root)

    finalize_cmd = [
        "uv",
        "run",
        "python",
        "scripts/materialize_final_results.py",
    ]
    if args.require_finetune:
        finalize_cmd.append("--require-finetune")
    _run(finalize_cmd, cwd=root)

    contract_cmd = [
        "uv",
        "run",
        "python",
        "scripts/check_final_artifacts_contract.py",
    ]
    if args.require_finetune and not args.skip_finetune:
        contract_cmd.append("--require-finetune")
    else:
        contract_cmd.append("--core-only")
    _run(contract_cmd, cwd=root)

    print("Done. Final result artifacts are refreshed and contract-validated.")


if __name__ == "__main__":
    main()
