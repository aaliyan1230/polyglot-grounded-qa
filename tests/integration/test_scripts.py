from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_pipeline_script() -> None:
    root = Path(__file__).resolve().parents[2]
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/run_pipeline.py",
        "What is grounded QA?",
        "--language",
        "base",
    ]
    subprocess.run(cmd, cwd=root, check=True)
