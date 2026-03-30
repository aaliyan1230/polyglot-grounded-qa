from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute project notebooks in order.")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    notebooks = sorted((project_root / "notebooks").glob("*.ipynb"))

    for notebook in notebooks:
        cmd = [
            "uv",
            "run",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--inplace",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            f"--ExecutePreprocessor.kernel_name={args.kernel}",
            str(notebook),
        ]
        print(f"Executing {notebook.name}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
