from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from polyglot_grounded_qa.adapters.kg import WikidataKGAdapter, get_default_concept_seeds
from polyglot_grounded_qa.core.kg_cache import (
    KG_CACHE_RELATIVE_PATH,
    KG_CACHE_REQUIRED_COLUMNS,
    write_graph_cache,
)
from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths


def _dedupe_paths(paths: list[object]) -> list[object]:
    deduped: dict[str, object] = {}
    for path in paths:
        deduped[path.path_id] = path
    return list(deduped.values())


def _has_reusable_cache(output_path: Path) -> bool:
    if not output_path.exists():
        return False
    df = pl.read_parquet(output_path)
    return KG_CACHE_REQUIRED_COLUMNS.issubset(set(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local KG cache with public Wikidata paths and seed fallback.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip public KG fetches and build the cache from local seed paths only.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild the cache even if an existing parquet is already present.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / KG_CACHE_RELATIVE_PATH
    if _has_reusable_cache(output_path) and not args.refresh:
        print(f"Reusing existing KG cache at {output_path}")
        return

    seed_paths = get_seed_graph_paths()
    fetched_paths = []
    if not args.offline:
        adapter = WikidataKGAdapter()
        fetched_paths = adapter.fetch_paths(get_default_concept_seeds())

    combined_paths = _dedupe_paths([*fetched_paths, *seed_paths])
    write_graph_cache(combined_paths, output_path)
    print(
        f"Wrote {len(combined_paths)} KG paths to {output_path} "
        f"({len(fetched_paths)} public, {len(seed_paths)} seed fallback)"
    )


if __name__ == "__main__":
    main()