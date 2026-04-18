from __future__ import annotations

from pathlib import Path

from polyglot_grounded_qa.core.kg_cache import load_graph_paths, serialize_graph_paths, write_graph_cache
from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths


def test_serialize_graph_paths_includes_cache_columns() -> None:
    df = serialize_graph_paths(get_seed_graph_paths())

    assert {
        "path_id",
        "language",
        "source",
        "aliases_json",
        "triples_json",
        "metadata_json",
    }.issubset(set(df.columns))


def test_load_graph_paths_roundtrips_from_cache(tmp_path: Path) -> None:
    project_root = tmp_path
    cache_path = project_root / "artifacts" / "indexes" / "kg_seed_paths.parquet"
    write_graph_cache(get_seed_graph_paths(), cache_path)

    paths = load_graph_paths(project_root)

    assert paths
    assert any(path.metadata.get("aliases") for path in paths)
    assert any(path.metadata.get("source", "seed") == "seed" for path in paths)
    assert all(not path.path_id.startswith("(") for path in paths)