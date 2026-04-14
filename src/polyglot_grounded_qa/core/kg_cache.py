from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths
from polyglot_grounded_qa.schemas.contracts import KnowledgeGraphPath, KnowledgeGraphTriple
from polyglot_grounded_qa.utils.io import write_parquet

KG_CACHE_RELATIVE_PATH = Path("artifacts/indexes/kg_seed_paths.parquet")
KG_CACHE_REQUIRED_COLUMNS = {
    "path_id",
    "language",
    "path_length",
    "score",
    "path_text",
    "source",
    "aliases_json",
    "triples_json",
    "metadata_json",
}


def _serialize_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def serialize_graph_paths(paths: list[KnowledgeGraphPath]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        aliases = list(path.metadata.get("aliases", []))
        metadata = {key: value for key, value in path.metadata.items() if key != "aliases"}
        triples = [triple.model_dump() for triple in path.triples]
        languages = path.languages or ["base"]
        for language in languages:
            rows.append(
                {
                    "path_id": path.path_id,
                    "language": language,
                    "path_length": len(path.triples),
                    "score": path.score,
                    "path_text": path.render_text(),
                    "alias_count": len(aliases),
                    "source": str(path.metadata.get("source", "seed")),
                    "entity_id": path.metadata.get("entity_id"),
                    "property_id": path.metadata.get("property_id"),
                    "object_id": path.metadata.get("object_id"),
                    "aliases_json": _serialize_json(aliases),
                    "triples_json": _serialize_json(triples),
                    "metadata_json": _serialize_json(metadata),
                }
            )
    return pl.DataFrame(rows)


def _parse_json_list(raw_value: object) -> list[object]:
    if not isinstance(raw_value, str):
        return []
    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def _parse_json_object(raw_value: object) -> dict[str, object]:
    if not isinstance(raw_value, str):
        return {}
    try:
        loaded = json.loads(raw_value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def deserialize_graph_paths(df: pl.DataFrame) -> list[KnowledgeGraphPath]:
    if df.is_empty():
        return []

    paths: list[KnowledgeGraphPath] = []
    for path_id, group in df.group_by("path_id"):
        row = group.sort("language").row(0, named=True)
        triples_data = _parse_json_list(row.get("triples_json"))
        triples = [KnowledgeGraphTriple.model_validate(item) for item in triples_data if isinstance(item, dict)]
        if not triples:
            continue
        aliases = _parse_json_list(row.get("aliases_json"))
        metadata = _parse_json_object(row.get("metadata_json"))
        metadata["aliases"] = [str(alias) for alias in aliases]
        paths.append(
            KnowledgeGraphPath(
                path_id=str(path_id),
                triples=triples,
                score=float(row.get("score", 0.0)),
                languages=[str(language) for language in group.get_column("language").to_list()],
                metadata=metadata,
            )
        )
    return paths


def load_graph_paths(project_root: Path) -> list[KnowledgeGraphPath]:
    cache_path = project_root / KG_CACHE_RELATIVE_PATH
    if not cache_path.exists():
        return get_seed_graph_paths()
    df = pl.read_parquet(cache_path)
    if not KG_CACHE_REQUIRED_COLUMNS.issubset(set(df.columns)):
        return get_seed_graph_paths()
    paths = deserialize_graph_paths(df)
    return paths or get_seed_graph_paths()


def write_graph_cache(paths: list[KnowledgeGraphPath], output_path: Path) -> None:
    df = serialize_graph_paths(paths)
    write_parquet(df, output_path)