"""Extract entities from real QA data and build KG paths via Wikidata.

Reads the test/val/train JSONL files, extracts candidate entity terms from
questions and contexts, deduplicates, queries Wikidata, and merges results
with the existing seed KG cache.

Usage:
    uv run python scripts/build_real_kg_cache.py
    uv run python scripts/build_real_kg_cache.py --max-entities 200 --source test
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from polyglot_grounded_qa.adapters.kg import WikidataKGAdapter, KGConceptSeed
from polyglot_grounded_qa.core.kg_cache import KG_CACHE_RELATIVE_PATH, write_graph_cache
from polyglot_grounded_qa.core.seed_data import get_seed_graph_paths
from polyglot_grounded_qa.adapters.kg.wikidata_adapter import DEFAULT_LANGUAGES


# Simple stop words for entity extraction — avoid querying Wikidata for these.
_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would shall should can could may might must of in on at to for "
    "with by from as into through during before after above below between "
    "out off over under again further then once here there when where why "
    "how all each every both few more most other some such no nor not only "
    "own same so than too very what which who whom this that these those "
    "and but or if while because until although even though since about "
    "its it he she they we you i my his her their our your me him them us "
    "also just still already yet however also el la los las un una de en "
    "por para con que es como se le du des les une et est dans pour avec "
    "bir bir de ve ile icin ne nasil nedir olan".split()
)


def _extract_entity_candidates(text: str) -> list[str]:
    """Extract candidate entity terms from text using simple heuristics."""
    candidates: list[str] = []

    # 1. Capitalized multi-word sequences (proper nouns / named entities)
    for match in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        term = match.group().strip()
        if len(term) > 3 and len(term.split()) <= 4:
            candidates.append(term)

    # 2. Single capitalized words that aren't sentence starters
    words = text.split()
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w]", "", word)
        if not clean or len(clean) < 3:
            continue
        if clean[0].isupper() and clean.lower() not in _STOP_WORDS:
            # Skip if it's a sentence starter (after period or first word)
            if i > 0 and not words[i - 1].endswith((".","?","!")):
                candidates.append(clean)

    # 3. Quoted terms
    for match in re.finditer(r'"([^"]+)"', text):
        term = match.group(1).strip()
        if 2 < len(term) < 50:
            candidates.append(term)

    return candidates


def _extract_entities_from_row(row: dict) -> list[str]:
    """Extract entity candidates from a single QA row."""
    entities: list[str] = []
    query = row.get("query", "")
    entities.extend(_extract_entity_candidates(query))

    # Also extract from first chunk context (limited to first 500 chars for speed)
    chunks = row.get("retrieved_chunks", [])
    if chunks:
        context = chunks[0].get("text", "")[:500]
        entities.extend(_extract_entity_candidates(context))

    return entities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KG cache from real QA entities via Wikidata."
    )
    parser.add_argument(
        "--source",
        choices=["test", "val", "train", "all"],
        default="test",
        help="Which split to extract entities from.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=100,
        help="Maximum unique entities to query Wikidata for.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between Wikidata API calls (seconds).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Timeout for each Wikidata API call.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "benchmarks" / "finetune"

    # Determine which files to read
    if args.source == "all":
        files = ["test.jsonl", "val.jsonl", "train.jsonl"]
    else:
        files = [f"{args.source}.jsonl"]

    # Load rows
    rows: list[dict] = []
    for fname in files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  Skipping {fpath} (not found)")
            continue
        with fpath.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"Loaded {len(rows)} rows from {files}")

    # Extract and count entity candidates
    entity_counts: Counter[str] = Counter()
    for row in rows:
        # Only extract from answerable rows with real context
        if row.get("label_type") != "answerable":
            continue
        candidates = _extract_entities_from_row(row)
        for c in candidates:
            normalized = c.strip()
            if len(normalized) >= 3:
                entity_counts[normalized] += 1

    # Deduplicate case-insensitively, keeping highest-count variant
    seen_lower: dict[str, tuple[str, int]] = {}
    for entity, count in entity_counts.items():
        key = entity.lower()
        if key not in seen_lower or count > seen_lower[key][1]:
            seen_lower[key] = (entity, count)

    # Sort by frequency and take top N
    sorted_entities = sorted(seen_lower.values(), key=lambda x: -x[1])
    selected = sorted_entities[: args.max_entities]

    print(f"Extracted {len(seen_lower)} unique entities, selected top {len(selected)}")
    print("Top 20 entities:")
    for entity, count in selected[:20]:
        print(f"  {entity}: {count} occurrences")

    # Build concept seeds from entities
    seeds: list[KGConceptSeed] = []
    for entity, _count in selected:
        seeds.append(
            KGConceptSeed(
                concept_id=f"real-entity:{entity.lower().replace(' ', '-')}",
                search_query=entity,
                aliases=[entity, entity.lower()],
                languages=DEFAULT_LANGUAGES,
            )
        )

    # Fetch from Wikidata
    adapter = WikidataKGAdapter(timeout_seconds=args.timeout)
    all_paths = []
    success = 0
    fail = 0
    for i, seed in enumerate(seeds):
        try:
            search_result = adapter._search_entity(seed.search_query)
            if not search_result:
                fail += 1
                continue
            paths = adapter._extract_paths(seed=seed, search_result=search_result)
            all_paths.extend(paths)
            success += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(seeds)} entities, {len(all_paths)} paths so far")
        except Exception as e:
            fail += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(seeds)} ({fail} failures)")
        time.sleep(args.delay)

    print(f"\nWikidata results: {success} found, {fail} failed, {len(all_paths)} total paths")

    # Merge with seed paths
    seed_paths = get_seed_graph_paths()
    combined: dict[str, object] = {}
    for p in seed_paths:
        combined[p.path_id] = p
    for p in all_paths:
        combined[p.path_id] = p

    output_path = project_root / KG_CACHE_RELATIVE_PATH
    write_graph_cache(list(combined.values()), output_path)
    print(
        f"Wrote {len(combined)} KG paths to {output_path} "
        f"({len(all_paths)} real, {len(seed_paths)} seed)"
    )


if __name__ == "__main__":
    main()
