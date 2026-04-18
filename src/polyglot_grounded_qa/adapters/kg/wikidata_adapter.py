from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polyglot_grounded_qa.schemas.contracts import KnowledgeGraphPath, KnowledgeGraphTriple

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
DEFAULT_LANGUAGES = ["base", "es", "fr", "tr"]
PROPERTY_LABELS = {
    "P31": "instance of",
    "P279": "subclass of",
    "P361": "part of",
    "P527": "has part",
    "P1269": "facet of",
}
PROPERTY_PRIORITY = ["P279", "P31", "P361", "P1269", "P527"]


@dataclass(slots=True)
class KGConceptSeed:
    concept_id: str
    search_query: str
    aliases: list[str]
    languages: list[str]


def get_default_concept_seeds() -> list[KGConceptSeed]:
    return [
        KGConceptSeed(
            concept_id="question-answering",
            search_query="question answering",
            aliases=[
                "grounded qa",
                "question answering",
                "qa",
                "respuesta a preguntas",
                "pregunta respuesta",
                "pregunta y respuesta",
                "question reponse",
                "soru cevaplama",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
        KGConceptSeed(
            concept_id="information-retrieval",
            search_query="information retrieval",
            aliases=[
                "retrieval",
                "text retrieval",
                "information retrieval",
                "recuperacion",
                "recuperacion de informacion",
                "recherche d information",
                "bilgi erisimi",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
        KGConceptSeed(
            concept_id="knowledge-graph",
            search_query="knowledge graph",
            aliases=[
                "knowledge graph",
                "kg",
                "graph support",
                "grafo de conocimiento",
                "grafico de conocimiento",
                "graphe de connaissances",
                "bilgi grafigi",
                "graf destekli alma",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
        KGConceptSeed(
            concept_id="citation",
            search_query="citation",
            aliases=[
                "citation",
                "citations",
                "cita",
                "citas",
                "citations requises",
                "atif",
                "atif gerekir",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
        KGConceptSeed(
            concept_id="evidence",
            search_query="evidence",
            aliases=[
                "evidence",
                "support",
                "weak evidence",
                "evidencia",
                "evidencia debil",
                "preuve",
                "kanit",
                "zayif kanit",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
        KGConceptSeed(
            concept_id="multilingualism",
            search_query="multilingualism",
            aliases=[
                "language packs",
                "locale inheritance",
                "multilingual",
                "locale",
                "espanol de mexico",
                "multilingue",
                "cok dilli",
                "dil paketleri mirasi",
            ],
            languages=DEFAULT_LANGUAGES,
        ),
    ]


class WikidataKGAdapter:
    def __init__(self, timeout_seconds: float = 8.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._label_cache: dict[str, str] = {}

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_url = url
        if params:
            request_url = f"{url}?{urlencode(params, doseq=True)}"
        request = Request(
            request_url,
            headers={"User-Agent": "polyglot-grounded-qa/0.1 (+local research cache builder)"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _search_entity(self, query: str) -> dict[str, Any] | None:
        payload = self._get_json(
            WIKIDATA_API_URL,
            {
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "type": "item",
                "limit": 1,
                "search": query,
            },
        )
        results = payload.get("search", [])
        if not results:
            return None
        return results[0]

    def _get_entity_data(self, entity_id: str) -> dict[str, Any]:
        payload = self._get_json(WIKIDATA_ENTITY_DATA_URL.format(entity_id=entity_id))
        entities = payload.get("entities", {})
        return entities.get(entity_id, {})

    def _get_entity_labels(self, entity_ids: list[str]) -> dict[str, str]:
        unresolved = [entity_id for entity_id in entity_ids if entity_id not in self._label_cache]
        if unresolved:
            payload = self._get_json(
                WIKIDATA_API_URL,
                {
                    "action": "wbgetentities",
                    "format": "json",
                    "languages": "en",
                    "props": "labels",
                    "ids": "|".join(unresolved),
                },
            )
            for entity_id, entity in payload.get("entities", {}).items():
                label = (
                    entity.get("labels", {}).get("en", {}).get("value")
                    or entity_id
                )
                self._label_cache[entity_id] = label
        return {entity_id: self._label_cache.get(entity_id, entity_id) for entity_id in entity_ids}

    def _collect_aliases(self, seed: KGConceptSeed, entity: dict[str, Any]) -> list[str]:
        aliases = list(seed.aliases)
        label = entity.get("labels", {}).get("en", {}).get("value")
        if label:
            aliases.append(str(label))
        for values in entity.get("aliases", {}).values():
            for item in values:
                value = item.get("value")
                if value:
                    aliases.append(str(value))
        deduped: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            normalized = alias.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(alias)
        return deduped

    def _extract_paths(self, seed: KGConceptSeed, search_result: dict[str, Any]) -> list[KnowledgeGraphPath]:
        entity_id = str(search_result.get("id", ""))
        if not entity_id:
            return []
        entity = self._get_entity_data(entity_id)
        if not entity:
            return []

        subject_label = (
            entity.get("labels", {}).get("en", {}).get("value")
            or search_result.get("label")
            or seed.search_query
        )
        claims = entity.get("claims", {})
        selected_claims: list[tuple[str, str]] = []
        for property_id in PROPERTY_PRIORITY:
            for claim in claims.get(property_id, []):
                mainsnak = claim.get("mainsnak", {})
                if mainsnak.get("snaktype") != "value":
                    continue
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value")
                if isinstance(value, dict) and isinstance(value.get("id"), str):
                    selected_claims.append((property_id, value["id"]))
                if len(selected_claims) >= 3:
                    break
            if len(selected_claims) >= 3:
                break

        if not selected_claims:
            return []

        object_labels = self._get_entity_labels([object_id for _, object_id in selected_claims])
        aliases = self._collect_aliases(seed, entity)
        paths: list[KnowledgeGraphPath] = []
        for index, (property_id, object_id) in enumerate(selected_claims):
            object_label = object_labels.get(object_id, object_id)
            score = round(0.72 - (index * 0.04), 4)
            paths.append(
                KnowledgeGraphPath(
                    path_id=f"wikidata::{entity_id}::{property_id}::{object_id}",
                    triples=[
                        KnowledgeGraphTriple(
                            subject=str(subject_label),
                            relation=PROPERTY_LABELS.get(property_id, property_id),
                            object=str(object_label),
                        )
                    ],
                    score=score,
                    languages=seed.languages,
                    metadata={
                        "aliases": aliases,
                        "source": "wikidata",
                        "entity_id": entity_id,
                        "property_id": property_id,
                        "object_id": object_id,
                        "search_query": seed.search_query,
                    },
                )
            )
        return paths

    def fetch_paths(self, concept_seeds: list[KGConceptSeed] | None = None) -> list[KnowledgeGraphPath]:
        seeds = concept_seeds or get_default_concept_seeds()
        paths: list[KnowledgeGraphPath] = []
        for seed in seeds:
            try:
                search_result = self._search_entity(seed.search_query)
                if not search_result:
                    continue
                paths.extend(self._extract_paths(seed=seed, search_result=search_result))
            except Exception:
                continue
        return paths