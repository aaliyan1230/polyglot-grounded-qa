from __future__ import annotations

from polyglot_grounded_qa.adapters.kg import KGConceptSeed, WikidataKGAdapter


def test_wikidata_adapter_fetches_paths_from_mocked_api(monkeypatch) -> None:
    adapter = WikidataKGAdapter()

    monkeypatch.setattr(
        adapter,
        "_search_entity",
        lambda query: {"id": "Q1", "label": "question answering"},
    )
    monkeypatch.setattr(
        adapter,
        "_get_entity_data",
        lambda entity_id: {
            "labels": {"en": {"value": "question answering"}},
            "aliases": {"en": [{"value": "qa"}]},
            "claims": {
                "P279": [
                    {
                        "mainsnak": {
                            "snaktype": "value",
                            "datavalue": {"value": {"id": "Q2"}},
                        }
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(adapter, "_get_entity_labels", lambda ids: {"Q2": "natural language processing"})

    paths = adapter.fetch_paths(
        [
            KGConceptSeed(
                concept_id="question-answering",
                search_query="question answering",
                aliases=["grounded qa", "question answering"],
                languages=["base", "es"],
            )
        ]
    )

    assert len(paths) == 1
    assert paths[0].metadata.get("source") == "wikidata"
    assert paths[0].triples[0].relation == "subclass of"
    assert "grounded qa" in paths[0].metadata.get("aliases", [])