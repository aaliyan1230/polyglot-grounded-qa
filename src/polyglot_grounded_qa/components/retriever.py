from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.config import RetrievalConfig
from polyglot_grounded_qa.schemas.contracts import KnowledgeGraphPath, RetrievedChunk


def _tokenize(text: str) -> set[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return set(tokens)


def _supports_language(available_languages: list[str], language: str) -> bool:
    return not available_languages or language in available_languages


def summarize_retrieved_chunks(chunks: list[RetrievedChunk], retrieval_mode: str) -> dict[str, int | float | str]:
    text_evidence_count = 0
    graph_evidence_count = 0
    graph_support_score = 0.0
    graph_quality_score = 0.0
    hybrid_policy = "naive"
    routing_decision = "static"
    top_evidence_type = "none"
    top_chunk_id = ""

    if chunks:
        top_evidence_type = str(chunks[0].metadata.get("evidence_type", "text"))
        top_chunk_id = chunks[0].chunk_id

    for chunk in chunks:
        hybrid_policy = str(chunk.metadata.get("hybrid_policy", hybrid_policy))
        routing_decision = str(chunk.metadata.get("routing_decision", routing_decision))
        evidence_type = chunk.metadata.get("evidence_type", "text")
        if evidence_type == "graph":
            graph_evidence_count += 1
            graph_support_score = max(
                graph_support_score,
                float(chunk.metadata.get("graph_path_score", chunk.score)),
            )
            graph_quality_score = max(
                graph_quality_score,
                float(chunk.metadata.get("graph_quality_score", 0.0)),
            )
            continue
        text_evidence_count += 1

    return {
        "retrieval_mode": retrieval_mode,
        "hybrid_policy": hybrid_policy,
        "routing_decision": routing_decision,
        "top_evidence_type": top_evidence_type,
        "top_chunk_id": top_chunk_id,
        "text_evidence_count": text_evidence_count,
        "graph_evidence_count": graph_evidence_count,
        "graph_support_score": round(graph_support_score, 4),
        "graph_quality_score": round(graph_quality_score, 4),
    }


@dataclass(slots=True)
class GraphQueryDiagnostics:
    query: str
    language: str
    linked_entity_count: int
    candidate_path_count: int
    returned_path_count: int
    max_path_score: float
    failure_bucket: str


@dataclass(slots=True)
class BaselineRetriever:
    """Minimal retriever stub for notebook and test plumbing."""

    corpus: list[RetrievedChunk]

    def retrieve(self, query: str, language: str, k: int) -> list[RetrievedChunk]:
        query_terms = _tokenize(query)
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in self.corpus:
            languages = list(chunk.metadata.get("languages", []))
            if not _supports_language(languages, language):
                continue
            overlap = len(query_terms.intersection(_tokenize(chunk.text)))
            score = float(overlap) + chunk.score
            metadata = {
                **chunk.metadata,
                "evidence_type": chunk.metadata.get("evidence_type", "text"),
                "retrieval_language": language,
            }
            scored.append(
                (
                    score,
                    chunk.model_copy(update={"score": score, "metadata": metadata}),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]


@dataclass(slots=True)
class SeedKnowledgeGraphRetriever:
    paths: list[KnowledgeGraphPath]
    min_path_score: float
    entity_link_min_score: float

    def _rank_paths(
        self, query: str, language: str
    ) -> list[tuple[float, float, int, KnowledgeGraphPath]]:
        query_terms = _tokenize(query)
        query_text = query.lower()
        ranked: list[tuple[float, float, int, KnowledgeGraphPath]] = []

        for path in self.paths:
            if not _supports_language(path.languages, language):
                continue

            alias_hits = 0
            alias_terms: set[str] = set()
            for alias in path.metadata.get("aliases", []):
                alias_text = str(alias).lower()
                alias_terms.update(_tokenize(alias_text))
                if alias_text and alias_text in query_text:
                    alias_hits += 1

            overlap = len(query_terms.intersection(alias_terms.union(_tokenize(path.render_text()))))
            normalized_overlap = 0.0 if not query_terms else overlap / len(query_terms)
            link_score = max(alias_hits * 0.35, normalized_overlap)
            final_score = path.score + link_score
            ranked.append((final_score, link_score, alias_hits, path))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def analyze_query(self, query: str, language: str, k: int) -> GraphQueryDiagnostics:
        ranked = self._rank_paths(query=query, language=language)
        if not ranked:
            return GraphQueryDiagnostics(
                query=query,
                language=language,
                linked_entity_count=0,
                candidate_path_count=0,
                returned_path_count=0,
                max_path_score=0.0,
                failure_bucket="unsupported-language",
            )

        linked_entity_count = sum(1 for _, link_score, alias_hits, _ in ranked if alias_hits > 0 or link_score >= self.entity_link_min_score)
        retrieved = [
            item
            for item in ranked
            if item[0] >= self.min_path_score and item[1] >= self.entity_link_min_score
        ][:k]
        max_path_score = round(max(score for score, _, _, _ in ranked), 4)
        max_link_score = max(link_score for _, link_score, _, _ in ranked)

        if linked_entity_count == 0:
            failure_bucket = "no-link"
        elif not retrieved:
            failure_bucket = "no-path"
        elif max_link_score < self.entity_link_min_score:
            failure_bucket = "low-confidence-link"
        else:
            failure_bucket = "supported"

        return GraphQueryDiagnostics(
            query=query,
            language=language,
            linked_entity_count=linked_entity_count,
            candidate_path_count=len(ranked),
            returned_path_count=len(retrieved),
            max_path_score=max_path_score,
            failure_bucket=failure_bucket,
        )

    def retrieve(self, query: str, language: str, k: int) -> list[RetrievedChunk]:
        ranked = self._rank_paths(query=query, language=language)
        retrieved: list[RetrievedChunk] = []
        for score, link_score, alias_hits, path in ranked:
            if score < self.min_path_score or link_score < self.entity_link_min_score:
                continue
            metadata = {
                **path.metadata,
                "evidence_type": "graph",
                "languages": path.languages,
                "graph_path_score": round(score, 4),
                "graph_link_score": round(link_score, 4),
                "linked_entity_hits": alias_hits,
                "path_length": len(path.triples),
                "retrieval_language": language,
            }
            retrieved.append(
                path.to_retrieved_chunk(score=round(score, 4)).model_copy(
                    update={"metadata": metadata}
                )
            )
            if len(retrieved) >= k:
                break
        return retrieved


@dataclass(slots=True)
class HybridRetriever:
    text_retriever: BaselineRetriever
    graph_retriever: SeedKnowledgeGraphRetriever
    retrieval_cfg: RetrievalConfig

    def _graph_quality_score(self, chunk: RetrievedChunk) -> float:
        link_score = float(chunk.metadata.get("graph_link_score", 0.0))
        path_score = min(float(chunk.metadata.get("graph_path_score", chunk.score)) / 1.5, 1.0)
        path_length = int(chunk.metadata.get("path_length", 1))
        source = str(chunk.metadata.get("source", "seed"))
        source_score = 1.0 if source == "wikidata" else 0.85
        length_score = max(0.5, 1.0 - (0.15 * max(path_length - 1, 0)))
        return round(
            (0.45 * link_score) + (0.3 * path_score) + (0.15 * source_score) + (0.1 * length_score),
            4,
        )

    def _apply_graph_policy(self, graph_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        enriched: list[RetrievedChunk] = []
        for chunk in graph_chunks:
            quality_score = self._graph_quality_score(chunk)
            enriched.append(
                chunk.model_copy(
                    update={
                        "metadata": {
                            **chunk.metadata,
                            "graph_quality_score": quality_score,
                        }
                    }
                )
            )

        if self.retrieval_cfg.hybrid_policy == "filtered":
            filtered = [
                chunk
                for chunk in enriched
                if float(chunk.metadata.get("graph_quality_score", 0.0))
                >= self.retrieval_cfg.graph_min_quality_score
            ]
            return filtered

        if self.retrieval_cfg.hybrid_policy == "routed":
            filtered = [
                chunk
                for chunk in enriched
                if float(chunk.metadata.get("graph_quality_score", 0.0))
                >= self.retrieval_cfg.graph_min_quality_score
            ]
            return filtered or enriched[:1]

        return enriched

    def _route_query(self, query: str) -> str:
        normalized_query = query.strip().lower()
        tokens = _tokenize(normalized_query)
        graph_first_prefixes = (
            "what is",
            "who is",
            "when is",
            "where is",
            "que es",
            "qu est",
            "nedir",
        )
        text_first_markers = (
            "why",
            "how",
            "explain",
            "compare",
            "por que",
            "comment",
            "neden",
            "nasil",
        )

        if any(normalized_query.startswith(prefix) for prefix in graph_first_prefixes) and len(tokens) <= 8:
            return "graph-first"
        if any(marker in normalized_query for marker in text_first_markers):
            return "text-first"
        if len(tokens) <= 6:
            return "graph-first"
        return "balanced"

    def _annotate_chunks(
        self, chunks: list[RetrievedChunk], routing_decision: str
    ) -> list[RetrievedChunk]:
        return [
            chunk.model_copy(
                update={
                    "metadata": {
                        **chunk.metadata,
                        "hybrid_policy": self.retrieval_cfg.hybrid_policy,
                        "routing_decision": routing_decision,
                    }
                }
            )
            for chunk in chunks
        ]

    def _apply_route_ordering(
        self, chunks: list[RetrievedChunk], routing_decision: str
    ) -> list[RetrievedChunk]:
        if routing_decision not in {"graph-first", "text-first"}:
            return chunks

        preferred_type = "graph" if routing_decision == "graph-first" else "text"
        preferred = [
            chunk for chunk in chunks if str(chunk.metadata.get("evidence_type", "text")) == preferred_type
        ]
        remaining = [
            chunk for chunk in chunks if str(chunk.metadata.get("evidence_type", "text")) != preferred_type
        ]
        return preferred + remaining

    def _assemble_routed_chunks(
        self,
        text_chunks: list[RetrievedChunk],
        graph_chunks: list[RetrievedChunk],
        routing_decision: str,
        k: int,
    ) -> list[RetrievedChunk]:
        if routing_decision == "graph-first":
            ordered = graph_chunks[: min(len(graph_chunks), max(2, k - 1))] + text_chunks[:1]
            return ordered[:k]
        if routing_decision == "text-first":
            ordered = text_chunks[: min(len(text_chunks), max(2, k - 1))] + graph_chunks[:1]
            return ordered[:k]
        fused = self._fuse_ranked_lists(
            text_chunks=text_chunks,
            graph_chunks=graph_chunks,
            k=k,
            graph_weight=self.retrieval_cfg.graph_weight,
            text_weight=self.retrieval_cfg.text_weight,
        )
        return self._apply_route_ordering(fused, routing_decision=routing_decision)

    def _fuse_ranked_lists(
        self,
        text_chunks: list[RetrievedChunk],
        graph_chunks: list[RetrievedChunk],
        k: int,
        graph_weight: float,
        text_weight: float,
    ) -> list[RetrievedChunk]:
        weighted_scores: dict[tuple[str, str], tuple[float, RetrievedChunk]] = {}

        for weight, chunks in (
            (text_weight, text_chunks),
            (graph_weight, graph_chunks),
        ):
            for rank, chunk in enumerate(chunks, start=1):
                fused_score = chunk.score + (weight / (60 + rank))
                key = (chunk.doc_id, chunk.chunk_id)
                previous = weighted_scores.get(key)
                updated_chunk = chunk.model_copy(update={"score": round(fused_score, 4)})
                if previous is None or fused_score > previous[0]:
                    weighted_scores[key] = (fused_score, updated_chunk)

        ordered = sorted(weighted_scores.values(), key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ordered[:k]]

    def retrieve(self, query: str, language: str, k: int) -> list[RetrievedChunk]:
        if self.retrieval_cfg.mode == "graph":
            graph_chunks = self.graph_retriever.retrieve(
                query=query,
                language=language,
                k=min(k, self.retrieval_cfg.graph_top_k),
            )
            graph_chunks = self._apply_graph_policy(graph_chunks)
            return self._annotate_chunks(graph_chunks, routing_decision="graph-only")

        text_chunks = self.text_retriever.retrieve(
            query=query,
            language=language,
            k=min(k, self.retrieval_cfg.top_k_dense),
        )
        if self.retrieval_cfg.mode == "text":
            return self._annotate_chunks(text_chunks, routing_decision="text-only")

        graph_chunks = self.graph_retriever.retrieve(
            query=query,
            language=language,
            k=min(k, self.retrieval_cfg.graph_top_k),
        )
        graph_chunks = self._apply_graph_policy(graph_chunks)

        routing_decision = "balanced"
        text_weight = self.retrieval_cfg.text_weight
        graph_weight = self.retrieval_cfg.graph_weight
        if self.retrieval_cfg.hybrid_policy == "routed":
            routing_decision = self._route_query(query)
            if routing_decision == "graph-first":
                graph_chunks = graph_chunks[: min(len(graph_chunks), max(2, self.retrieval_cfg.graph_top_k))]
                text_chunks = text_chunks[:1]
            elif routing_decision == "text-first":
                graph_chunks = graph_chunks[:1]
                text_chunks = text_chunks[: min(len(text_chunks), max(3, self.retrieval_cfg.top_k_dense // 2))]

        if self.retrieval_cfg.hybrid_policy == "routed":
            fused = self._assemble_routed_chunks(
                text_chunks=text_chunks,
                graph_chunks=graph_chunks,
                routing_decision=routing_decision,
                k=k,
            )
        else:
            fused = self._fuse_ranked_lists(
                text_chunks=text_chunks,
                graph_chunks=graph_chunks,
                k=k,
                graph_weight=graph_weight,
                text_weight=text_weight,
            )
            fused = self._apply_route_ordering(fused, routing_decision=routing_decision)
        return self._annotate_chunks(fused, routing_decision=routing_decision)
