from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.components.retriever import summarize_retrieved_chunks
from polyglot_grounded_qa.core.protocols import (
    AbstentionPolicy,
    Generator,
    Reranker,
    Retriever,
    Verifier,
)
from polyglot_grounded_qa.schemas.contracts import GroundedAnswer


@dataclass(slots=True)
class GroundedQAPipeline:
    retriever: Retriever
    reranker: Reranker
    generator: Generator
    verifier: Verifier
    abstention: AbstentionPolicy
    top_k_retrieve: int = 25
    top_k_rerank: int = 10

    def run(self, query: str, language: str) -> GroundedAnswer:
        chunks = self.retriever.retrieve(query=query, language=language, k=self.top_k_retrieve)
        chunks = self.reranker.rerank(query=query, chunks=chunks, k=self.top_k_rerank)
        retrieval_mode = "text"
        if hasattr(self.retriever, "retrieval_cfg"):
            retrieval_mode = getattr(self.retriever, "retrieval_cfg").mode
        retrieval_summary = summarize_retrieved_chunks(chunks=chunks, retrieval_mode=retrieval_mode)
        answer = self.generator.generate(query=query, chunks=chunks, language=language)
        answer = answer.model_copy(update={"metadata": {**retrieval_summary, **answer.metadata}})
        claims = self.verifier.verify(claims=answer.claims, chunks=chunks, language=language)
        answer = answer.model_copy(update={"claims": claims})
        abstained = self.abstention.should_abstain(answer)
        return answer.model_copy(update={"abstained": abstained})
