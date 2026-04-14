from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.schemas.config import ThresholdConfig
from polyglot_grounded_qa.schemas.contracts import GroundedAnswer


@dataclass(slots=True)
class ThresholdAbstentionPolicy:
    thresholds: ThresholdConfig

    def should_abstain(self, answer: GroundedAnswer) -> bool:
        if not answer.citations:
            return True
        support_values = [
            claim.confidence
            for claim in answer.claims
            if claim.supported and claim.confidence is not None
        ]
        if not support_values:
            return False
        avg_support = sum(support_values) / len(support_values)
        return avg_support < self.thresholds.abstain_min_support


@dataclass(slots=True)
class GraphAwareAbstentionPolicy:
    thresholds: ThresholdConfig

    def should_abstain(self, answer: GroundedAnswer) -> bool:
        fallback = ThresholdAbstentionPolicy(self.thresholds)
        if fallback.should_abstain(answer):
            return True

        retrieval_mode = str(answer.metadata.get("retrieval_mode", "text"))
        if retrieval_mode == "text":
            return False

        graph_evidence_count = int(answer.metadata.get("graph_evidence_count", 0))
        graph_support_score = float(answer.metadata.get("graph_support_score", 0.0))
        if graph_evidence_count < self.thresholds.graph_min_path_count:
            return True
        if graph_support_score < self.thresholds.graph_min_path_score:
            return True

        if retrieval_mode == "hybrid":
            text_evidence_count = int(answer.metadata.get("text_evidence_count", 0))
            if text_evidence_count < self.thresholds.hybrid_min_text_evidence:
                return True

        return False
