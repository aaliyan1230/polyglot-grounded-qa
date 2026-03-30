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
