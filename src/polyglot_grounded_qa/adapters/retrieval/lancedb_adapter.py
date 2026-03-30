from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LanceDBAdapter:
    """Optional embedded vector backend for larger local experiments."""

    uri: Path

    def connect(self) -> None:
        _ = self.uri
