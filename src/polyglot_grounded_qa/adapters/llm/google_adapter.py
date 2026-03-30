from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GoogleGenAIAdapter:
    model: str

    def complete(self, prompt: str) -> str:
        _ = prompt
        return ""
