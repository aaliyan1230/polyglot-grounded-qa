from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OllamaAdapter:
    model: str
    host: str = "http://localhost:11434"

    def complete(self, prompt: str) -> str:
        _ = prompt
        return ""
