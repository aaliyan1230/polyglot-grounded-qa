from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VllmAdapter:
    model: str
    api_base: str

    def complete(self, prompt: str) -> str:
        _ = prompt
        return ""
