from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OllamaAdapter:
    model: str
    host: str = "http://localhost:11434"

    def complete(self, prompt: str) -> str:
        try:
            import ollama
        except Exception:
            return ""

        try:
            client = ollama.Client(host=self.host)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            message = response.get("message", {}) if isinstance(response, dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            return str(content).strip()
        except Exception:
            return ""
