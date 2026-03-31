from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class GoogleGenAIAdapter:
    model: str

    def complete(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return ""

        try:
            from google import genai
        except Exception:
            return ""

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=self.model, contents=prompt)
            text = getattr(response, "text", None)
            return text or ""
        except Exception:
            return ""
