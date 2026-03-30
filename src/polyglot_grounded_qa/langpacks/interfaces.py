from __future__ import annotations

from typing import Protocol

from polyglot_grounded_qa.schemas.config import LanguageConfig


class TextNormalizer(Protocol):
    def normalize(self, text: str) -> str: ...


class TokenizerAdapter(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


class LanguagePack(Protocol):
    config: LanguageConfig

    def normalize(self, text: str) -> str: ...

    def tokenize(self, text: str) -> list[str]: ...
