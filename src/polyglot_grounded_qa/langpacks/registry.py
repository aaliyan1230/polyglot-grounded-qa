from __future__ import annotations

from dataclasses import dataclass

from polyglot_grounded_qa.langpacks.interfaces import LanguagePack
from polyglot_grounded_qa.schemas.config import LanguageConfig


class DefaultNormalizer:
    def normalize(self, text: str) -> str:
        return " ".join(text.strip().split())


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


@dataclass(slots=True)
class BasicLanguagePack:
    config: LanguageConfig

    def normalize(self, text: str) -> str:
        return DefaultNormalizer().normalize(text)

    def tokenize(self, text: str) -> list[str]:
        return WhitespaceTokenizer().tokenize(text)


def build_pack(config: LanguageConfig) -> LanguagePack:
    return BasicLanguagePack(config=config)
