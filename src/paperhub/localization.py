"""Language helpers for PaperHub output."""

from __future__ import annotations

from typing import Literal

OutputLanguage = Literal["en", "tr"]


def normalize_language(language: str | None = None) -> OutputLanguage:
    """Normalize user-facing language values to PaperHub's supported codes."""

    value = (language or "en").strip().lower().replace("_", "-")
    if value in {"en", "eng", "english"}:
        return "en"
    if value in {"tr", "turkish", "turkce", "türkçe"}:
        return "tr"
    raise ValueError("language must be 'en' or 'tr'")
