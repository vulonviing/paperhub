"""Fetcher protocol shared by JSON-API and HTML fallback implementations."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from ..models import PaperMeta


class Fetcher(Protocol):
    """A source of `PaperMeta` records for a single calendar date."""

    async def papers_for_date(self, d: date, limit: int = 100) -> list[PaperMeta]: ...
