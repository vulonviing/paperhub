"""Parse English and Turkish natural-language queries into `RunRequest`.

The parser is regex-first by design: it must be deterministic, dependency-free,
and import-safe with no API keys.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .models import RunRequest

TR_MONTHS: dict[str, int] = {
    "ocak": 1,
    "şubat": 2,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayıs": 5,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "ağustos": 8,
    "agustos": 8,
    "eylül": 9,
    "eylul": 9,
    "ekim": 10,
    "kasım": 11,
    "kasim": 11,
    "aralık": 12,
    "aralik": 12,
}

EN_MONTHS: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

ALL_MONTHS: dict[str, int] = {**TR_MONTHS, **EN_MONTHS}

_MONTH_PATTERN = "|".join(sorted(ALL_MONTHS.keys(), key=len, reverse=True))


class QueryParseError(ValueError):
    """Raised when a query cannot be interpreted."""


def parse(text: str, today: date | None = None) -> RunRequest:
    """Parse a natural-language query into a `RunRequest`.

    Recognized forms:
    - `may 2026 top 10 papers`
    - `mayıs 2026 top 10 paper`
    - `2025 year top 20`
    - `2025 yılı top 20`
    - `bugün`, `dün`, `bu hafta`, `geçen hafta`, `bu ay`, `geçen ay`, `bu yıl`
    - `today`, `yesterday`, `this week`, `last month`, `this year`
    - `2026-05-15`
    - `2026-05-01 to 2026-05-31`
    """

    if not text or not text.strip():
        raise QueryParseError("empty query")

    today = today or date.today()
    raw = text.strip()
    t = raw.lower()
    top_n = _parse_top_n(t)

    custom = _try_custom_range(t)
    if custom is not None:
        start, end = custom
        return RunRequest(period="custom", start=start, end=end, top_n=top_n)

    iso = _try_iso_date(t)
    if iso is not None:
        return RunRequest(period="day", year=iso.year, month=iso.month, day=iso.day, top_n=top_n)

    relative = _try_relative(t, today)
    if relative is not None:
        return relative.model_copy(update={"top_n": top_n})

    month_year = _try_month_year(t)
    if month_year is not None:
        year, month = month_year
        return RunRequest(period="month", year=year, month=month, top_n=top_n)

    year_only = _try_year_only(t)
    if year_only is not None:
        return RunRequest(period="year", year=year_only, top_n=top_n)

    raise QueryParseError(f"could not parse query: {raw!r}")


def _parse_top_n(t: str) -> int:
    m = re.search(r"top\s*(\d+)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\ben\s+i?yi\s*(\d+)\b", t)
    if m:
        return int(m.group(1))
    return 10


def _try_iso_date(t: str) -> date | None:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _try_custom_range(t: str) -> tuple[date, date] | None:
    pattern = (
        r"\b(\d{4})-(\d{2})-(\d{2})\b"
        r"\s*(?:to|->|-|–|ile|—|until|ile)\s*"
        r"\b(\d{4})-(\d{2})-(\d{2})\b"
    )
    m = re.search(pattern, t)
    if not m:
        return None
    try:
        start = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        end = date(int(m.group(4)), int(m.group(5)), int(m.group(6)))
    except ValueError:
        return None
    if end < start:
        return None
    return start, end


def _try_relative(t: str, today: date) -> RunRequest | None:
    if re.search(r"\bbug[uü]n|today\b", t):
        return RunRequest(period="day", year=today.year, month=today.month, day=today.day)
    if re.search(r"\bd[uü]n|yesterday\b", t):
        d = today - timedelta(days=1)
        return RunRequest(period="day", year=d.year, month=d.month, day=d.day)

    if re.search(r"\b(bu\s*hafta|this\s*week)\b", t):
        iso = today.isocalendar()
        return RunRequest(period="week", year=iso.year, week=iso.week)
    if re.search(r"\b(ge[cç]en\s*hafta|last\s*week|previous\s*week)\b", t):
        d = today - timedelta(days=7)
        iso = d.isocalendar()
        return RunRequest(period="week", year=iso.year, week=iso.week)

    if re.search(r"\b(bu\s*ay|this\s*month)\b", t):
        return RunRequest(period="month", year=today.year, month=today.month)
    if re.search(r"\b(ge[cç]en\s*ay|last\s*month|previous\s*month)\b", t):
        if today.month == 1:
            return RunRequest(period="month", year=today.year - 1, month=12)
        return RunRequest(period="month", year=today.year, month=today.month - 1)

    if re.search(r"\b(bu\s*y[iı]l|this\s*year)\b", t):
        return RunRequest(period="year", year=today.year)
    if re.search(r"\b(ge[cç]en\s*y[iı]l|last\s*year|previous\s*year)\b", t):
        return RunRequest(period="year", year=today.year - 1)

    return None


def _try_month_year(t: str) -> tuple[int, int] | None:
    pattern = rf"\b({_MONTH_PATTERN})\s+(\d{{4}})\b"
    m = re.search(pattern, t)
    if m:
        return int(m.group(2)), ALL_MONTHS[m.group(1)]

    pattern_rev = rf"\b(\d{{4}})\s+({_MONTH_PATTERN})\b"
    m = re.search(pattern_rev, t)
    if m:
        return int(m.group(1)), ALL_MONTHS[m.group(2)]

    return None


def _try_year_only(t: str) -> int | None:
    m = re.search(r"\b(\d{4})\b\s*(?:y[iı]l[ıi]?|year)?", t)
    if m:
        year = int(m.group(1))
        if 2000 <= year <= 2100:
            return year
    return None
