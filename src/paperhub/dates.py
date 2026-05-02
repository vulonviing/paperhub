"""Resolve `RunRequest` into concrete `(start_date, end_date)` ranges."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta

from .localization import normalize_language
from .models import RunRequest


def resolve_range(req: RunRequest, today: date | None = None) -> tuple[date, date]:
    """Map a request's period descriptor to a concrete inclusive date range.

    Future end dates are clipped to `today` so we never query for tomorrow.
    """

    today = today or date.today()

    if req.period == "day":
        if req.year is None or req.month is None or req.day is None:
            raise ValueError("day period requires year, month, day")
        d = date(req.year, req.month, req.day)
        return d, d

    if req.period == "week":
        if req.year is None or req.week is None:
            raise ValueError("week period requires year and week")
        start = date.fromisocalendar(req.year, req.week, 1)
        end = start + timedelta(days=6)
        return start, min(end, today)

    if req.period == "month":
        if req.year is None or req.month is None:
            raise ValueError("month period requires year and month")
        start = date(req.year, req.month, 1)
        last = monthrange(req.year, req.month)[1]
        end = date(req.year, req.month, last)
        return start, min(end, today)

    if req.period == "year":
        if req.year is None:
            raise ValueError("year period requires year")
        return date(req.year, 1, 1), min(date(req.year, 12, 31), today)

    if req.period == "custom":
        if req.start is None or req.end is None:
            raise ValueError("custom period requires start and end")
        if req.end < req.start:
            raise ValueError("custom range end must be >= start")
        return req.start, min(req.end, today)

    raise ValueError(f"unknown period: {req.period}")


def iter_dates(start: date, end: date) -> list[date]:
    """Inclusive list of dates between `start` and `end`."""

    if end < start:
        return []
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def pretty_period(req: RunRequest, language: str | None = None) -> str:
    """Human-friendly label for a request, used in render headers."""

    lang = normalize_language(language or req.language)
    if req.period == "day" and req.year and req.month and req.day:
        return f"{req.day:02d}.{req.month:02d}.{req.year}"
    if req.period == "week" and req.year and req.week:
        return f"{req.year} W{req.week:02d}"
    if req.period == "month" and req.year and req.month:
        return f"{_MONTH_NAMES[lang][req.month]} {req.year}"
    if req.period == "year" and req.year:
        return str(req.year)
    if req.period == "custom" and req.start and req.end:
        if lang == "tr":
            return f"{req.start.isoformat()} - {req.end.isoformat()}"
        return f"{req.start.isoformat()} to {req.end.isoformat()}"
    return req.period


_MONTH_NAMES = {
    "en": {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    },
    "tr": {
        1: "Ocak",
        2: "Şubat",
        3: "Mart",
        4: "Nisan",
        5: "Mayıs",
        6: "Haziran",
        7: "Temmuz",
        8: "Ağustos",
        9: "Eylül",
        10: "Ekim",
        11: "Kasım",
        12: "Aralık",
    },
}
