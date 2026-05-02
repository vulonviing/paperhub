"""Tests for the Turkish/English NL parser."""

from __future__ import annotations

from datetime import date

import pytest

from paperhub.nl_parser import QueryParseError, parse


def test_month_year_with_top_n() -> None:
    req = parse("mayıs 2026 top 10 paper")
    assert req.period == "month"
    assert req.year == 2026
    assert req.month == 5
    assert req.top_n == 10


def test_month_year_ascii() -> None:
    req = parse("mayis 2026 top 5")
    assert req.period == "month"
    assert req.month == 5
    assert req.top_n == 5


def test_english_month() -> None:
    req = parse("may 2026 top 7")
    assert req.month == 5
    assert req.top_n == 7


def test_year_only() -> None:
    req = parse("2025 yılı top 20")
    assert req.period == "year"
    assert req.year == 2025
    assert req.top_n == 20


def test_today() -> None:
    req = parse("bugün top 3", today=date(2026, 5, 1))
    assert req.period == "day"
    assert req.day == 1
    assert req.top_n == 3


def test_yesterday_ascii() -> None:
    req = parse("dun top 2", today=date(2026, 5, 2))
    assert req.period == "day"
    assert req.day == 1


def test_this_week() -> None:
    req = parse("bu hafta top 5", today=date(2026, 5, 6))
    assert req.period == "week"
    assert req.year is not None and req.week is not None


def test_last_month_handles_january() -> None:
    req = parse("geçen ay", today=date(2026, 1, 15))
    assert req.period == "month"
    assert req.year == 2025
    assert req.month == 12


def test_iso_date_single() -> None:
    req = parse("2026-05-15 top 4")
    assert req.period == "day"
    assert req.day == 15
    assert req.top_n == 4


def test_custom_range() -> None:
    req = parse("2026-05-01 to 2026-05-31 top 2")
    assert req.period == "custom"
    assert req.start == date(2026, 5, 1)
    assert req.end == date(2026, 5, 31)
    assert req.top_n == 2


def test_empty_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("")


def test_unparseable_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("blarg fasdfasdf")


def test_default_top_n_is_10() -> None:
    req = parse("haziran 2026")
    assert req.top_n == 10
