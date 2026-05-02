"""Tests for the `dates` module."""

from __future__ import annotations

from datetime import date

import pytest

from paperhub.dates import iter_dates, pretty_period, resolve_range
from paperhub.models import RunRequest


def test_resolve_day() -> None:
    req = RunRequest(period="day", year=2026, month=5, day=15)
    start, end = resolve_range(req, today=date(2026, 6, 1))
    assert start == end == date(2026, 5, 15)


def test_resolve_month_clipped_to_today() -> None:
    req = RunRequest(period="month", year=2026, month=5)
    start, end = resolve_range(req, today=date(2026, 5, 10))
    assert start == date(2026, 5, 1)
    assert end == date(2026, 5, 10)


def test_resolve_month_full_when_in_past() -> None:
    req = RunRequest(period="month", year=2025, month=2)
    start, end = resolve_range(req, today=date(2026, 5, 1))
    assert start == date(2025, 2, 1)
    assert end == date(2025, 2, 28)


def test_resolve_year_clipped_to_today() -> None:
    req = RunRequest(period="year", year=2026)
    start, end = resolve_range(req, today=date(2026, 5, 10))
    assert start == date(2026, 1, 1)
    assert end == date(2026, 5, 10)


def test_resolve_iso_week() -> None:
    req = RunRequest(period="week", year=2026, week=18)
    start, end = resolve_range(req, today=date(2026, 12, 31))
    assert start.isoweekday() == 1
    assert (end - start).days == 6


def test_resolve_custom_range() -> None:
    req = RunRequest(period="custom", start=date(2026, 4, 1), end=date(2026, 4, 30))
    start, end = resolve_range(req, today=date(2026, 5, 1))
    assert start == date(2026, 4, 1)
    assert end == date(2026, 4, 30)


def test_resolve_custom_range_clip() -> None:
    req = RunRequest(period="custom", start=date(2026, 4, 1), end=date(2027, 4, 30))
    start, end = resolve_range(req, today=date(2026, 5, 1))
    assert end == date(2026, 5, 1)


def test_iter_dates_inclusive() -> None:
    days = iter_dates(date(2026, 5, 1), date(2026, 5, 3))
    assert days == [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)]


def test_iter_dates_empty_when_reversed() -> None:
    assert iter_dates(date(2026, 5, 5), date(2026, 5, 1)) == []


def test_pretty_period_month() -> None:
    req = RunRequest(period="month", year=2026, month=5)
    assert pretty_period(req) == "May 2026"


def test_pretty_period_month_turkish() -> None:
    req = RunRequest(period="month", year=2026, month=5, language="tr")
    assert pretty_period(req) == "Mayıs 2026"


def test_resolve_day_requires_full_date() -> None:
    req = RunRequest(period="day", year=2026, month=5)
    with pytest.raises(ValueError):
        resolve_range(req)
