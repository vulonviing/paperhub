"""Smoke tests for the interactive CLI module."""

from __future__ import annotations

from paperhub.interactive_cli import main, parse_date_args


def test_interactive_cli_is_importable() -> None:
    assert callable(main)


def test_parse_date_month() -> None:
    period, year, month, day, week, start, end = parse_date_args(["2026-05"])
    assert period == "month"
    assert year == 2026
    assert month == 5
    assert day is None


def test_parse_date_year() -> None:
    period, year, month, day, week, start, end = parse_date_args(["2026"])
    assert period == "year"
    assert year == 2026


def test_parse_date_day() -> None:
    period, year, month, day, week, start, end = parse_date_args(["2026-05-15"])
    assert period == "day"
    assert year == 2026
    assert month == 5
    assert day == 15


def test_parse_date_week() -> None:
    period, year, month, day, week, start, end = parse_date_args(["2026-W18"])
    assert period == "week"
    assert year == 2026
    assert week == 18


def test_parse_date_custom_range() -> None:
    from datetime import date

    period, year, month, day, week, start, end = parse_date_args(["2026-05-01", "2026-05-31"])
    assert period == "custom"
    assert start == date(2026, 5, 1)
    assert end == date(2026, 5, 31)
