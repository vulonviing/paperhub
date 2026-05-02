"""Smoke tests for the interactive CLI module."""

from __future__ import annotations

from rich.console import Console

import paperhub.interactive_cli as cli
from paperhub.agents.health import LLMHealthCheck
from paperhub.config import Settings
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


def test_api_keys_doc_fallback_when_docs_are_not_installed(tmp_path, monkeypatch) -> None:
    console = Console(record=True)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    cli.print_doc(console, "docs/API_KEYS.md")

    output = console.export_text()
    assert "PaperHub API Keys" in output
    assert "Missing documentation file" not in output


def test_startup_version_command(capsys) -> None:
    console = Console()
    state = cli.LauncherState(
        provider="openai",
        model=None,
        language="en",
        period="month",
        year=2026,
        month=5,
        day=None,
        week=None,
        start=None,
        end=None,
        top_n=5,
        concurrency=2,
    )

    assert cli.dispatch_startup_command(console, state, "version", []) == 0
    assert "PaperHub v" in capsys.readouterr().out


def test_set_key_runs_llm_health_check(tmp_path, monkeypatch) -> None:
    console = Console(record=True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = cli.LauncherState(
        provider="openai",
        model=None,
        language="en",
        period="month",
        year=2026,
        month=5,
        day=None,
        week=None,
        start=None,
        end=None,
        top_n=5,
        concurrency=2,
    )
    seen: dict[str, str | None] = {}

    monkeypatch.setattr(
        cli,
        "save_user_config_value",
        lambda key, value: tmp_path / ".env",
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: Settings(_env_file=None, PAPERHUB_OPENAI_MODEL="gpt-5.4-mini"),
    )

    async def fake_check(provider, **kwargs):
        seen.update(provider=provider, api_key=kwargs["api_key"], model=kwargs["model"])
        return LLMHealthCheck(
            provider="openai",
            model="gpt-5.4-mini",
            ok=True,
            message="LLM check passed.",
        )

    monkeypatch.setattr(cli, "check_llm", fake_check)

    notice = cli.save_api_key(console, state, ["openai", "sk-test"])

    assert notice is not None
    assert "Saved OPENAI_API_KEY" in notice
    assert "LLM check passed" in notice
    assert seen == {
        "provider": "openai",
        "api_key": "sk-test",
        "model": "gpt-5.4-mini",
    }


def test_set_key_warns_when_shell_env_overrides_saved_key(tmp_path, monkeypatch) -> None:
    console = Console(record=True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
    state = cli.LauncherState(
        provider="openai",
        model=None,
        language="en",
        period="month",
        year=2026,
        month=5,
        day=None,
        week=None,
        start=None,
        end=None,
        top_n=5,
        concurrency=2,
    )
    seen: dict[str, str | None] = {}

    monkeypatch.setattr(
        cli,
        "save_user_config_value",
        lambda key, value: tmp_path / ".env",
    )
    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda: Settings(_env_file=None, PAPERHUB_OPENAI_MODEL="gpt-5.4-mini"),
    )

    async def fake_check(provider, **kwargs):
        seen.update(provider=provider, api_key=kwargs["api_key"], model=kwargs["model"])
        return LLMHealthCheck(
            provider="openai",
            model="gpt-5.4-mini",
            ok=True,
            message="LLM check passed.",
        )

    monkeypatch.setattr(cli, "check_llm", fake_check)

    notice = cli.save_api_key(console, state, ["openai", "sk-saved"])

    assert notice is not None
    assert "overrides the saved PaperHub config value" in notice
    assert seen["api_key"] == "sk-shell"
