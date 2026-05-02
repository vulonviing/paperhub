"""CLI smoke tests using Click's test runner."""

from __future__ import annotations

from click.testing import CliRunner

from paperhub.cli import main


def test_cli_no_args_exits_with_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    assert "Usage" in result.output or "Usage" in (result.stderr or "")


def test_cli_no_args_can_be_turkish() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--language", "tr"])
    assert result.exit_code == 2
    assert "Kullanım" in result.output or "Kullanım" in (result.stderr or "")


def test_cli_unparseable_query_exits_2() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["asdfasdf"])
    assert result.exit_code == 2


def test_cli_missing_provider_key_exits_3(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["mayis 2026 top 3"])
    assert result.exit_code == 3
    assert "API key" in (result.stderr or "") + (result.output or "")


def test_cli_missing_provider_key_can_be_turkish(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["--language", "tr", "mayis 2026 top 3"])
    assert result.exit_code == 3
    assert "API anahtarı" in (result.stderr or "") + (result.output or "")
