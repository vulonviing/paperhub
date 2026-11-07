"""Agent-CLI provider: auto-detection, subprocess client, platform guard."""

from __future__ import annotations

import json
import stat
import sys
import textwrap
from typing import Any, cast

import pytest

from paperhub.agents.base import build_llm, default_model_for_provider, detect_provider
from paperhub.agents.cli_client import CLIClient, CLIClientError
from paperhub.agents.health import check_llm
from paperhub.config import Settings
from paperhub.utils import UNSUPPORTED_PLATFORM_MESSAGE, is_unsupported_platform


def _fake_cli(tmp_path, name: str, body: str):
    script = tmp_path / name
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def path_with(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    return tmp_path


def test_detect_provider_prefers_installed_cli(path_with) -> None:
    _fake_cli(path_with, "codex", "")
    assert detect_provider({"openai": "key"}) == "codex-cli"


def test_detect_provider_prefers_claude_over_codex(path_with) -> None:
    _fake_cli(path_with, "codex", "")
    _fake_cli(path_with, "claude", "")
    assert detect_provider() == "claude-cli"


def test_detect_provider_falls_back_to_api_key_then_default(path_with) -> None:
    assert detect_provider({"anthropic": "key"}) == "anthropic"
    assert detect_provider({}) == "openai"


def test_settings_explicit_provider_beats_detection(path_with, monkeypatch) -> None:
    _fake_cli(path_with, "claude", "")
    monkeypatch.setenv("PAPERHUB_PROVIDER", "ollama")
    assert Settings().resolved_provider() == "ollama"


def test_cli_provider_accepts_matching_family_model(path_with, monkeypatch) -> None:
    monkeypatch.delenv("PAPERHUB_PROVIDER", raising=False)
    settings = Settings(PAPERHUB_MODEL="claude-sonnet-4-6")
    assert settings.model_for_provider("claude-cli") == "claude-sonnet-4-6"
    assert settings.model_for_provider("codex-cli") == default_model_for_provider("codex-cli")


def test_build_llm_returns_cli_client() -> None:
    client = build_llm(provider="claude-cli")
    assert isinstance(client, CLIClient)
    assert client.provider == "claude-cli"
    assert client.model == "sonnet"
    assert client.effort == "medium"


async def test_claude_client_parses_json_result(path_with) -> None:
    _fake_cli(
        path_with,
        "claude",
        """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"is_error": False, "result": '{"ok": true}'}))
        """,
    )
    client = CLIClient(provider="claude-cli")
    out = await client.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert json.loads(out) == {"ok": True}


async def test_claude_client_passes_model_only_when_set(path_with) -> None:
    _fake_cli(
        path_with,
        "claude",
        """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"is_error": False, "result": " ".join(sys.argv[1:])}))
        """,
    )
    default = await CLIClient(provider="claude-cli").complete(system="s", messages=[])
    assert "--model sonnet" in default
    chosen = await CLIClient(provider="claude-cli", model="sonnet").complete(
        system="s", messages=[]
    )
    assert "--model sonnet" in chosen


async def test_claude_client_raises_on_reported_error(path_with) -> None:
    _fake_cli(
        path_with,
        "claude",
        """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"is_error": True, "result": "Not logged in"}))
        """,
    )
    with pytest.raises(CLIClientError, match="Not logged in"):
        await CLIClient(provider="claude-cli").complete(system="s", messages=[])


async def test_codex_client_reads_last_message_file(path_with) -> None:
    _fake_cli(
        path_with,
        "codex",
        """
        import sys
        args = sys.argv
        stdin = sys.stdin.read()
        out = args[args.index("--output-last-message") + 1]
        open(out, "w").write('{"echo": %d}' % len(stdin))
        print("noise on stdout")
        """,
    )
    client = CLIClient(provider="codex-cli")
    out = await client.complete(system="sys", messages=[{"role": "user", "content": "hello"}])
    assert json.loads(out)["echo"] > 0


async def test_cli_client_nonzero_exit_includes_stderr(path_with) -> None:
    _fake_cli(
        path_with,
        "codex",
        """
        import sys
        sys.stdin.read()
        sys.stderr.write("please run codex login")
        sys.exit(3)
        """,
    )
    with pytest.raises(CLIClientError, match="please run codex login"):
        await CLIClient(provider="codex-cli").complete(system="s", messages=[])


async def test_cli_client_timeout(path_with) -> None:
    _fake_cli(path_with, "claude", "import time\ntime.sleep(30)\n")
    with pytest.raises(CLIClientError, match="did not answer"):
        await CLIClient(provider="claude-cli", timeout=0.5).complete(system="s", messages=[])


async def test_cli_client_missing_executable_has_install_hint(path_with) -> None:
    with pytest.raises(CLIClientError, match="not found on PATH"):
        await CLIClient(provider="claude-cli").complete(system="s", messages=[])


async def test_health_check_reports_missing_cli(path_with) -> None:
    result = await check_llm("claude-cli")
    assert not result.ok
    assert "not found on PATH" in result.message


def test_unsupported_platform_detection(monkeypatch) -> None:
    for platform in ("win32", "linux"):
        monkeypatch.setattr("sys.platform", platform)
        assert is_unsupported_platform()
    monkeypatch.setattr("sys.platform", "darwin")
    assert not is_unsupported_platform()
    assert "macOS only" in UNSUPPORTED_PLATFORM_MESSAGE


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_main_refuses_non_macos(monkeypatch, capsys, platform) -> None:
    from paperhub import interactive_cli as cli

    monkeypatch.setattr("sys.platform", platform)
    assert cli.main() == 1
    assert "macOS only" in capsys.readouterr().err


def test_list_cli_models_claude_aliases() -> None:
    from paperhub.agents.cli_client import list_cli_models

    assert list_cli_models("claude-cli") == ["fable", "opus", "sonnet"]
    assert list_cli_models("openai") == []


def test_list_cli_models_codex_reads_cache(tmp_path, monkeypatch) -> None:
    from paperhub.agents.cli_client import list_cli_models

    (tmp_path / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "gpt-a", "visibility": "list"},
                    {"slug": "hidden-one", "visibility": "hide"},
                    {"slug": "gpt-b", "visibility": "list"},
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert list_cli_models("codex-cli") == ["gpt-a", "gpt-b"]


def test_list_cli_models_codex_missing_cache_is_empty(tmp_path, monkeypatch) -> None:
    from paperhub.agents.cli_client import list_cli_models

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert list_cli_models("codex-cli") == []


def _state(provider: str):
    from paperhub import interactive_cli as cli

    return cli.LauncherState(
        provider=provider, model=None, language="en", period="month", year=2026, month=5,
        day=None, week=None, start=None, end=None, top_n=5, concurrency=2,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("answers", "expected"),
    [(["2"], "opus"), (["default"], None), (["custom", "haiku"], "haiku")],
)
def test_model_picker_for_claude_cli(monkeypatch, answers, expected) -> None:
    from rich.console import Console

    from paperhub import interactive_cli as cli

    replies = iter(answers)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(replies))
    state = _state("claude-cli")
    cli.choose_model(Console(record=True), state)
    assert state.model == expected
    assert state.provider == "claude-cli"


def test_typing_openai_model_on_codex_cli_keeps_provider() -> None:
    from paperhub import interactive_cli as cli

    state = _state("codex-cli")
    cli.apply_model_choice(state, "gpt-5.4-mini")
    assert state.provider == "codex-cli"
    assert state.model == "gpt-5.4-mini"


def test_cli_provider_specific_env_model(monkeypatch) -> None:
    monkeypatch.delenv("PAPERHUB_PROVIDER", raising=False)
    assert Settings(PAPERHUB_CLAUDE_CLI_MODEL="opus").model_for_provider("claude-cli") == "opus"


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    home = tmp_path / "codex_home"
    home.mkdir()
    (home / "config.toml").write_text('model = "gpt-z"\nmodel_reasoning_effort = "low"\n')
    (home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-z",
                        "visibility": "list",
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "high"},
                            {"effort": "ultra"},
                        ],
                    },
                    {"slug": "gpt-y", "visibility": "list"},
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


def test_codex_default_model_comes_from_its_config(codex_home) -> None:
    assert default_model_for_provider("codex-cli") == "gpt-z"


def test_codex_default_model_falls_back_to_first_listed(codex_home) -> None:
    (codex_home / "config.toml").unlink()
    assert default_model_for_provider("codex-cli") == "gpt-z"


def test_codex_effort_levels_follow_model_and_hide_ultra(codex_home) -> None:
    from paperhub.agents.cli_models import cli_effort_levels

    assert cli_effort_levels("codex-cli", "gpt-z") == ["low", "high"]
    assert cli_effort_levels("codex-cli", "gpt-y") == ["low", "medium", "high", "xhigh"]
    assert cli_effort_levels("claude-cli", "sonnet") == ["low", "medium", "high", "xhigh", "max"]


async def test_claude_client_passes_effort(path_with) -> None:
    _fake_cli(
        path_with,
        "claude",
        """
        import json, sys
        sys.stdin.read()
        print(json.dumps({"is_error": False, "result": " ".join(sys.argv[1:])}))
        """,
    )
    out = await CLIClient(provider="claude-cli", effort="high").complete(system="s", messages=[])
    assert "--effort high" in out
    out = await CLIClient(provider="claude-cli").complete(system="s", messages=[])
    assert "--effort medium" in out


async def test_codex_client_passes_effort_and_model(path_with, codex_home) -> None:
    _fake_cli(
        path_with,
        "codex",
        """
        import sys
        sys.stdin.read()
        out = sys.argv[sys.argv.index("--output-last-message") + 1]
        open(out, "w").write(" ".join(sys.argv[1:]))
        """,
    )
    out = await CLIClient(provider="codex-cli", effort="high").complete(system="s", messages=[])
    assert 'model_reasoning_effort="high"' in out
    assert "--model gpt-z" in out


def test_effort_command_and_run_table(codex_home, monkeypatch) -> None:
    from rich.console import Console

    from paperhub import interactive_cli as cli

    monkeypatch.delenv("PAPERHUB_PROVIDER", raising=False)
    settings = Settings()
    state = _state("codex-cli")
    assert cli.model_with_effort(settings, state) == "gpt-z · effort medium"
    assert cli.set_effort(state, "high") == "Effort set to high."
    assert cli.model_with_effort(settings, state) == "gpt-z · effort high"
    assert "must be one of" in (cli.set_effort(state, "max") or "")
    # A stale pick is ignored once the model no longer supports it.
    state.model = "gpt-y"
    state.effort = "max"
    assert cli.effective_effort(settings, state) == "medium"
    state.provider = "ollama"
    assert cli.effective_effort(settings, state) is None
    assert cli.handle_set(Console(record=True), state, "effort", ["high"]) == (
        "ollama has no effort setting."
    )


def test_openai_effort_shows_in_run_table() -> None:
    from paperhub import interactive_cli as cli

    state = _state("openai")
    state.model = "gpt-5.4-mini"
    assert cli.model_with_effort(Settings(), state) == "gpt-5.4-mini · effort medium"


def test_paperhub_effort_reaches_cli_client(path_with, tmp_path) -> None:
    from paperhub import PaperHub

    hub = PaperHub(provider="claude-cli", effort="max", settings=Settings(), cache_dir=tmp_path)
    assert hub.effort == "max"
    assert cast(Any, hub.llm).effort == "max"
    openai_hub = PaperHub(
        provider="openai", effort="low", settings=Settings(OPENAI_API_KEY="k"), cache_dir=tmp_path
    )
    assert cast(Any, openai_hub.llm).reasoning_effort == "low"


def _help_text(state) -> str:
    from rich.console import Console

    from paperhub import interactive_cli as cli

    console = Console(record=True, width=160)
    console.print(cli.command_table(state))
    return console.export_text()


def _home_text(state) -> str:
    from rich.console import Console

    from paperhub import interactive_cli as cli

    console = Console(record=True, width=200)
    cli.render_home(console, state)
    return console.export_text()


@pytest.mark.parametrize(
    ("provider", "shown"),
    [
        ("claude-cli", True),
        ("codex-cli", True),
        ("openai", True),
        ("anthropic", False),
        ("google", False),
        ("ollama", False),
    ],
)
def test_effort_command_listed_only_for_providers_with_effort(provider, shown) -> None:
    state = _state(provider)
    assert ("/effort" in _help_text(state)) is shown
    assert ("/effort" in _home_text(state)) is shown


def test_effort_command_disappears_when_provider_changes() -> None:
    state = _state("claude-cli")
    assert "/effort" in _home_text(state)
    state.provider = "ollama"
    assert "/effort" not in _home_text(state)
