"""Interactive PaperHub launcher — `paperhub` console entrypoint.

Start the launcher:

    paperhub
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from . import PaperHub, __version__, load_settings
from .agents.base import (
    DEFAULT_MODELS,
    DEFAULT_PROVIDER,
    OPENAI_CHEAPER_ALTERNATIVE,
    default_model_for_provider,
    infer_provider_from_model,
)
from .agents.health import LLMHealthCheck, check_llm
from .config import delete_user_config_value, save_user_config_value, user_config_env_path
from .dates import pretty_period, resolve_range
from .fetchers import HFAPIFetcher, HFHtmlFetcher, papers_in_range
from .formatter import LABELS
from .localization import normalize_language
from .models import PaperSummary, RunRequest
from .orchestrator import run_all


def _humanize_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fetch_ollama_models(base_url: str) -> list[dict]:
    """Return installed Ollama models as [{name, size}] sorted by name.

    Calls Ollama's /api/tags endpoint (native Ollama API, not OpenAI-compat).
    Returns an empty list if Ollama is unreachable.
    """

    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return sorted(
            [
                {"name": m["name"], "size": _humanize_bytes(m.get("size", 0))}
                for m in data.get("models", [])
            ],
            key=lambda x: x["name"],
        )
    except Exception:
        return []


def find_project_root(start: Path) -> Path:
    for root in (start, Path(__file__).resolve()):
        for candidate in [root, *root.parents]:
            if (candidate / "pyproject.toml").exists() and (
                candidate / "src" / "paperhub"
            ).exists():
                return candidate
    return Path.cwd().resolve()


REPO_ROOT = find_project_root(Path.cwd().resolve())

PROVIDERS = ("openai", "anthropic", "google", "ollama")
LOCAL_PROVIDERS = frozenset({"ollama"})
LANGUAGES = ("en", "tr")
MODEL_CHOICES = (
    ("openai", DEFAULT_MODELS["openai"]),
    ("openai", OPENAI_CHEAPER_ALTERNATIVE),  # gpt-4.1-mini — budget option
    ("anthropic", DEFAULT_MODELS["anthropic"]),
    ("google", DEFAULT_MODELS["google"]),
    ("ollama", "gemma4:e2b"),  # local — ~2 B params
    ("ollama", "gemma4:e4b"),  # local — ~4 B params
)
KEY_ENV_BY_PROVIDER: dict[str, str | None] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": None,  # No API key — Ollama is a local service
}

LOGO = r"""
 ____   _    ____  _____ ____  _   _ _   _ ____
|  _ \ / \  |  _ \| ____|  _ \| | | | | | | __ )
| |_) / _ \ | |_) |  _| | |_) | |_| | | | |  _ \
|  __/ ___ \|  __/| |___|  _ <|  _  | |_| | |_) |
|_| /_/   \_\_|   |_____|_| \_\_| |_|\___/|____/
                     ____ _     ___
                    / ___| |   |_ _|
                   | |   | |    | |
                   | |___| |___ | |
                    \____|_____|___|
"""

DOC_FALLBACKS = {
    "docs/API_KEYS.md": """
# PaperHub API Keys

Save provider keys with:

```bash
paperhub version
paperhub set-key openai
paperhub set-key anthropic
paperhub set-key google
paperhub keys
paperhub check-llm
paperhub check-llm ollama
paperhub api-keys
paperhub config-path
paperhub clear-cache keys openai
```

Ollama is a local provider — no API key needed:

```bash
paperhub check-llm ollama
paperhub --provider ollama --model gemma4:e2b
```

Inside the interactive launcher:

```text
/set-key openai
/check-llm
/check-llm ollama
/api-keys
/clear-cache keys openai
```

PaperHub stores keys in its own per-user config file instead of the current
project's `.env`. Shell environment variables such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` still work and take priority.
""",
    "docs/HOW_TO_START.md": """
# PaperHub How To Start

```bash
pip install paperhub
paperhub set-key openai
paperhub
```

Common launcher commands:

```text
/help
/status
/provider
/provider ollama
/version
/model
/date 2026-05
/top 5
/concurrency 2
/metadata
/run
/set-key openai
/keys
/check-llm
/check-llm ollama
/config-path
/guide
/api-keys
/clear-cache
/clear-cache summaries
/clear-cache pdfs
/clear-cache keys openai
/quit
```

Python/Jupyter usage:

```python
from paperhub import PaperHub

hub = PaperHub(provider="openai")
hub.run(period="month", year=2026, month=5, top_n=5)

# Local model via Ollama (no API key needed):
hub = PaperHub(provider="ollama", model="gemma4:e2b")
hub.run(period="month", year=2026, month=5, top_n=5)
```
""",
}


@dataclass
class LauncherState:
    provider: str
    model: str | None
    language: str
    period: str
    year: int | None
    month: int | None
    day: int | None
    week: int | None
    start: date | None
    end: date | None
    top_n: int
    concurrency: int


def provider_key(settings, provider: str) -> str | None:
    return {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "google": settings.google_api_key,
    }.get(provider)


def effective_model(settings, state: LauncherState) -> str:
    return state.model or settings.model_for_provider(state.provider)


def build_request(state: LauncherState) -> RunRequest:
    return RunRequest(
        period=state.period,  # type: ignore[arg-type]
        year=state.year,
        month=state.month,
        day=state.day,
        week=state.week,
        start=state.start,
        end=state.end,
        top_n=state.top_n,
        language=normalize_language(state.language),
    )


def provider_table(settings, state: LauncherState) -> Table:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("API key")
    table.add_column("Env")
    table.add_column("Default model")
    for provider in PROVIDERS:
        is_local = provider in LOCAL_PROVIDERS
        has_key = bool(provider_key(settings, provider))
        marker = "selected" if provider == state.provider else ""
        key_status = "local — no key" if is_local else ("ready" if has_key else "missing")
        env_var = KEY_ENV_BY_PROVIDER.get(provider) or "(local)"
        table.add_row(
            f"{provider} {marker}".strip(),
            key_status,
            env_var,
            default_model_for_provider(provider),
        )
    return table


def current_run_table(settings, state: LauncherState) -> Table:
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    try:
        request = build_request(state)
        start, end = resolve_range(request)
        period = pretty_period(request)
        date_range = f"{start.isoformat()} to {end.isoformat()}"
        top_n = str(request.top_n)
    except Exception as exc:
        period = "not set"
        date_range = f"invalid: {exc}"
        top_n = str(state.top_n)

    selected_key = provider_key(settings, state.provider)
    if state.provider in LOCAL_PROVIDERS:
        key_status = "local LLM — no key needed"
    else:
        key_status = "ready" if selected_key else "missing"
    table.add_row("Provider", state.provider)
    table.add_row("Provider API key", key_status)
    table.add_row("Model", effective_model(settings, state))
    table.add_row("Language", state.language)
    table.add_row("Period", period)
    table.add_row("Date range", date_range)
    table.add_row("Top papers", top_n)
    table.add_row("Concurrency", str(state.concurrency))
    table.add_row("Working dir", str(REPO_ROOT))
    return table


def status_panel(console: Console, state: LauncherState) -> None:
    settings = load_settings()
    dashboard = Table.grid(expand=True)
    dashboard.add_column(ratio=1, no_wrap=True)
    dashboard.add_column(ratio=1)
    dashboard.add_row(
        LOGO.rstrip(),
        Group("[bold cyan]Current Run[/bold cyan]", current_run_table(settings, state)),
    )
    console.print(Panel(dashboard, border_style="cyan", title=f"PaperHub v{__version__}"))


def render_home(console: Console, state: LauncherState, notice: str | None = None) -> None:
    console.clear()
    status_panel(console, state)
    if notice:
        console.print(notice)
    console.print(
        "[dim]Commands: /run  /model  /provider  /date  /top  /check-llm  /set-key  "
        "/metadata  /keys  — type [bold]/help[/bold] for the full list  ·  /quit to exit[/dim]"
    )


def command_table() -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold white",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
    )
    table.add_column("Command", style="bold cyan", no_wrap=True, min_width=16)
    table.add_column("Input", style="dim", no_wrap=True, min_width=32)
    table.add_column("What it does", style="")

    # (command, example_input, description)
    rows = [
        ("/help", "", "Show this command list."),
        ("/status", "", "Show provider, model, API key status, date range and top-n."),
        ("/version", "", "Show the installed PaperHub CLI version."),
        (
            "/provider",
            "/provider openai",
            "Switch provider. Opens picker when called with no argument.",
        ),
        (
            "/model",
            "/model gpt-5.4-mini",
            "Switch model. Use 'default' to reset, 'custom' to free-type.",
        ),
        ("/language", "/language tr", "Set output language. Options: en, tr."),
        ("/date", "/date 2026-05", "Set period — month. Format: YYYY-MM"),
        ("", "/date 2026", "Set period — full year. Format: YYYY"),
        ("", "/date 2026-05-15", "Set period — single day. Format: YYYY-MM-DD"),
        ("", "/date 2026-W18", "Set period — ISO week. Format: YYYY-WNN"),
        (
            "",
            "/date 2026-05-01 2026-05-31",
            "Set period — custom range. Two ISO dates separated by space.",
        ),
        ("/top", "/top 10", "Override number of papers to fetch and summarize."),
        ("/concurrency", "/concurrency 2", "Set how many paper agents can run at once."),
        ("/metadata", "", "Fetch paper list from HuggingFace without calling the LLM."),
        ("/run", "", "Run the full pipeline: fetch → download PDFs → summarize."),
        ("/set-key", "/set-key openai", "Save a provider API key to PaperHub's user config file."),
        ("/keys", "", "Show provider key status."),
        (
            "/check-llm",
            "/check-llm openai",
            "Send a tiny live request to verify the selected provider/model (also: ollama).",
        ),
        ("/config-path", "", "Print PaperHub's user config path."),
        ("/guide", "", "Print the getting-started guide (docs/HOW_TO_START.md)."),
        ("/api-keys", "", "Show API key status and setup instructions."),
        ("/clear-cache", "", "Delete all cached summaries and PDFs."),
        ("", "/clear-cache summaries", "Delete only LLM summaries (keep PDFs)."),
        ("", "/clear-cache pdfs", "Delete only downloaded PDF files."),
        (
            "",
            "/clear-cache keys openai",
            "Delete a saved API key (provider: openai | anthropic | google | all).",
        ),
        ("/clear", "", "Redraw the launcher dashboard."),
        ("/quit", "", "Exit the launcher."),
    ]

    for cmd, example, desc in rows:
        table.add_row(cmd, example, desc)

    return table


def model_table() -> Table:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Choice")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Notes")
    notes = {
        ("openai", DEFAULT_MODELS["openai"]): "default — reasoning model",
        ("openai", OPENAI_CHEAPER_ALTERNATIVE): "budget option — standard chat pricing",
        ("ollama", "gemma4:e2b"): "local — requires Ollama running",
        ("ollama", "gemma4:e4b"): "local — requires Ollama running",
    }
    for idx, (provider, model) in enumerate(MODEL_CHOICES, 1):
        table.add_row(str(idx), provider, model, notes.get((provider, model), ""))
    table.add_row("default", "selected", "Use the selected provider's default", "")
    table.add_row("custom", "selected or inferred", "Enter any model id", "")
    return table


def print_doc(console: Console, relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    if not path.exists():
        fallback = DOC_FALLBACKS.get(relative_path)
        if fallback:
            console.print(Markdown(fallback))
            return
        console.print(f"[red]Missing documentation file:[/red] {path}")
        return
    console.print(Markdown(path.read_text(encoding="utf-8")))


def print_api_key_status(console: Console, state: LauncherState) -> None:
    settings = load_settings()
    console.print(Panel(provider_table(settings, state), border_style="magenta", title="API Keys"))
    console.print(f"[dim]PaperHub user config:[/dim] {user_config_env_path()}")


def save_api_key(console: Console, state: LauncherState, args: list[str]) -> str | None:
    if args:
        provider = args[0].lower()
    else:
        provider = Prompt.ask(
            "Choose provider",
            choices=list(PROVIDERS),
            default=state.provider,
        )
    if provider not in PROVIDERS:
        console.print(
            f"[red]Use:[/red] /set-key {' | '.join(p for p in PROVIDERS if p not in LOCAL_PROVIDERS)}"
        )
        return None

    # Local providers don't use API keys.
    if provider in LOCAL_PROVIDERS:
        if state.provider != provider:
            state.provider = provider
            state.model = None
        console.print(
            f"[yellow]{provider} is a local provider — no API key is required.[/yellow]\n"
            "Make sure Ollama is running: [dim]ollama serve[/dim]\n"
            "Then pull the model:         [dim]ollama pull gemma4:e2b[/dim]"
        )
        health = run_llm_health_check(console, state, provider)
        return format_health_check(health)

    env_name = KEY_ENV_BY_PROVIDER[provider]
    assert env_name is not None
    if len(args) >= 2:
        api_key = args[1].strip()
    else:
        api_key = Prompt.ask(f"{env_name}", password=True).strip()
    if not api_key:
        console.print("[red]API key cannot be empty.[/red]")
        return None

    path = save_user_config_value(env_name, api_key)
    if state.provider != provider:
        state.model = None
    state.provider = provider
    shell_api_key = os.environ.get(env_name)
    effective_api_key = shell_api_key or api_key
    override_notice = ""
    if shell_api_key and shell_api_key != api_key:
        override_notice = (
            f"\n[yellow]{env_name} is also set in your shell, so it overrides "
            "the saved PaperHub config value for this session.[/yellow]"
        )

    health = run_llm_health_check(console, state, provider, api_key=effective_api_key)
    return (
        f"[green]Saved {env_name} to {path}.[/green]"
        f"{override_notice}\n{format_health_check(health)}"
    )


def run_llm_health_check(
    console: Console,
    state: LauncherState,
    provider: str | None = None,
    *,
    api_key: str | None = None,
) -> LLMHealthCheck:
    provider = (provider or state.provider).lower()
    settings = load_settings()
    model = state.model if provider == state.provider else None
    resolved_model = model or settings.model_for_provider(provider)
    console.print(f"[cyan]Checking {provider} / {resolved_model}...[/cyan]")
    return _run_async(
        check_llm(
            provider,
            model=resolved_model,
            api_key=api_key or provider_key(settings, provider),
            openai_reasoning_effort=settings.openai_reasoning_effort(),
            ollama_base_url=settings.paperhub_ollama_base_url,
        )
    )


def format_health_check(result: LLMHealthCheck) -> str:
    prefix = f"{result.provider} / {result.model}"
    if result.ok:
        return f"[green]LLM check passed:[/green] {prefix}"
    return f"[red]LLM check failed:[/red] {prefix} - {result.message}"


def check_llm_command(console: Console, state: LauncherState, args: list[str]) -> bool:
    provider = args[0].lower() if args else state.provider
    if provider not in PROVIDERS:
        console.print(f"[red]Use:[/red] /check-llm {' | '.join(PROVIDERS)}")
        return False

    settings = load_settings()
    api_key = provider_key(settings, provider)

    # Local providers (Ollama) don't need an API key.
    if provider not in LOCAL_PROVIDERS and not api_key:
        env_name = KEY_ENV_BY_PROVIDER.get(provider)
        console.print(
            f"[red]Missing {env_name}.[/red] Run `paperhub set-key {provider}` "
            f"or `/set-key {provider}` first."
        )
        return False

    # For Ollama: show installed models before the health check.
    if provider == "ollama":
        _print_ollama_models(console, settings.paperhub_ollama_base_url)

    result = run_llm_health_check(console, state, provider, api_key=api_key)
    console.print(format_health_check(result))
    return result.ok


def _print_ollama_models(console: Console, base_url: str) -> None:
    """Print a table of locally installed Ollama models."""

    models = _fetch_ollama_models(base_url)
    if not models:
        console.print(
            "[yellow]No installed Ollama models found.[/yellow] "
            "Is Ollama running? Try: [dim]ollama serve[/dim]"
        )
        return
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Size", justify="right")
    for idx, m in enumerate(models, 1):
        table.add_row(str(idx), m["name"], m["size"])
    console.print(Panel(table, border_style="yellow", title="Installed Ollama Models"))


def parse_date_args(
    args: list[str],
) -> tuple[str, int | None, int | None, int | None, int | None, date | None, date | None]:
    """Parse structured date arguments into (period, year, month, day, week, start, end).

    Accepted formats:
      YYYY-MM-DD YYYY-MM-DD   custom range
      YYYY-MM-DD              single day
      YYYY-WXX or YYYY-WX     ISO week (e.g. 2026-W18)
      YYYY-MM                 month
      YYYY                    year
    """
    if not args:
        raise ValueError(
            "Use: /date YYYY-MM, /date YYYY, /date YYYY-MM-DD, /date YYYY-WXX, "
            "or /date YYYY-MM-DD YYYY-MM-DD"
        )

    text = " ".join(args).strip()

    # Custom range: two ISO dates
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})$", text)
    if m:
        return (
            "custom",
            None,
            None,
            None,
            None,
            date.fromisoformat(m.group(1)),
            date.fromisoformat(m.group(2)),
        )

    # Single day: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return "day", y, mo, d, None, None, None

    # ISO week: YYYY-WXX or YYYY-WX
    m = re.match(r"^(\d{4})-W(\d{1,2})$", text, re.IGNORECASE)
    if m:
        y, w = int(m.group(1)), int(m.group(2))
        return "week", y, None, None, w, None, None

    # Month: YYYY-MM
    m = re.match(r"^(\d{4})-(\d{2})$", text)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return "month", y, mo, None, None, None, None

    # Year: YYYY
    m = re.match(r"^(\d{4})$", text)
    if m:
        return "year", int(m.group(1)), None, None, None, None, None

    raise ValueError(
        f"Unrecognized date format: {text!r}. "
        "Use YYYY-MM, YYYY, YYYY-MM-DD, YYYY-WXX, or 'YYYY-MM-DD YYYY-MM-DD'."
    )


async def fetch_metadata(state: LauncherState):
    settings = load_settings()
    request = build_request(state)
    start, end = resolve_range(request)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=settings.paperhub_request_timeout_s,
    ) as client:
        primary = HFAPIFetcher(client)
        fallback = HFHtmlFetcher(client)
        return await papers_in_range(primary, start, end, request.top_n, fallback=fallback)


def print_metadata(console: Console, state: LauncherState) -> None:
    try:
        papers = _run_async(fetch_metadata(state))
    except Exception as exc:
        console.print(f"[red]Metadata fetch failed:[/red] {exc}")
        return

    if not papers:
        console.print("[yellow]No papers found for the selected period.[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Upvotes", justify="right")
    table.add_column("arXiv")
    table.add_column("Title")
    for idx, paper in enumerate(papers, 1):
        table.add_row(str(idx), str(paper.upvotes), paper.arxiv_id, paper.title)
    console.print(Panel(table, border_style="cyan", title="Metadata Preview"))


def _suppress_cleanup_errors(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Suppress the 'Event loop is closed' noise from httpx / OpenAI SDK teardown.

    AsyncOpenAI internally holds an httpx transport. When the event loop closes,
    httpx's connection-pool cleanup fires after the loop is gone, which produces a
    RuntimeError that Python prints to stderr. The error is harmless — all work is
    already done — so we silence it here and surface everything else normally.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
        return
    loop.default_exception_handler(context)


def _run_async(coro):
    """Run a coroutine, draining background cleanup tasks before closing the loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_suppress_cleanup_errors)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            # Drain in two passes: the first pass may schedule more cleanup tasks
            # (e.g. OpenAI SDK httpx transport teardown), the second catches those.
            for _ in range(2):
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        finally:
            loop.close()
            asyncio.set_event_loop(None)


async def _pipeline_with_progress(
    console: Console,
    hub: PaperHub,
    request: RunRequest,
    settings,
) -> list[PaperSummary]:
    """Run the full fetch + summarize pipeline with live Rich feedback."""

    start_date, end_date = resolve_range(request)
    period_label = pretty_period(request)

    console.print()
    console.print(
        Panel(
            f"[bold cyan]{period_label}[/bold cyan]  "
            f"[dim]{start_date.isoformat()} → {end_date.isoformat()}  ·  "
            f"top {request.top_n}  ·  {hub.provider} / {hub.model}[/dim]",
            border_style="cyan",
            title="[bold]PaperHub Run[/bold]",
            padding=(0, 1),
        )
    )
    console.print()

    timeout = settings.paperhub_request_timeout_s

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        # ── Metadata fetch ─────────────────────────────────────────────────
        with console.status("[cyan]Connecting to HuggingFace…[/cyan]", spinner="dots"):
            primary = HFAPIFetcher(client)
            fallback = HFHtmlFetcher(client)
            papers = await papers_in_range(
                primary, start_date, end_date, request.top_n, fallback=fallback
            )

        if not papers:
            console.print("[yellow]  ⚠  No papers found for this period.[/yellow]")
            return []

        console.print(
            f"  [green]✓[/green]  HuggingFace — [bold]{len(papers)}[/bold] papers fetched"
        )
        console.print()

        for meta in papers:
            hub.cache.put_meta(meta)

        # ── Per-paper agents ───────────────────────────────────────────────
        completed: list[PaperSummary] = []
        total = len(papers)

        with Progress(
            SpinnerColumn(),
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("[bold cyan]{task.completed}[/bold cyan]/[dim]{task.total}[/dim]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task(f"Summarizing {total} papers…", total=total)

            def _on_done(summary: PaperSummary) -> None:
                completed.append(summary)
                n = len(completed)
                icon = "[green]✓[/green]" if not summary.error else "[red]✗[/red]"
                short_title = summary.title[:52] + ("…" if len(summary.title) > 52 else "")
                elapsed = f"{summary.elapsed_s:.1f}s"
                source = "(cached)" if summary.elapsed_s < 0.05 else elapsed
                progress.print(f"  {icon}  [{n}/{total}]  {short_title}  [dim]{source}[/dim]")
                progress.update(task_id, advance=1)

            summaries = await run_all(
                papers,
                hub.llm,
                hub.cache,
                concurrency=hub.concurrency,
                max_pdf_chars=hub.max_pdf_chars,
                http_client=client,
                on_progress=_on_done,
                language=request.language,
            )

    console.print()
    ok = sum(1 for s in summaries if not s.error)
    err = len(summaries) - ok
    parts = [f"[green]{ok} summarized[/green]"]
    if err:
        parts.append(f"[red]{err} failed[/red]")
    console.print("  [bold]Done.[/bold]  " + "  ·  ".join(parts))
    console.print()

    return summaries


def render_rich_summaries(
    console: Console, summaries: list[PaperSummary], request: RunRequest
) -> None:
    """Print summaries as structured Rich panels in the terminal."""

    labels = LABELS[normalize_language(request.language)]
    period_label = pretty_period(request)

    # ── Run header ────────────────────────────────────────────────────────
    header = Text()
    header.append("PaperHub", style="bold cyan")
    header.append(f"  {period_label}", style="bold white")
    header.append(f"  ·  {len(summaries)} {labels['paper_unit']}", style="dim")
    console.print(Panel(header, border_style="cyan", padding=(0, 1)))
    console.print()

    for idx, s in enumerate(summaries, 1):
        # ── Error paper ───────────────────────────────────────────────────
        if s.error:
            console.print(
                Panel(
                    f"[red]{labels['error']}:[/red] {s.error}\n\n[dim]arxiv:{s.arxiv_id}[/dim]",
                    title=f"[bold]{idx}. {s.title}[/bold]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            console.print()
            continue

        # ── Links row ─────────────────────────────────────────────────────
        links = (
            f"[dim]arxiv:{s.arxiv_id}[/dim]  "
            f"[link=https://arxiv.org/pdf/{s.arxiv_id}][blue]PDF ↗[/blue][/link]  "
            f"[link=https://huggingface.co/papers/{s.arxiv_id}][blue]HF ↗[/blue][/link]"
        )

        # ── Section blocks ────────────────────────────────────────────────
        body = Text()

        for label_key, value in [
            ("motivation", s.motivation),
            ("method", s.method),
            ("findings", s.findings),
        ]:
            body.append(f"{labels[label_key]}\n", style="bold yellow")
            body.append(f"{value}\n\n", style="")

        if s.real_world_examples:
            body.append(f"{labels['examples']}\n", style="bold yellow")
            for ex in s.real_world_examples:
                body.append(f"  • {ex}\n", style="dim white")
            body.append("\n")

        body.append(f"{labels['summary']}\n", style="bold green")
        body.append(s.summary, style="")

        footer = f"[dim]{s.elapsed_s:.1f}s · {s.model_used}[/dim]"

        console.print(
            Panel(
                Group(links, "", body),
                title=f"[bold white]{idx}. {s.title}[/bold white]",
                subtitle=footer,
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        console.print()


def _clear_api_keys(console: Console, provider: str) -> None:
    """Delete one or all saved API keys from PaperHub's user config file."""

    cloud_providers = [p for p in PROVIDERS if p not in LOCAL_PROVIDERS]
    targets = cloud_providers if provider == "all" else [provider]

    invalid = [p for p in targets if p not in cloud_providers]
    if invalid:
        console.print(
            f"[red]Unknown provider(s):[/red] {', '.join(invalid)}\n"
            f"[dim]Valid choices: {' | '.join(cloud_providers)} | all[/dim]"
        )
        return

    removed, skipped = [], []
    for p in targets:
        env_name = KEY_ENV_BY_PROVIDER.get(p)
        if not env_name:
            continue
        _, existed = delete_user_config_value(env_name)
        if existed:
            removed.append(f"{p} ({env_name})")
        else:
            skipped.append(p)

    if removed:
        console.print(f"[green]API key(s) deleted:[/green] {', '.join(removed)}")
    if skipped:
        console.print(f"[dim]No saved key found for: {', '.join(skipped)}[/dim]")
    console.print(f"[dim]Config file: {user_config_env_path()}[/dim]")


def clear_cache_command(console: Console, args: list[str]) -> None:
    target = args[0].lower() if args else "all"

    # API key deletion: /clear-cache keys [provider|all]
    if target == "keys":
        provider = args[1].lower() if len(args) > 1 else "all"
        _clear_api_keys(console, provider)
        return

    cloud_providers = {p for p in PROVIDERS if p not in LOCAL_PROVIDERS}
    if target in cloud_providers:
        # /clear-cache openai  →  shorthand for /clear-cache keys openai
        _clear_api_keys(console, target)
        return

    valid_cache_targets = {"all", "summaries", "pdfs"}
    if target not in valid_cache_targets:
        console.print(
            "[red]Use:[/red] /clear-cache  |  /clear-cache summaries  |  /clear-cache pdfs  |  "
            "/clear-cache keys [openai|anthropic|google|all]"
        )
        return

    settings = load_settings()
    hub = PaperHub(settings=settings)
    cache = hub.cache

    do_summaries = target in {"all", "summaries"}
    do_pdfs = target in {"all", "pdfs"}
    deleted = cache.clear(summaries=do_summaries, pdfs=do_pdfs)

    parts = []
    if do_summaries:
        parts.append(f"[green]{deleted['summaries']}[/green] summaries")
    if do_pdfs:
        parts.append(f"[green]{deleted['pdf_text']}[/green] PDF text rows")
        parts.append(f"[green]{deleted['pdf_files']}[/green] PDF files")

    stats_after = cache.stats()
    console.print(f"[bold]Cache cleared:[/bold] {', '.join(parts)}")
    console.print(
        f"[dim]Remaining: {stats_after['summaries']} summaries, "
        f"{stats_after['pdf_files']} PDF files[/dim]"
    )


def run_full_pipeline(console: Console, state: LauncherState) -> None:
    settings = load_settings()
    selected_key = provider_key(settings, state.provider)
    if not selected_key and state.provider not in LOCAL_PROVIDERS:
        env_name = KEY_ENV_BY_PROVIDER.get(state.provider)
        console.print(
            Panel(
                f"Missing API key for provider '{state.provider}'. Save {env_name} with "
                f"/set-key {state.provider}, run paperhub set-key {state.provider}, "
                "or set it in the shell environment before running summaries.",
                border_style="red",
                title="API Key Required",
            )
        )
        print_doc(console, "docs/API_KEYS.md")
        return

    try:
        request = build_request(state)
    except Exception as exc:
        console.print(f"[red]Invalid date configuration:[/red] {exc}")
        return

    hub = PaperHub(
        provider=state.provider,
        model=state.model,
        language=state.language,
        concurrency=state.concurrency,
        settings=settings,
    )

    try:
        summaries = _run_async(_pipeline_with_progress(console, hub, request, settings))
    except Exception as exc:
        console.print(f"[red]Run failed:[/red] {exc}")
        return

    if summaries:
        render_rich_summaries(console, summaries, request)


def apply_model_choice(state: LauncherState, model: str | None) -> str:
    if model is None:
        state.model = None
        return f"Model set to {effective_model(load_settings(), state)}."

    inferred_provider = infer_provider_from_model(model)
    state.model = model
    if inferred_provider and inferred_provider in PROVIDERS and inferred_provider != state.provider:
        state.provider = inferred_provider
        return f"Provider set to {state.provider}; model set to {state.model}."
    return f"Model set to {state.model}."


def choose_provider(console: Console, state: LauncherState) -> str:
    settings = load_settings()
    console.print(Panel(provider_table(settings, state), border_style="magenta", title="/provider"))
    choice = Prompt.ask(
        "Choose provider",
        choices=list(PROVIDERS),
        default=state.provider,
    )
    state.provider = choice
    state.model = None
    return _provider_then_model(console, state)


def choose_model(console: Console, state: LauncherState) -> str | None:
    # When the active provider is Ollama, show locally installed models.
    if state.provider == "ollama":
        return _choose_ollama_model(console, state)

    console.print(Panel(model_table(), border_style="magenta", title="/model"))
    numbered = [str(i) for i in range(1, len(MODEL_CHOICES) + 1)]
    choice = Prompt.ask(
        "Choose model",
        choices=numbered + ["default", "custom"],
        default="default",
    )
    if choice == "default":
        return apply_model_choice(state, None)
    if choice == "custom":
        value = Prompt.ask("Model id").strip()
        if not value:
            console.print("[red]Model id cannot be empty.[/red]")
            return None
        return apply_model_choice(state, value)

    provider, model = MODEL_CHOICES[int(choice) - 1]
    state.provider = provider
    state.model = model
    return f"Provider set to {state.provider}; model set to {state.model}."


def _choose_ollama_model(console: Console, state: LauncherState) -> str | None:
    """Interactive model picker that lists locally installed Ollama models.

    - No models found  → model set to None with a hint to pull one.
    - Models found     → numbered list, first entry is the default selection.
                         No "provider default" option since it may not be installed.
    """
    settings = load_settings()
    console.print("[cyan]Fetching installed Ollama models…[/cyan]")
    models = _fetch_ollama_models(settings.paperhub_ollama_base_url)

    if not models:
        state.model = None
        console.print(
            "[yellow]No installed Ollama models found.[/yellow] "
            "Is Ollama running?  [dim]ollama serve[/dim]\n"
            "Pull a model first: [dim]ollama pull gemma4:e2b[/dim]\n"
            "[dim]Model set to none — run /model after pulling to select one.[/dim]"
        )
        return None

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Choice")
    table.add_column("Model")
    table.add_column("Size", justify="right")
    for idx, m in enumerate(models, 1):
        label = m["name"]
        if idx == 1:
            label += "  [dim](default)[/dim]"
        table.add_row(str(idx), label, m["size"])
    table.add_row("custom", "Enter any model id", "")
    console.print(Panel(table, border_style="magenta", title="/model — Installed Ollama models"))

    numbered = [str(i) for i in range(1, len(models) + 1)]
    choice = Prompt.ask(
        "Choose model",
        choices=numbered + ["custom"],
        default="1",  # First installed model is the safe default
    )
    if choice == "custom":
        value = Prompt.ask("Model id").strip()
        if not value:
            console.print("[red]Model id cannot be empty.[/red]")
            return None
        return apply_model_choice(state, value)

    selected_name = models[int(choice) - 1]["name"]
    return apply_model_choice(state, selected_name)


def _provider_then_model(console: Console, state: LauncherState) -> str:
    """After a provider change, immediately offer the model picker.

    Returns a combined notice string that covers both the provider and model selection.
    """
    provider_part = f"Provider set to [bold]{state.provider}[/bold]."
    console.print(f"\n{provider_part} Select a model:\n")
    model_part = choose_model(console, state)
    if model_part:
        return f"{provider_part} {model_part}"
    return provider_part


def choose_language(state: LauncherState) -> str:
    choice = Prompt.ask(
        "Choose output language",
        choices=list(LANGUAGES),
        default=state.language,
    )
    state.language = normalize_language(choice)
    return f"Language set to {state.language}."


def handle_set(console: Console, state: LauncherState, command: str, args: list[str]) -> str | None:
    if command in {"provider", "p"}:
        if not args:
            return choose_provider(console, state)
        if args[0].lower() not in PROVIDERS:
            console.print(f"[red]Use:[/red] /provider {' | '.join(PROVIDERS)}")
            return None
        state.provider = args[0].lower()
        state.model = None
        return _provider_then_model(console, state)

    if command in {"model", "m"}:
        if not args:
            return choose_model(console, state)
        value = " ".join(args).strip()
        if value.lower() in {"default", "none", "auto"}:
            return apply_model_choice(state, None)
        if value.lower() in {"custom", "other"}:
            custom_model = Prompt.ask("Model id").strip()
            if not custom_model:
                console.print("[red]Model id cannot be empty.[/red]")
                return None
            return apply_model_choice(state, custom_model)
        return apply_model_choice(state, value)

    if command in {"language", "lang", "l"}:
        if not args:
            return choose_language(state)
        try:
            state.language = normalize_language(args[0])
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return None
        return f"Language set to {state.language}."

    if command in {"date", "d"}:
        try:
            period, year, month, day, week, start, end = parse_date_args(args)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return None
        state.period = period
        state.year = year
        state.month = month
        state.day = day
        state.week = week
        state.start = start
        state.end = end
        label = " ".join(args)
        return f"Date set to: {label} ({period})"

    if command in {"top", "top-n"}:
        if not args:
            console.print("[red]Use:[/red] /top 5")
            return None
        try:
            top_value = int(args[0])
        except ValueError:
            console.print("[red]top must be an integer.[/red]")
            return None
        if not 1 <= top_value <= 200:
            console.print("[red]top must be between 1 and 200.[/red]")
            return None
        state.top_n = top_value
        return f"Top papers set to {state.top_n}."

    if command in {"concurrency", "workers"}:
        if not args:
            console.print("[red]Use:[/red] /concurrency 2")
            return None
        try:
            concurrency_value = int(args[0])
        except ValueError:
            console.print("[red]concurrency must be an integer.[/red]")
            return None
        state.concurrency = max(1, concurrency_value)
        return f"Concurrency set to {state.concurrency}."

    console.print(f"[red]Unknown command:[/red] {command}")
    return None


def normalize_command(token: str) -> str:
    return token[1:].lower() if token.startswith("/") else token.lower()


def command_name(line: str) -> str | None:
    try:
        parts = shlex.split(line)
    except ValueError:
        return None
    if not parts:
        return None
    return normalize_command(parts[0])


def needs_inline_prompt(line: str) -> bool:
    try:
        parts = shlex.split(line)
    except ValueError:
        return False
    if not parts:
        return False
    command = normalize_command(parts[0])
    # Provider always triggers interactive model picker afterwards — skip spinner.
    if command in {"provider", "p"}:
        return True
    if command in {"model", "m", "language", "lang", "l"} and len(parts) == 1:
        return True
    if command in {"set-key", "apikey", "api-key"} and len(parts) <= 2:
        return True
    return command in {"model", "m"} and len(parts) == 2 and parts[1].lower() in {"custom", "other"}


def dispatch(console: Console, state: LauncherState, line: str) -> bool:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        console.print(f"[red]Could not parse command:[/red] {exc}")
        return True

    if not parts:
        return True

    command = normalize_command(parts[0])
    args = parts[1:]

    if command in {"quit", "exit"}:
        return False
    if command in {"help", "?", "commands", "shortcuts"}:
        console.print(command_table())
        return True
    if command == "status":
        render_home(console, state)
        return True
    if command in {"version", "v"}:
        console.print(f"PaperHub v{__version__}")
        return True
    if command in {"providers", "keys"}:
        print_api_key_status(console, state)
        return True
    if command in {"check-llm", "llm-check", "health"}:
        check_llm_command(console, state, args)
        return True
    if command in {"guide", "how-to-start", "start"}:
        print_doc(console, "docs/HOW_TO_START.md")
        return True
    if command in {"api-keys", "apikeys", "setup-keys"}:
        print_api_key_status(console, state)
        print_doc(console, "docs/API_KEYS.md")
        return True
    if command in {"set-key", "apikey", "api-key"}:
        notice = save_api_key(console, state, args)
        if notice:
            render_home(console, state, notice)
        return True
    if command in {"config-path", "config"}:
        console.print(str(user_config_env_path()))
        return True
    if command == "metadata":
        print_metadata(console, state)
        return True
    if command == "run":
        run_full_pipeline(console, state)
        return True
    if command == "clear":
        render_home(console, state)
        return True
    if command in {"clear-cache", "cache-clear", "clearcache"}:
        clear_cache_command(console, args)
        return True
    if command == "set" and args:
        notice = handle_set(console, state, normalize_command(args[0]), args[1:])
        if notice:
            render_home(console, state, notice)
        return True

    notice = handle_set(console, state, command, args)
    if notice:
        render_home(console, state, notice)
    return True


def dispatch_with_loading(console: Console, state: LauncherState, line: str) -> bool:
    command = command_name(line)
    if command and not needs_inline_prompt(line):
        label = "Running PaperHub..." if command == "run" else "Loading..."
        with console.status(f"[bold cyan]{label}[/bold cyan]", spinner="dots"):
            return dispatch(console, state, line)
    return dispatch(console, state, line)


def initial_state(args: argparse.Namespace) -> LauncherState:
    settings = load_settings()
    provider = (args.provider or settings.paperhub_provider or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        provider = DEFAULT_PROVIDER
    today = date.today()
    return LauncherState(
        provider=provider,
        model=args.model,
        language=normalize_language(args.language),
        period="month",
        year=today.year,
        month=today.month,
        day=None,
        week=None,
        start=None,
        end=None,
        top_n=args.top_n or 5,
        concurrency=args.concurrency or min(settings.paperhub_concurrency, 2),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the interactive PaperHub launcher.")
    parser.add_argument("--provider", choices=PROVIDERS, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", choices=LANGUAGES, default="en")
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("command", nargs="?")
    parser.add_argument("command_args", nargs="*")
    return parser.parse_args()


def dispatch_startup_command(
    console: Console,
    state: LauncherState,
    command: str,
    args: list[str],
) -> int:
    normalized = normalize_command(command)
    if normalized in {"set-key", "apikey", "api-key"}:
        notice = save_api_key(console, state, args)
        if notice:
            console.print(notice)
            return 0
        return 2
    if normalized in {"keys", "providers"}:
        print_api_key_status(console, state)
        return 0
    if normalized in {"check-llm", "llm-check", "health"}:
        return 0 if check_llm_command(console, state, args) else 1
    if normalized in {"api-keys", "apikeys", "setup-keys"}:
        print_api_key_status(console, state)
        print_doc(console, "docs/API_KEYS.md")
        return 0
    if normalized in {"config-path", "config"}:
        console.print(str(user_config_env_path()))
        return 0
    if normalized in {"version", "v", "--version"}:
        console.print(f"PaperHub v{__version__}")
        return 0
    if normalized in {"clear-cache", "cache-clear", "clearcache"}:
        clear_cache_command(console, args)
        return 0
    console.print(f"[red]Unknown command:[/red] {command}")
    console.print(
        "[dim]Use `paperhub`, `paperhub version`, `paperhub set-key openai`, "
        "`paperhub check-llm`, `paperhub check-llm ollama`, or `paperhub api-keys`.[/dim]"
    )
    return 2


def main() -> int:
    os.chdir(REPO_ROOT)
    args = parse_args()
    console = Console()
    state = initial_state(args)

    if args.command:
        return dispatch_startup_command(console, state, args.command, args.command_args)

    render_home(console, state)

    try:
        while True:
            line = Prompt.ask("[bold cyan]paperhub[/bold cyan]").strip()
            if not dispatch_with_loading(console, state, line):
                break
    except KeyboardInterrupt:
        console.print("\n[dim]Safe quit.[/dim]")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
