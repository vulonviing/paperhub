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
    default_model_for_provider,
    infer_provider_from_model,
)
from .dates import pretty_period, resolve_range
from .fetchers import HFAPIFetcher, HFHtmlFetcher, papers_in_range
from .formatter import LABELS
from .localization import normalize_language
from .models import PaperSummary, RunRequest
from .orchestrator import run_all


def find_project_root(start: Path) -> Path:
    for root in (start, Path(__file__).resolve()):
        for candidate in [root, *root.parents]:
            if (candidate / "pyproject.toml").exists() and (
                candidate / "src" / "paperhub"
            ).exists():
                return candidate
    return Path.cwd().resolve()


REPO_ROOT = find_project_root(Path.cwd().resolve())

PROVIDERS = ("openai", "anthropic", "google")
LANGUAGES = ("en", "tr")
MODEL_CHOICES = (
    ("openai", DEFAULT_MODELS["openai"]),
    ("anthropic", DEFAULT_MODELS["anthropic"]),
    ("google", DEFAULT_MODELS["google"]),
)
KEY_ENV_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
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
        has_key = bool(provider_key(settings, provider))
        marker = "selected" if provider == state.provider else ""
        table.add_row(
            f"{provider} {marker}".strip(),
            "ready" if has_key else "missing",
            KEY_ENV_BY_PROVIDER[provider],
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
    table.add_row("Provider", state.provider)
    table.add_row("Provider API key", "ready" if selected_key else "missing")
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
    console.print(command_table())
    if notice:
        console.print(f"[green]{notice}[/green]")
    console.print("[dim]Type '/help' for commands. Type '/quit' to exit.[/dim]")


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
        ("/status",   "",                              "Show provider, model, API key status, date range and top-n."),
        ("/provider", "/provider openai",              "Switch provider. Opens picker when called with no argument."),
        ("/model",    "/model gpt-5.4-mini",           "Switch model. Use 'default' to reset, 'custom' to free-type."),
        ("/language", "/language tr",                  "Set output language. Options: en, tr."),
        ("/date",     "/date 2026-05",                 "Set period — month. Format: YYYY-MM"),
        ("",          "/date 2026",                    "Set period — full year. Format: YYYY"),
        ("",          "/date 2026-05-15",              "Set period — single day. Format: YYYY-MM-DD"),
        ("",          "/date 2026-W18",                "Set period — ISO week. Format: YYYY-WNN"),
        ("",          "/date 2026-05-01 2026-05-31",   "Set period — custom range. Two ISO dates separated by space."),
        ("/top",      "/top 10",                       "Override number of papers to fetch and summarize."),
        ("/metadata", "",                              "Fetch paper list from HuggingFace without calling the LLM."),
        ("/run",      "",                              "Run the full pipeline: fetch → download PDFs → summarize."),
        ("/guide",    "",                              "Print the getting-started guide (docs/HOW_TO_START.md)."),
        ("/api-keys", "",                              "Print API key setup instructions (docs/API_KEYS.md)."),
        ("/quit",     "",                              "Exit the launcher."),
    ]

    for cmd, example, desc in rows:
        table.add_row(cmd, example, desc)

    return table


def model_table() -> Table:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Choice")
    table.add_column("Provider")
    table.add_column("Model")
    for idx, (provider, model) in enumerate(MODEL_CHOICES, 1):
        table.add_row(str(idx), provider, model)
    table.add_row("default", "selected", "Use the selected provider's default")
    table.add_row("custom", "selected or inferred", "Enter any model id")
    return table


def print_doc(console: Console, relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    if not path.exists():
        console.print(f"[red]Missing documentation file:[/red] {path}")
        return
    console.print(Markdown(path.read_text(encoding="utf-8")))


def parse_date_args(args: list[str]) -> tuple[str, int | None, int | None, int | None, int | None, date | None, date | None]:
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
        return "custom", None, None, None, None, date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))

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


def _run_async(coro):
    """Run a coroutine, draining httpx background cleanup tasks before closing the loop.

    asyncio.run() closes the loop immediately after the coroutine finishes, which
    causes httpx connection-pool cleanup tasks to raise RuntimeError('Event loop is
    closed'). Running on an explicit loop and draining pending tasks first avoids that.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
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
        with console.status(
            "[cyan]Connecting to HuggingFace…[/cyan]", spinner="dots"
        ):
            primary = HFAPIFetcher(client)
            fallback = HFHtmlFetcher(client)
            papers = await papers_in_range(
                primary, start_date, end_date, request.top_n, fallback=fallback
            )

        if not papers:
            console.print("[yellow]  ⚠  No papers found for this period.[/yellow]")
            return []

        console.print(
            f"  [green]✓[/green]  HuggingFace — "
            f"[bold]{len(papers)}[/bold] papers fetched"
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
                progress.print(
                    f"  {icon}  [{n}/{total}]  {short_title}  [dim]{source}[/dim]"
                )
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
                    f"[red]{labels['error']}:[/red] {s.error}\n\n"
                    f"[dim]arxiv:{s.arxiv_id}[/dim]",
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


def run_full_pipeline(console: Console, state: LauncherState) -> None:
    settings = load_settings()
    selected_key = provider_key(settings, state.provider)
    if not selected_key:
        env_name = KEY_ENV_BY_PROVIDER[state.provider]
        console.print(
            Panel(
                f"Missing API key for provider '{state.provider}'. Set {env_name} in .env "
                "or the shell environment before running summaries.",
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
    return f"Provider set to {state.provider}; model reset to provider default."


def choose_model(console: Console, state: LauncherState) -> str | None:
    console.print(Panel(model_table(), border_style="magenta", title="/model"))
    choice = Prompt.ask(
        "Choose model",
        choices=["1", "2", "3", "default", "custom"],
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
            console.print("[red]Use:[/red] /provider openai|anthropic|google")
            return None
        state.provider = args[0].lower()
        state.model = None
        return f"Provider set to {state.provider}; model reset to provider default."

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
    if command in {"provider", "p", "model", "m", "language", "lang", "l"} and len(parts) == 1:
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
    if command in {"providers", "keys"}:
        console.print(provider_table(load_settings(), state))
        return True
    if command in {"guide", "how-to-start", "start"}:
        print_doc(console, "docs/HOW_TO_START.md")
        return True
    if command in {"api-keys", "apikeys", "setup-keys"}:
        print_doc(console, "docs/API_KEYS.md")
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
    return parser.parse_args()


def main() -> int:
    os.chdir(REPO_ROOT)
    args = parse_args()
    console = Console()
    state = initial_state(args)

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
