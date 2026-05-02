"""PaperHub: Jupyter-friendly summaries of HuggingFace Daily Papers."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from datetime import date
from pathlib import Path
from typing import Any, cast, get_args

import httpx

from .agents.base import LLMClient, build_llm
from .cache import Cache
from .config import Settings, load_settings
from .dates import resolve_range
from .fetchers import HFAPIFetcher, HFHtmlFetcher, papers_in_range
from .formatter import display_summaries, render_markdown, render_plain
from .localization import OutputLanguage, normalize_language
from .models import PaperMeta, PaperSummary, Period, RunRequest
from .nl_parser import parse as parse_query
from .orchestrator import run_all

__all__ = [
    "PaperHub",
    "PaperMeta",
    "PaperSummary",
    "RunRequest",
    "Settings",
    "OutputLanguage",
    "build_llm",
    "load_settings",
    "normalize_language",
    "parse_query",
    "render_markdown",
    "render_plain",
]

__version__ = "0.1.0"


class PaperHub:
    """High-level entrypoint.

    Construct once, call `run(...)` per query. The class is intentionally
    thin: the heavy lifting lives in `fetchers`, `agents`, `orchestrator`.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        provider: str | None = None,
        cache_dir: str | Path | None = None,
        concurrency: int | None = None,
        max_pdf_chars: int | None = None,
        api_key: str | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
        language: str | None = None,
    ):
        self.settings = settings or load_settings()
        self.provider = provider or self.settings.paperhub_provider
        self.model = model or self.settings.model_for_provider(self.provider)
        self.concurrency = concurrency or self.settings.paperhub_concurrency
        self.max_pdf_chars = max_pdf_chars or self.settings.paperhub_max_pdf_chars
        self.language: OutputLanguage = normalize_language(language)
        cache_root = Path(cache_dir) if cache_dir else self.settings.cache_dir()
        self.cache = Cache(cache_root)
        provider_api_key = api_key if api_key is not None else self._settings_api_key(self.provider)
        self.llm = llm or build_llm(
            model=self.model,
            provider=self.provider,
            api_key=provider_api_key,
        )

    def run(
        self,
        query: str | None = None,
        *,
        period: str | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        week: int | None = None,
        start: date | None = None,
        end: date | None = None,
        top_n: int | None = None,
        language: str | None = None,
        display: bool = True,
        plain: bool = False,
    ) -> list[PaperSummary]:
        """Execute the full pipeline for a query (NL or programmatic).

        - `display=True` renders Markdown in Jupyter; otherwise it just
          returns the summary list and the Markdown is available via
          `render_markdown`.
        - `plain=True` skips the IPython render and returns plain text.
        """

        request = self._build_request(
            query=query,
            period=period,
            year=year,
            month=month,
            day=day,
            week=week,
            start=start,
            end=end,
            top_n=top_n,
            language=language,
        )
        summaries = _run_coro_sync(self._run_async(request))
        if display:
            display_summaries(summaries, request, force_plain=plain)
        return summaries

    async def arun(
        self,
        query: str | None = None,
        **kwargs: Any,
    ) -> list[PaperSummary]:
        """Async variant of `run` for use inside an existing event loop."""

        request = self._build_request(query=query, **kwargs)
        return await self._run_async(request)

    def _build_request(
        self,
        *,
        query: str | None,
        period: str | None = None,
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
        week: int | None = None,
        start: date | None = None,
        end: date | None = None,
        top_n: int | None = None,
        language: str | None = None,
    ) -> RunRequest:
        request_language = normalize_language(language or self.language)
        if query:
            parsed = parse_query(query)
            if top_n is not None:
                parsed = parsed.model_copy(update={"top_n": top_n})
            return parsed.model_copy(update={"language": request_language})
        if period is None:
            raise ValueError("Provide either a query string or period=...")
        if period not in get_args(Period):
            allowed = ", ".join(get_args(Period))
            raise ValueError(f"period must be one of: {allowed}")
        return RunRequest(
            period=cast(Period, period),
            year=year,
            month=month,
            day=day,
            week=week,
            start=start,
            end=end,
            top_n=top_n if top_n is not None else 10,
            language=request_language,
        )

    async def _run_async(self, request: RunRequest) -> list[PaperSummary]:
        start_date, end_date = resolve_range(request)
        timeout = self.settings.paperhub_request_timeout_s
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            primary = HFAPIFetcher(client)
            fallback = HFHtmlFetcher(client)
            papers = await papers_in_range(
                primary,
                start_date,
                end_date,
                request.top_n,
                fallback=fallback,
            )
            for meta in papers:
                self.cache.put_meta(meta)
            return await run_all(
                papers,
                self.llm,
                self.cache,
                concurrency=self.concurrency,
                max_pdf_chars=self.max_pdf_chars,
                http_client=client,
                language=request.language,
            )

    def _settings_api_key(self, provider: str) -> str | None:
        provider = (provider or "anthropic").lower()
        return {
            "anthropic": self.settings.anthropic_api_key,
            "openai": self.settings.openai_api_key,
            "google": self.settings.google_api_key,
        }.get(provider)


def _run_coro_sync(coro: Coroutine[Any, Any, list[PaperSummary]]) -> list[PaperSummary]:
    """Run a coroutine from sync code, including inside IPython/Jupyter loops."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[PaperSummary] | None = None
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            error = exc

    thread = threading.Thread(target=_runner, name="paperhub-async-runner")
    thread.start()
    thread.join()
    if error is not None:
        raise error
    assert result is not None
    return result
