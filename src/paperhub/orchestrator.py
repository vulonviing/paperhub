"""Parallel orchestrator: one `PaperAgent` per paper, bounded by a semaphore."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable

import httpx

from .agents.base import LLMClient
from .agents.paper_agent import PaperAgent
from .cache import Cache
from .models import PaperMeta, PaperSummary

ProgressCallback = Callable[[PaperSummary], Awaitable[None] | None]


async def run_all(
    papers: Iterable[PaperMeta],
    llm: LLMClient,
    cache: Cache,
    *,
    concurrency: int = 5,
    max_pdf_chars: int = 60_000,
    http_client: httpx.AsyncClient | None = None,
    on_progress: ProgressCallback | None = None,
    language: str | None = None,
) -> list[PaperSummary]:
    """Summarize each paper in parallel, returning results in input order.

    A failure in one agent never breaks the run; instead the failing agent
    returns a `PaperSummary` with `error` populated.
    """

    paper_list = list(papers)
    if not paper_list:
        return []

    sem = asyncio.Semaphore(max(1, concurrency))
    owns_client = http_client is None
    if http_client is None:
        http_client = httpx.AsyncClient(follow_redirects=True, timeout=60.0)

    async def _bounded(paper: PaperMeta) -> PaperSummary:
        async with sem:
            agent = PaperAgent(
                paper,
                llm,
                cache,
                max_pdf_chars=max_pdf_chars,
                http_client=http_client,
                language=language,
            )
            result = await agent.run()
            if on_progress is not None:
                outcome = on_progress(result)
                if asyncio.iscoroutine(outcome):
                    await outcome
            return result

    try:
        return await asyncio.gather(*[_bounded(p) for p in paper_list], return_exceptions=False)
    finally:
        if owns_client:
            await http_client.aclose()
