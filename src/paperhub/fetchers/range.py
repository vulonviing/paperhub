"""Range aggregation: walk a date range, dedupe, sort by upvotes, return top_n.

The function takes a primary fetcher and an optional fallback. If the primary
errors for a given day, it tries the fallback for that day; if both fail the
day is skipped (logged) — we do not stop the run for one bad day.
"""

from __future__ import annotations

import asyncio
from datetime import date

from ..dates import iter_dates
from ..models import PaperMeta
from ..utils import get_logger
from .base import Fetcher

_LOG = get_logger("paperhub.fetchers")


async def _safe_fetch(fetcher: Fetcher | None, d: date, limit: int) -> list[PaperMeta] | None:
    if fetcher is None:
        return None
    try:
        return await fetcher.papers_for_date(d, limit=limit)
    except Exception as exc:  # pragma: no cover - logged path
        _LOG.warning("fetch failed for %s: %s", d.isoformat(), exc)
        return None


async def papers_in_range(
    primary: Fetcher,
    start: date,
    end: date,
    top_n: int,
    *,
    fallback: Fetcher | None = None,
    per_day_limit: int = 100,
) -> list[PaperMeta]:
    """Aggregate papers across `[start, end]`, dedupe, sort, slice to `top_n`."""

    days = iter_dates(start, end)
    results = await asyncio.gather(*[_safe_fetch(primary, d, per_day_limit) for d in days])

    fallback_indices = [idx for idx, value in enumerate(results) if value is None]
    if fallback_indices and fallback is not None:
        fallback_results = await asyncio.gather(
            *[_safe_fetch(fallback, days[idx], per_day_limit) for idx in fallback_indices]
        )
        for idx, value in zip(fallback_indices, fallback_results, strict=True):
            results[idx] = value

    flat: list[PaperMeta] = []
    for batch in results:
        if not batch:
            continue
        flat.extend(batch)

    deduped: dict[str, PaperMeta] = {}
    for meta in flat:
        existing = deduped.get(meta.arxiv_id)
        if existing is None or meta.upvotes > existing.upvotes:
            deduped[meta.arxiv_id] = meta

    sorted_papers = sorted(
        deduped.values(),
        key=lambda p: (p.upvotes, p.num_comments, p.title),
        reverse=True,
    )
    return sorted_papers[:top_n]
