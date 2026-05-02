"""HuggingFace JSON API fetcher (Plan A).

Endpoint: `GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`.
The API returns a JSON array of `{ paper, numComments, submittedBy, ... }`
items. We map each to a `PaperMeta`. The HTML fallback is only used if this
fetcher fails repeatedly.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import PaperMeta


class HFAPIError(RuntimeError):
    """Raised when the HuggingFace JSON API responds with an unrecoverable error."""


def _extract_arxiv_id(item: dict[str, Any]) -> str | None:
    """Best-effort extraction of an arXiv id from a daily_papers item."""

    paper = item.get("paper") or {}
    candidate = paper.get("id") or item.get("arxivId")
    if candidate:
        return str(candidate).strip()
    url = paper.get("url") or item.get("url")
    if isinstance(url, str) and "/papers/" in url:
        return url.rsplit("/papers/", 1)[-1].split("/")[0].strip()
    return None


def _parse_publish_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def map_item(item: dict[str, Any]) -> PaperMeta | None:
    """Convert a raw API item into a `PaperMeta`, or return None if invalid."""

    arxiv_id = _extract_arxiv_id(item)
    if not arxiv_id:
        return None
    paper = item.get("paper") or {}
    title = paper.get("title") or item.get("title")
    if not title:
        return None
    authors_raw = paper.get("authors") or []
    authors = [a.get("name", "").strip() for a in authors_raw if isinstance(a, dict)]
    authors = [name for name in authors if name]
    submitted = item.get("submittedBy") or paper.get("submittedBy") or {}
    submitted_by = (
        submitted.get("name")
        if isinstance(submitted, dict)
        else str(submitted)
        if submitted
        else None
    )
    upvotes = (
        paper.get("upvotes") if isinstance(paper.get("upvotes"), int) else item.get("upvotes", 0)
    )
    num_comments = item.get("numComments", paper.get("numComments", 0))
    abstract = paper.get("summary") or paper.get("abstract")
    published = _parse_publish_date(paper.get("publishedAt") or item.get("publishedAt"))
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=str(title).strip(),
        authors=authors,
        abstract=abstract.strip() if isinstance(abstract, str) else None,
        upvotes=int(upvotes or 0),
        num_comments=int(num_comments or 0),
        submitted_by=submitted_by,
        published_at=published,
        hf_url=f"https://huggingface.co/papers/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


class HFAPIFetcher:
    """Fetch HuggingFace Daily Papers via the official JSON API."""

    BASE = "https://huggingface.co/api"

    def __init__(self, client: httpx.AsyncClient, *, timeout: float = 20.0):
        self.client = client
        self.timeout = timeout

    async def papers_for_date(self, d: date, limit: int = 100) -> list[PaperMeta]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=4, min=4, max=60),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                response = await self.client.get(
                    f"{self.BASE}/daily_papers",
                    params={"date": d.isoformat(), "limit": limit},
                    timeout=self.timeout,
                    headers={"User-Agent": "PaperHub/0.1 (+https://github.com/paperhub)"},
                )
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "rate limited",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
        items = payload if isinstance(payload, list) else payload.get("papers", [])
        out: list[PaperMeta] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                meta = map_item(item)
            except Exception:
                continue
            if meta is not None:
                out.append(meta)
        return out
