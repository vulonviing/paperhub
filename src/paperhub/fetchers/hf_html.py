"""HuggingFace HTML fallback (Plan B).

Used when the JSON API misbehaves. Mirrors the selectors used by the
`huggingface-paper-explorer` Next.js project, but in async Python.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..models import PaperMeta


def _parse_int(text: str | None) -> int:
    if not text:
        return 0
    digits = re.findall(r"\d+", text)
    if not digits:
        return 0
    try:
        return int("".join(digits))
    except ValueError:
        return 0


def _extract_arxiv_id(href: str) -> str | None:
    if not href:
        return None
    href = href.rstrip("/")
    if "/papers/" in href:
        return href.rsplit("/papers/", 1)[-1].split("?")[0].split("#")[0]
    return None


def parse_html(html: str) -> list[PaperMeta]:
    """Parse a HuggingFace daily papers page into a list of `PaperMeta`."""

    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    out: list[PaperMeta] = []

    for anchor in soup.select("a[href*='/papers/']"):
        href_attr = anchor.get("href") or ""
        href = href_attr[0] if isinstance(href_attr, list) else str(href_attr)
        arxiv_id = _extract_arxiv_id(href)
        if not arxiv_id or arxiv_id in seen:
            continue
        title_text = anchor.get_text(strip=True)
        if not title_text:
            heading = anchor.find(["h3", "h2"])
            title_text = heading.get_text(strip=True) if heading else ""
        if not title_text:
            continue

        # walk up to a card container to find upvote/comment counts
        card = anchor
        for _ in range(6):
            if card.parent is None:
                break
            card = card.parent

        upvotes = 0
        num_comments = 0
        submitted_by: str | None = None

        upvote_node = card.select_one("[class*='upvote'], [data-test*='upvote'], button + div")
        if upvote_node is not None:
            upvotes = _parse_int(upvote_node.get_text(" ", strip=True))

        comment_link = card.select_one("a[href*='#community']")
        if comment_link is not None:
            num_comments = _parse_int(comment_link.get_text(" ", strip=True))

        submitter_node = card.select_one(".pointer-events-none, [class*='submitted']")
        if submitter_node is not None:
            submitted_by = submitter_node.get_text(" ", strip=True) or None

        seen.add(arxiv_id)
        out.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=title_text,
                upvotes=upvotes,
                num_comments=num_comments,
                submitted_by=submitted_by,
                hf_url=f"https://huggingface.co/papers/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
    return out


class HFHtmlFetcher:
    """Fetch HuggingFace Daily Papers via HTML scraping (fallback)."""

    BASE = "https://huggingface.co/papers"

    def __init__(self, client: httpx.AsyncClient, *, timeout: float = 30.0):
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
                    self.BASE,
                    params={"date": d.isoformat()},
                    timeout=self.timeout,
                    headers={
                        "User-Agent": "PaperHub/0.1 (+https://github.com/paperhub)",
                        "Accept": "text/html",
                    },
                    follow_redirects=True,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                html = response.text
        return parse_html(html)[:limit] if limit else parse_html(html)


def _to_meta_safe(item: Any) -> PaperMeta | None:
    if isinstance(item, PaperMeta):
        return item
    return None
