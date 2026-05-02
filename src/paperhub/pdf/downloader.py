"""Async PDF download from arXiv with on-disk caching."""

from __future__ import annotations

from pathlib import Path

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


async def download_pdf(
    arxiv_id: str,
    cache_dir: Path,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download a paper's PDF, caching it under `cache_dir`.

    Returns the path of the cached file. If the file already exists and is
    non-empty, no network call is made.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{arxiv_id.replace('/', '_')}.pdf"
    if target.exists() and target.stat().st_size > 1024:
        return target

    url = _arxiv_pdf_url(arxiv_id)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True, timeout=timeout)

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=4, min=4, max=60),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                response = await client.get(
                    url,
                    headers={"User-Agent": "PaperHub/0.1 (mailto:research@paperhub.dev)"},
                    timeout=timeout,
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                content = response.content
        target.write_bytes(content)
    finally:
        if owns_client:
            await client.aclose()

    return target
