"""Fetcher tests: JSON API mapping, HTML fallback, dedupe, sorting, top_n."""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from paperhub.fetchers import HFAPIFetcher, papers_in_range
from paperhub.fetchers.hf_api import map_item
from paperhub.fetchers.hf_html import parse_html
from paperhub.models import PaperMeta

_API_ITEM = {
    "paper": {
        "id": "2604.12345",
        "title": "Self-Improving Diffusion Models",
        "authors": [{"name": "Jane Doe"}, {"name": "John Roe"}],
        "summary": "An abstract.",
        "upvotes": 99,
        "publishedAt": "2026-05-12T00:00:00Z",
    },
    "numComments": 4,
    "submittedBy": {"name": "alice"},
}


def test_map_item_happy_path() -> None:
    meta = map_item(_API_ITEM)
    assert isinstance(meta, PaperMeta)
    assert meta.arxiv_id == "2604.12345"
    assert meta.title == "Self-Improving Diffusion Models"
    assert meta.authors == ["Jane Doe", "John Roe"]
    assert meta.upvotes == 99
    assert meta.num_comments == 4
    assert meta.submitted_by == "alice"
    assert meta.published_at == date(2026, 5, 12)
    assert meta.pdf_url == "https://arxiv.org/pdf/2604.12345"


def test_map_item_missing_id_returns_none() -> None:
    bad = {"paper": {"title": "no id"}}
    assert map_item(bad) is None


def test_map_item_missing_title_returns_none() -> None:
    bad = {"paper": {"id": "2604.12345"}}
    assert map_item(bad) is None


@pytest.mark.asyncio
async def test_hf_api_fetcher_uses_mock_transport() -> None:
    handler = lambda req: httpx.Response(200, json=[_API_ITEM])  # noqa: E731
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = HFAPIFetcher(client)
        out = await fetcher.papers_for_date(date(2026, 5, 12))
    assert len(out) == 1
    assert out[0].arxiv_id == "2604.12345"


@pytest.mark.asyncio
async def test_papers_in_range_dedupes_and_sorts() -> None:
    """Same arxiv_id appearing on two days collapses to highest upvote count."""

    def make_item(idx: int, upvotes: int) -> dict:
        return {
            "paper": {
                "id": f"2604.{idx:05d}",
                "title": f"Paper {idx}",
                "authors": [{"name": "Anon"}],
                "summary": None,
                "upvotes": upvotes,
                "publishedAt": "2026-05-01T00:00:00Z",
            },
            "numComments": 0,
        }

    payloads = {
        "2026-05-01": [make_item(1, 10), make_item(2, 30)],
        "2026-05-02": [make_item(2, 50), make_item(3, 20)],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        d = request.url.params.get("date")
        return httpx.Response(200, json=payloads.get(d, []))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = HFAPIFetcher(client)
        result = await papers_in_range(
            fetcher,
            date(2026, 5, 1),
            date(2026, 5, 2),
            top_n=2,
        )
    assert len(result) == 2
    assert result[0].arxiv_id == "2604.00002"
    assert result[0].upvotes == 50
    # After dedupe: paper 1 (10), paper 2 (50), paper 3 (20). Top 2 by upvotes.
    assert result[1].arxiv_id == "2604.00003"
    assert result[1].upvotes == 20


@pytest.mark.asyncio
async def test_papers_in_range_falls_back_when_primary_fails() -> None:
    class BoomPrimary:
        async def papers_for_date(self, d, limit=100):
            raise RuntimeError("boom")

    class GoodFallback:
        async def papers_for_date(self, d, limit=100):
            return [
                PaperMeta(
                    arxiv_id="2604.99999",
                    title="Fallback paper",
                    upvotes=1,
                    hf_url="https://huggingface.co/papers/2604.99999",
                    pdf_url="https://arxiv.org/pdf/2604.99999",
                )
            ]

    out = await papers_in_range(
        BoomPrimary(), date(2026, 5, 1), date(2026, 5, 1), top_n=5, fallback=GoodFallback()
    )
    assert len(out) == 1
    assert out[0].arxiv_id == "2604.99999"


def test_html_parser_extracts_arxiv_ids() -> None:
    html = """
    <html><body>
      <a href='/papers/2604.00001'>First Paper</a>
      <a href='/papers/2604.00002'>Second Paper</a>
      <a href='/about'>Other</a>
    </body></html>
    """
    metas = parse_html(html)
    ids = {m.arxiv_id for m in metas}
    assert ids == {"2604.00001", "2604.00002"}
    titles = {m.title for m in metas}
    assert "First Paper" in titles


def test_html_parser_dedupes_repeated_anchors() -> None:
    html = """
    <a href='/papers/2604.00001'>Foo</a>
    <a href='/papers/2604.00001'>Foo</a>
    """
    assert len(parse_html(html)) == 1


@pytest.mark.asyncio
async def test_papers_in_range_respects_top_n() -> None:
    items = [
        {
            "paper": {
                "id": f"2604.{i:05d}",
                "title": f"P{i}",
                "authors": [],
                "upvotes": 100 - i,
                "publishedAt": "2026-05-01T00:00:00Z",
            },
            "numComments": 0,
        }
        for i in range(20)
    ]
    handler = lambda req: httpx.Response(200, json=items)  # noqa: E731
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        fetcher = HFAPIFetcher(client)
        out = await papers_in_range(
            fetcher,
            date(2026, 5, 1),
            date(2026, 5, 1),
            top_n=5,
        )
    assert len(out) == 5
    assert out[0].upvotes >= out[-1].upvotes
