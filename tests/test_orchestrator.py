"""Orchestrator tests: concurrency limits and error isolation."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from paperhub.agents.paper_agent import PaperAgent
from paperhub.models import PaperMeta, PaperSummary
from paperhub.orchestrator import run_all


def _make_paper(idx: int) -> PaperMeta:
    return PaperMeta(
        arxiv_id=f"2604.{idx:05d}",
        title=f"Paper {idx}",
        upvotes=10 - idx,
        hf_url=f"https://huggingface.co/papers/2604.{idx:05d}",
        pdf_url=f"https://arxiv.org/pdf/2604.{idx:05d}",
        published_at=date(2026, 5, 1),
    )


@pytest.mark.asyncio
async def test_orchestrator_runs_all_in_order(monkeypatch, cache, fake_llm) -> None:
    papers = [_make_paper(i) for i in range(5)]
    payload = {
        "motivation": "m",
        "method": "y",
        "findings": "b",
        "real_world_examples": ["a"],
        "summary": "short summary.",
    }
    llm = fake_llm([payload] * len(papers))

    async def fake_run(self):
        return PaperSummary(
            arxiv_id=self.paper.arxiv_id,
            title=self.paper.title,
            summary="ok",
            motivation="m",
            method="y",
            findings="b",
            real_world_examples=["e"],
            pdf_chars=10,
            model_used=self.llm.model,
        )

    monkeypatch.setattr(PaperAgent, "run", fake_run)

    out = await run_all(papers, llm, cache, concurrency=3)
    assert [s.arxiv_id for s in out] == [p.arxiv_id for p in papers]


@pytest.mark.asyncio
async def test_orchestrator_isolates_failure(monkeypatch, cache, fake_llm) -> None:
    papers = [_make_paper(i) for i in range(4)]
    llm = fake_llm([])

    async def maybe_run(self):
        if self.paper.arxiv_id.endswith("00002"):
            return PaperSummary(
                arxiv_id=self.paper.arxiv_id,
                title=self.paper.title,
                summary="",
                error="PDF download 404",
                model_used=self.llm.model,
            )
        return PaperSummary(
            arxiv_id=self.paper.arxiv_id,
            title=self.paper.title,
            summary="good summary.",
            motivation="m",
            method="y",
            findings="b",
            real_world_examples=["e"],
            model_used=self.llm.model,
        )

    monkeypatch.setattr(PaperAgent, "run", maybe_run)

    out = await run_all(papers, llm, cache, concurrency=2)
    assert len(out) == 4
    errors = [s for s in out if s.error]
    assert len(errors) == 1
    assert errors[0].arxiv_id.endswith("00002")
    okay = [s for s in out if not s.error]
    assert len(okay) == 3


@pytest.mark.asyncio
async def test_orchestrator_respects_concurrency(monkeypatch, cache, fake_llm) -> None:
    papers = [_make_paper(i) for i in range(10)]
    llm = fake_llm([])
    concurrent = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow_run(self):
        nonlocal concurrent, peak
        async with lock:
            concurrent += 1
            peak = max(peak, concurrent)
        try:
            await asyncio.sleep(0.01)
            return PaperSummary(
                arxiv_id=self.paper.arxiv_id,
                title=self.paper.title,
                summary="ok",
                motivation="m",
                method="y",
                findings="b",
                real_world_examples=["e"],
                model_used=self.llm.model,
            )
        finally:
            async with lock:
                concurrent -= 1

    monkeypatch.setattr(PaperAgent, "run", slow_run)

    await run_all(papers, llm, cache, concurrency=3)
    assert peak <= 3


@pytest.mark.asyncio
async def test_orchestrator_empty_input(cache, fake_llm) -> None:
    llm = fake_llm([])
    assert await run_all([], llm, cache) == []
