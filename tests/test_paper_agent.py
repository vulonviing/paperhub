"""Per-paper agent: happy path, JSON repair, char limit, error isolation."""

from __future__ import annotations

import pytest

from paperhub.agents.paper_agent import PaperAgent


@pytest.mark.asyncio
async def test_agent_happy_path(
    monkeypatch, fake_llm, cache, sample_paper, good_summary_payload
) -> None:
    llm = fake_llm([good_summary_payload])

    async def fake_get_text(self):
        return "abundant pdf text. " * 200

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    agent = PaperAgent(sample_paper, llm, cache)
    summary = await agent.run()

    assert summary.error is None
    assert summary.motivation
    assert summary.method
    assert summary.findings
    assert len(summary.real_world_examples) >= 1
    assert 0 < len(summary.summary) <= 6000
    assert summary.model_used == "fake-model"
    assert summary.pdf_chars > 0
    assert summary.language == "en"


@pytest.mark.asyncio
async def test_agent_uses_cached_summary(
    monkeypatch, fake_llm, cache, sample_paper, good_summary_payload
) -> None:
    llm = fake_llm([good_summary_payload])

    async def fake_get_text(self):
        return "irrelevant"

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    first = await PaperAgent(sample_paper, llm, cache).run()
    assert first.error is None

    second_llm = fake_llm([])  # zero responses; would error if called
    second = await PaperAgent(sample_paper, second_llm, cache).run()
    assert second.summary == first.summary
    assert second_llm.calls == []  # cache hit, no LLM call


@pytest.mark.asyncio
async def test_agent_repairs_invalid_json(
    monkeypatch, fake_llm, cache, sample_paper, good_summary_payload
) -> None:
    llm = fake_llm(["not json", good_summary_payload])

    async def fake_get_text(self):
        return "x" * 3000

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    summary = await PaperAgent(sample_paper, llm, cache).run()
    assert summary.error is None
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_agent_trims_summary_to_6000(monkeypatch, fake_llm, cache, sample_paper) -> None:
    overlong = {
        "motivation": "m",
        "method": "y",
        "findings": "b",
        "real_world_examples": ["a", "b"],
        "summary": "Sentence. " * 2000,
    }
    llm = fake_llm([overlong])

    async def fake_get_text(self):
        return "x" * 3000

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    summary = await PaperAgent(sample_paper, llm, cache).run()
    assert summary.error is None
    assert len(summary.summary) <= 6000


@pytest.mark.asyncio
async def test_agent_isolates_llm_failure(monkeypatch, fake_llm, cache, sample_paper) -> None:
    llm = fake_llm([RuntimeError("provider down"), RuntimeError("still down")])

    async def fake_get_text(self):
        return "x" * 3000

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    summary = await PaperAgent(sample_paper, llm, cache).run()
    assert summary.error is not None
    assert "provider down" in summary.error or "still down" in summary.error
    assert summary.summary == ""


@pytest.mark.asyncio
async def test_agent_can_request_turkish_output(monkeypatch, fake_llm, cache, sample_paper) -> None:
    payload = {
        "motivation": "m",
        "method": "y",
        "findings": "b",
        "real_world_examples": ["e"],
        "summary": "Türkçe özet.",
    }
    llm = fake_llm([payload])

    async def fake_get_text(self):
        return "x" * 3000

    monkeypatch.setattr(PaperAgent, "_get_text", fake_get_text)

    summary = await PaperAgent(sample_paper, llm, cache, language="tr").run()
    assert summary.language == "tr"
    assert summary.summary == "Türkçe özet."
    assert "Türkçe" in llm.calls[0]["system"]


@pytest.mark.asyncio
async def test_agent_isolates_pdf_failure(
    monkeypatch, fake_llm, cache, sample_paper, good_summary_payload
) -> None:
    """If the PDF fails, the abstract is used and the agent still returns a summary."""

    llm = fake_llm([good_summary_payload])

    async def fail_text(self):
        return ""

    monkeypatch.setattr(PaperAgent, "_get_text", fail_text)

    summary = await PaperAgent(sample_paper, llm, cache).run()
    assert summary.error is None  # we have an abstract on sample_paper
    assert summary.pdf_chars == len(sample_paper.abstract)
