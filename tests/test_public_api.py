"""Sanity checks on the public API: import-safety and PaperHub construction."""

from __future__ import annotations

from typing import Any, cast

import pytest


def test_import_works_without_api_keys(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    import importlib

    import paperhub

    importlib.reload(paperhub)
    assert hasattr(paperhub, "PaperHub")
    assert hasattr(paperhub, "PaperSummary")
    assert hasattr(paperhub, "PaperMeta")


def test_paperhub_with_explicit_llm(tmp_path, fake_llm) -> None:
    from paperhub import PaperHub

    llm = fake_llm([])
    hub = PaperHub(llm=llm, cache_dir=tmp_path / "cache")
    assert hub.llm is llm
    assert hub.cache.root == tmp_path / "cache"
    assert hub.language == "en"


def test_paperhub_accepts_turkish_language(tmp_path, fake_llm) -> None:
    from paperhub import PaperHub

    hub = PaperHub(llm=fake_llm([]), cache_dir=tmp_path / "cache", language="tr")
    assert hub.language == "tr"


def test_paperhub_uses_settings_api_key(tmp_path, monkeypatch) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(ANTHROPIC_API_KEY="from-env-file")

    hub = PaperHub(settings=settings, provider="anthropic", cache_dir=tmp_path / "cache")

    assert cast(Any, hub.llm)._api_key == "from-env-file"


def test_paperhub_uses_provider_default_model_when_model_omitted(tmp_path) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    settings = Settings(
        OPENAI_API_KEY="openai-key",
        PAPERHUB_MODEL="claude-haiku-4-5-20251001",
    )

    hub = PaperHub(provider="openai", settings=settings, cache_dir=tmp_path / "cache")

    assert hub.provider == "openai"
    assert hub.model == "gpt-5.4-mini"
    assert cast(Any, hub.llm).reasoning_effort == "xhigh"
    assert cast(Any, hub.llm)._api_key == "openai-key"


def test_paperhub_defaults_to_openai(tmp_path) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    settings = Settings(OPENAI_API_KEY="openai-key")

    hub = PaperHub(settings=settings, cache_dir=tmp_path / "cache")

    assert hub.provider == "openai"
    assert hub.model == "gpt-5.4-mini"
    assert cast(Any, hub.llm).reasoning_effort == "xhigh"
    assert cast(Any, hub.llm)._api_key == "openai-key"


def test_paperhub_uses_provider_specific_env_model(tmp_path) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    settings = Settings(
        OPENAI_API_KEY="openai-key",
        PAPERHUB_OPENAI_MODEL="gpt-provider-default",
    )

    hub = PaperHub(provider="openai", settings=settings, cache_dir=tmp_path / "cache")

    assert hub.model == "gpt-provider-default"


def test_openai_reasoning_effort_can_be_disabled(tmp_path) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    settings = Settings(
        OPENAI_API_KEY="openai-key",
        PAPERHUB_OPENAI_REASONING_EFFORT="",
    )

    hub = PaperHub(provider="openai", settings=settings, cache_dir=tmp_path / "cache")

    assert cast(Any, hub.llm).reasoning_effort is None


def test_paperhub_explicit_model_wins_over_provider_default(tmp_path) -> None:
    from paperhub import PaperHub
    from paperhub.config import Settings

    settings = Settings(
        OPENAI_API_KEY="openai-key",
        PAPERHUB_OPENAI_MODEL="gpt-provider-default",
    )

    hub = PaperHub(
        model="gpt-explicit",
        provider="openai",
        settings=settings,
        cache_dir=tmp_path / "cache",
    )

    assert hub.model == "gpt-explicit"


def test_build_llm_uses_provider_default_when_model_omitted() -> None:
    from paperhub import build_llm
    from paperhub.agents.base import default_model_for_provider

    assert build_llm().provider == "openai"
    assert build_llm().model == "gpt-5.4-mini"
    assert cast(Any, build_llm()).reasoning_effort == "xhigh"
    assert build_llm(provider="openai").model == "gpt-5.4-mini"
    assert default_model_for_provider("anthropic") == "claude-haiku-4-5-20251001"
    assert default_model_for_provider("google") == "gemini-3-flash-preview"


@pytest.mark.asyncio
async def test_paperhub_arun_uses_fake_llm(monkeypatch, tmp_path, fake_llm) -> None:
    from datetime import date

    from paperhub import PaperHub
    from paperhub.fetchers.range import papers_in_range as _real_range  # noqa: F401
    from paperhub.models import PaperMeta, PaperSummary

    payload = {
        "motivation": "m",
        "method": "y",
        "findings": "b",
        "real_world_examples": ["e1", "e2"],
        "summary": "English summary.",
    }
    llm = fake_llm([payload])
    hub = PaperHub(llm=llm, cache_dir=tmp_path / "cache")

    fake_papers = [
        PaperMeta(
            arxiv_id="2604.0001",
            title="Hello",
            authors=["A"],
            abstract="abs",
            upvotes=10,
            hf_url="https://huggingface.co/papers/2604.0001",
            pdf_url="https://arxiv.org/pdf/2604.0001",
            published_at=date(2026, 5, 1),
        )
    ]

    async def fake_papers_in_range(*args, **kwargs):
        return fake_papers

    monkeypatch.setattr("paperhub.papers_in_range", fake_papers_in_range, raising=False)
    monkeypatch.setattr("paperhub.fetchers.range.papers_in_range", fake_papers_in_range)

    async def fake_run(self):
        return PaperSummary(
            arxiv_id=self.paper.arxiv_id,
            title=self.paper.title,
            motivation="m",
            method="y",
            findings="b",
            real_world_examples=["e"],
            summary="English summary.",
            model_used=self.llm.model,
        )

    from paperhub.agents.paper_agent import PaperAgent

    monkeypatch.setattr(PaperAgent, "run", fake_run)

    summaries = await hub.arun(period="day", year=2026, month=5, day=1, top_n=1)
    assert len(summaries) == 1
    assert summaries[0].arxiv_id == "2604.0001"


@pytest.mark.asyncio
async def test_paperhub_run_works_inside_running_event_loop(
    monkeypatch, tmp_path, fake_llm
) -> None:
    from paperhub import PaperHub
    from paperhub.models import PaperSummary

    hub = PaperHub(llm=fake_llm([]), cache_dir=tmp_path / "cache")

    async def fake_run_async(request):
        return [
            PaperSummary(
                arxiv_id="2604.0002",
                title="Notebook Paper",
                summary="English summary.",
                model_used="fake-model",
            )
        ]

    monkeypatch.setattr(hub, "_run_async", fake_run_async)

    summaries = hub.run(period="day", year=2026, month=5, day=1, display=False)
    assert len(summaries) == 1
    assert summaries[0].arxiv_id == "2604.0002"
