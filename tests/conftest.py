"""Shared fixtures: a fake LLM and helper builders for paper records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pytest

from paperhub.cache import Cache
from paperhub.models import PaperMeta


class FakeLLM:
    """In-memory async LLM stand-in. Returns canned responses in order.

    `responses` may contain strings (returned verbatim) or dicts (json-encoded).
    """

    provider = "fake"

    def __init__(self, responses: Iterable[object], *, model: str = "fake-model"):
        self.model = model
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system: str,
        messages,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise RuntimeError("FakeLLM ran out of responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, dict):
            return json.dumps(nxt, ensure_ascii=False)
        return str(nxt)


@pytest.fixture
def fake_llm():
    def _make(responses):
        return FakeLLM(responses)

    return _make


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache")


@pytest.fixture
def sample_paper() -> PaperMeta:
    return PaperMeta(
        arxiv_id="2026.00001",
        title="A Test Paper",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="We test things.",
        upvotes=42,
        num_comments=3,
        submitted_by="ada",
        published_at=date(2026, 5, 1),
        hf_url="https://huggingface.co/papers/2026.00001",
        pdf_url="https://arxiv.org/pdf/2026.00001",
    )


@pytest.fixture
def good_summary_payload() -> dict:
    return {
        "motivation": "This paper addresses an important problem.",
        "method": "It uses a simple approach.",
        "findings": "The results are successful.",
        "real_world_examples": [
            "customer support automation",
            "medical image analysis",
        ],
        "summary": (
            "This paper explains the everyday impact of AI in plain language. "
            "It emphasizes the limits of current systems, describes an "
            "understandable method, and points to practical improvements."
        ),
    }
