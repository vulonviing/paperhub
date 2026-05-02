"""Tests for Pydantic models."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from paperhub.models import PaperMeta, PaperSummary, RunRequest


def test_paper_meta_minimum() -> None:
    meta = PaperMeta(
        arxiv_id="2026.0001",
        title=" Hello ",
        hf_url="https://huggingface.co/papers/2026.0001",
        pdf_url="https://arxiv.org/pdf/2026.0001",
    )
    assert meta.arxiv_id == "2026.0001"
    assert meta.title == "Hello"  # whitespace stripped
    assert meta.upvotes == 0
    assert meta.authors == []


def test_summary_max_chars() -> None:
    with pytest.raises(ValidationError):
        PaperSummary(
            arxiv_id="x",
            title="t",
            summary="a" * 6001,
        )


def test_summary_at_limit_ok() -> None:
    s = PaperSummary(
        arxiv_id="x",
        title="t",
        summary="a" * 6000,
    )
    assert len(s.summary) == 6000


def test_summary_accepts_legacy_summary_tr() -> None:
    s = PaperSummary.model_validate({"arxiv_id": "x", "title": "t", "summary_tr": "legacy text"})
    assert s.summary == "legacy text"
    assert s.summary_tr == "legacy text"


def test_run_request_validates_month() -> None:
    with pytest.raises(ValidationError):
        RunRequest(period="month", year=2026, month=13)


def test_run_request_top_n_bounds() -> None:
    req = RunRequest(period="month", year=2026, month=5, top_n=10)
    assert req.top_n == 10
    with pytest.raises(ValidationError):
        RunRequest(period="month", year=2026, month=5, top_n=0)


def test_run_request_language_defaults_to_english() -> None:
    req = RunRequest(period="month", year=2026, month=5)
    assert req.language == "en"


def test_run_request_language_accepts_turkish_alias() -> None:
    req = RunRequest(period="month", year=2026, month=5, language=cast(Any, "türkçe"))
    assert req.language == "tr"
