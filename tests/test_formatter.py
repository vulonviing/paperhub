"""Formatter tests: Markdown and plain-text rendering."""

from __future__ import annotations

from paperhub.formatter import display_summaries, render_markdown, render_plain
from paperhub.models import PaperSummary, RunRequest


def _summary(arxiv_id: str = "2604.0001", error: str | None = None) -> PaperSummary:
    return PaperSummary(
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        motivation="because.",
        method="like this.",
        findings="like that.",
        real_world_examples=["example 1", "example 2"],
        summary="English summary text.",
        pdf_chars=1234,
        model_used="fake-model",
        error=error,
    )


def test_render_markdown_includes_all_papers() -> None:
    req = RunRequest(period="month", year=2026, month=5, top_n=2)
    md = render_markdown([_summary("2604.0001"), _summary("2604.0002")], req)
    assert "## 1. Paper 2604.0001" in md
    assert "## 2. Paper 2604.0002" in md
    assert "May 2026" in md
    assert "**Motivation.**" in md
    assert "https://arxiv.org/pdf/2604.0001" in md


def test_render_markdown_renders_error_block() -> None:
    req = RunRequest(period="month", year=2026, month=5, top_n=1)
    md = render_markdown([_summary("2604.0001", error="boom")], req)
    assert "Error: boom" in md


def test_render_markdown_can_render_turkish_labels() -> None:
    req = RunRequest(period="month", year=2026, month=5, top_n=1, language="tr")
    md = render_markdown([_summary("2604.0001", error="boom")], req)
    assert "Mayıs 2026" in md
    assert "Hata: boom" in md


def test_render_plain_works() -> None:
    req = RunRequest(period="day", year=2026, month=5, day=1, top_n=1)
    text = render_plain([_summary()], req)
    assert "Paper 2604.0001" in text
    assert "English summary text" in text


def test_display_summaries_returns_markdown_when_force_plain() -> None:
    req = RunRequest(period="month", year=2026, month=5, top_n=1)
    out = display_summaries([_summary()], req, force_plain=True)
    assert "## 1. Paper" in out
