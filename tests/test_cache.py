"""SQLite cache tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from paperhub.cache import Cache
from paperhub.models import PaperMeta, PaperSummary


def test_cache_creates_schema(tmp_path: Path) -> None:
    Cache(tmp_path / "cache")
    assert (tmp_path / "cache" / "paperhub.sqlite").exists()
    assert (tmp_path / "cache" / "pdfs").is_dir()


def test_meta_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    meta = PaperMeta(
        arxiv_id="2604.00001",
        title="Hello",
        authors=["A"],
        abstract="abs",
        upvotes=5,
        num_comments=2,
        submitted_by="bob",
        published_at=date(2026, 5, 1),
        hf_url="https://huggingface.co/papers/2604.00001",
        pdf_url="https://arxiv.org/pdf/2604.00001",
    )
    cache.put_meta(meta)
    fetched = cache.get_meta("2604.00001")
    assert fetched == meta


def test_pdf_text_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    cache.put_pdf_text("2604.00001", "hello world")
    assert cache.get_pdf_text("2604.00001") == "hello world"


def test_summary_keyed_by_model(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    s1 = PaperSummary(arxiv_id="x", title="t", model_used="m1", summary="a")
    s2 = PaperSummary(arxiv_id="x", title="t", model_used="m2", summary="b")
    cache.put_summary(s1)
    cache.put_summary(s2)
    got_m1 = cache.get_summary("x", "m1")
    got_m2 = cache.get_summary("x", "m2")
    assert got_m1 is not None
    assert got_m2 is not None
    assert got_m1.summary == "a"
    assert got_m2.summary == "b"
    assert cache.get_summary("x", "m3") is None


def test_summary_keyed_by_language(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    en = PaperSummary(arxiv_id="x", title="t", model_used="m1", language="en", summary="hello")
    tr = PaperSummary(arxiv_id="x", title="t", model_used="m1", language="tr", summary="merhaba")
    cache.put_summary(en)
    cache.put_summary(tr)
    got_en = cache.get_summary("x", "m1", "en")
    got_tr = cache.get_summary("x", "m1", "tr")
    assert got_en is not None
    assert got_tr is not None
    assert got_en.summary == "hello"
    assert got_tr.summary == "merhaba"


def test_summary_replace(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    cache.put_summary(PaperSummary(arxiv_id="x", title="t", model_used="m1", summary="a"))
    cache.put_summary(PaperSummary(arxiv_id="x", title="t", model_used="m1", summary="b"))
    got = cache.get_summary("x", "m1")
    assert got is not None
    assert got.summary == "b"
