"""PDF extraction tests, focused on the pypdf-then-pdfplumber fallback path."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperhub.pdf import extractor as pdf_extractor


def test_extract_text_missing_file_returns_empty(tmp_path: Path) -> None:
    assert pdf_extractor.extract_text(tmp_path / "missing.pdf") == ""


def test_extract_text_uses_pypdf_when_long_enough(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "x.pdf"
    target.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(pdf_extractor, "_pypdf_text", lambda p: "x" * 3000)

    def boom(p: Path) -> str:
        raise AssertionError("pdfplumber should not be called when pypdf is enough")

    monkeypatch.setattr(pdf_extractor, "_pdfplumber_text", boom)
    assert len(pdf_extractor.extract_text(target)) == 3000


def test_extract_text_falls_back_when_short(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "x.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(pdf_extractor, "_pypdf_text", lambda p: "abc")
    monkeypatch.setattr(pdf_extractor, "_pdfplumber_text", lambda p: "y" * 5000)

    out = pdf_extractor.extract_text(target, max_chars=5000)
    assert out == "y" * 5000


def test_extract_text_truncates(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "x.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(pdf_extractor, "_pypdf_text", lambda p: "z" * 9000)
    monkeypatch.setattr(pdf_extractor, "_pdfplumber_text", lambda p: "")
    out = pdf_extractor.extract_text(target, max_chars=2500)
    assert len(out) == 2500


@pytest.mark.asyncio
async def test_download_pdf_skips_when_cached(tmp_path: Path) -> None:
    from paperhub.pdf.downloader import download_pdf

    cache_dir = tmp_path / "pdfs"
    cache_dir.mkdir()
    pre = cache_dir / "2604.00001.pdf"
    pre.write_bytes(b"x" * 4096)

    # No client passed; if the function tries to fetch, this will fail since
    # network is mocked elsewhere. Cached file presence should short-circuit.
    out = await download_pdf("2604.00001", cache_dir)
    assert out == pre
    assert out.read_bytes() == b"x" * 4096
