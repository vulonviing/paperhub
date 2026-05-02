"""PDF text extraction with a fast pypdf path and a pdfplumber fallback."""

from __future__ import annotations

from pathlib import Path

from ..utils import get_logger

_LOG = get_logger("paperhub.pdf")


def _pypdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - dependency missing
        _LOG.warning("pypdf unavailable: %s", exc)
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        chunks: list[str] = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks).strip()
    except Exception as exc:
        _LOG.warning("pypdf failed for %s: %s", pdf_path, exc)
        return ""


def _pdfplumber_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - dependency missing
        _LOG.warning("pdfplumber unavailable: %s", exc)
        return ""
    try:
        chunks: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(chunks).strip()
    except Exception as exc:
        _LOG.warning("pdfplumber failed for %s: %s", pdf_path, exc)
        return ""


def extract_text(pdf_path: Path | str, max_chars: int = 60_000) -> str:
    """Return PDF text up to `max_chars`.

    Strategy: pypdf first (fast), pdfplumber fallback when the first cut is
    suspiciously short. Empty result triggers no exceptions — callers may
    fall back to abstract+title.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return ""

    text = _pypdf_text(pdf_path)
    if len(text) < 2000:
        plumber = _pdfplumber_text(pdf_path)
        if len(plumber) > len(text):
            text = plumber

    if max_chars and len(text) > max_chars:
        return text[:max_chars]
    return text
