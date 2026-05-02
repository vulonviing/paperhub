"""Per-paper agent: PDF → LLM → validated `PaperSummary`.

One instance per paper. Errors are caught and converted into a `PaperSummary`
with `error` set, so a single failed paper never poisons the orchestrator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from ..cache import Cache
from ..localization import OutputLanguage, normalize_language
from ..models import PaperMeta, PaperSummary
from ..pdf import download_pdf, extract_text
from ..utils import (
    coerce_str_list,
    get_logger,
    safe_json_extract,
    trim_to_sentence_boundary,
)
from .base import LLMClient
from .prompts import (
    SUMMARY_MAX_CHARS,
    build_messages,
    repair_instruction,
    system_prompt,
)

_LOG = get_logger("paperhub.agent")


class PaperAgent:
    """Summarize a single paper using an `LLMClient` and a local cache."""

    def __init__(
        self,
        paper: PaperMeta,
        llm: LLMClient,
        cache: Cache,
        *,
        max_pdf_chars: int = 60_000,
        http_client: httpx.AsyncClient | None = None,
        language: str | None = None,
    ):
        self.paper = paper
        self.llm = llm
        self.cache = cache
        self.max_pdf_chars = max_pdf_chars
        self.http_client = http_client
        self.language: OutputLanguage = normalize_language(language)

    async def run(self) -> PaperSummary:
        t0 = time.perf_counter()
        try:
            cached = self.cache.get_summary(self.paper.arxiv_id, self.llm.model, self.language)
            if cached is not None:
                return cached

            text = await self._get_text()
            if not text and self.paper.abstract:
                text = self.paper.abstract

            payload = await self._summarize(text)
            summary = self._build_summary(payload, text, t0)
            summary = self._enforce_limits(summary, text, t0, payload)
            self.cache.put_summary(summary)
            return summary
        except Exception as exc:
            _LOG.warning("agent failed for %s: %s", self.paper.arxiv_id, exc)
            return PaperSummary(
                arxiv_id=self.paper.arxiv_id,
                title=self.paper.title,
                summary="",
                motivation="",
                method="",
                findings="",
                real_world_examples=[],
                language=self.language,
                pdf_chars=0,
                model_used=getattr(self.llm, "model", ""),
                elapsed_s=time.perf_counter() - t0,
                error=str(exc) or exc.__class__.__name__,
            )

    async def _get_text(self) -> str:
        cached_text = self.cache.get_pdf_text(self.paper.arxiv_id)
        if cached_text:
            return cached_text[: self.max_pdf_chars]
        try:
            pdf_path = await download_pdf(
                self.paper.arxiv_id,
                self.cache.pdf_dir,
                client=self.http_client,
            )
        except Exception as exc:
            _LOG.warning("pdf download failed for %s: %s", self.paper.arxiv_id, exc)
            return ""
        text = extract_text(pdf_path, self.max_pdf_chars)
        if text:
            self.cache.put_pdf_text(self.paper.arxiv_id, text)
        return text

    async def _summarize(self, text: str) -> dict:
        messages = build_messages(
            self.paper,
            text,
            max_pdf_chars=self.max_pdf_chars,
            language=self.language,
        )
        raw = await self.llm.complete(
            system=system_prompt(self.language),
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
        )
        try:
            return self._parse_json(raw)
        except Exception as exc:
            messages = list(messages) + [
                {
                    "role": "user",
                    "content": repair_instruction(self.language).format(
                        issue=self._repair_issue(exc)
                    ),
                }
            ]
            raw = await self.llm.complete(
                system=system_prompt(self.language),
                messages=messages,
                max_tokens=2048,
                temperature=0.2,
            )
            return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        body = safe_json_extract(raw)
        data = json.loads(body)
        if not isinstance(data, dict):
            raise ValueError("LLM JSON is not an object")
        return data

    def _build_summary(self, payload: dict, text: str, t0: float) -> PaperSummary:
        summary_text = payload.get("summary", payload.get("summary_tr", ""))
        return PaperSummary(
            arxiv_id=self.paper.arxiv_id,
            title=self.paper.title,
            motivation=str(payload.get("motivation", "")).strip(),
            method=str(payload.get("method", "")).strip(),
            findings=str(payload.get("findings", "")).strip(),
            real_world_examples=coerce_str_list(payload.get("real_world_examples")),
            summary=trim_to_sentence_boundary(str(summary_text).strip(), SUMMARY_MAX_CHARS),
            language=self.language,
            pdf_chars=len(text),
            model_used=self.llm.model,
            elapsed_s=time.perf_counter() - t0,
        )

    def _enforce_limits(
        self,
        summary: PaperSummary,
        text: str,
        t0: float,
        payload: dict,
    ) -> PaperSummary:
        if len(summary.summary) <= SUMMARY_MAX_CHARS and summary.summary:
            return summary
        # Already trimmed in `_build_summary`, but defend against any drift.
        summary.summary = trim_to_sentence_boundary(
            summary.summary or str(payload.get("summary", payload.get("summary_tr", ""))),
            SUMMARY_MAX_CHARS,
        )
        summary.elapsed_s = time.perf_counter() - t0
        return summary

    def _repair_issue(self, exc: Exception) -> str:
        if self.language == "tr":
            return f"geçersiz JSON üretti ({exc})"
        return f"produced invalid JSON ({exc})"

    @staticmethod
    def cache_pdf_path(cache: Cache, arxiv_id: str) -> Path:
        return cache.pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
