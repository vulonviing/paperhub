"""Pydantic data models for PaperHub.

`PaperMeta` describes a paper as listed on HuggingFace Daily Papers.
`PaperSummary` is the AI-generated summary, one per paper.
`RunRequest` is the parsed user request: a date period and a `top_n`.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .localization import OutputLanguage, normalize_language

SUMMARY_MAX_CHARS = 6000

Period = Literal["day", "week", "month", "year", "custom"]


class PaperMeta(BaseModel):
    """Metadata for a HuggingFace Daily Paper entry."""

    model_config = ConfigDict(str_strip_whitespace=True)

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    upvotes: int = 0
    num_comments: int = 0
    submitted_by: str | None = None
    published_at: date | None = None
    hf_url: str
    pdf_url: str

    @field_validator("arxiv_id")
    @classmethod
    def _strip_arxiv_id(cls, value: str) -> str:
        return value.strip()


class PaperSummary(BaseModel):
    """Structured summary of a paper."""

    model_config = ConfigDict(str_strip_whitespace=False, extra="ignore")

    arxiv_id: str
    title: str
    summary: str = Field(default="", max_length=SUMMARY_MAX_CHARS)
    motivation: str = ""
    method: str = ""
    findings: str = ""
    real_world_examples: list[str] = Field(default_factory=list)
    language: OutputLanguage = "en"
    pdf_chars: int = 0
    model_used: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_summary_tr(cls, data):
        if isinstance(data, dict) and "summary" not in data and "summary_tr" in data:
            return {**data, "summary": data["summary_tr"]}
        return data

    @field_validator("summary")
    @classmethod
    def _enforce_max_chars(cls, value: str) -> str:
        if len(value) > SUMMARY_MAX_CHARS:
            raise ValueError(f"summary exceeds {SUMMARY_MAX_CHARS} characters (got {len(value)})")
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: str | None) -> OutputLanguage:
        return normalize_language(value)

    @property
    def summary_tr(self) -> str:
        """Backward-compatible alias for the pre-0.2 Turkish summary field."""

        return self.summary

    @summary_tr.setter
    def summary_tr(self, value: str) -> None:
        self.summary = value


class RunRequest(BaseModel):
    """A parsed user request describing which papers to summarize."""

    model_config = ConfigDict(str_strip_whitespace=True)

    period: Period
    year: int | None = None
    month: int | None = None
    day: int | None = None
    week: int | None = None
    start: date | None = None
    end: date | None = None
    top_n: int = Field(default=10, ge=1, le=200)
    language: OutputLanguage = "en"

    @field_validator("month")
    @classmethod
    def _validate_month(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 12:
            raise ValueError("month must be between 1 and 12")
        return value

    @field_validator("day")
    @classmethod
    def _validate_day(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 31:
            raise ValueError("day must be between 1 and 31")
        return value

    @field_validator("week")
    @classmethod
    def _validate_week(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 53:
            raise ValueError("week must be between 1 and 53")
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: str | None) -> OutputLanguage:
        return normalize_language(value)
