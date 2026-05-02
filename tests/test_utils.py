"""Utility tests: JSON extraction and sentence-boundary trimming."""

from __future__ import annotations

import pytest

from paperhub.utils import coerce_str_list, safe_json_extract, trim_to_sentence_boundary


def test_safe_json_extract_plain() -> None:
    assert safe_json_extract('{"a":1}') == '{"a":1}'


def test_safe_json_extract_with_fence() -> None:
    raw = '```json\n{"a":1}\n```'
    assert safe_json_extract(raw) == '{"a":1}'


def test_safe_json_extract_with_prefix_text() -> None:
    raw = 'Here is the JSON:\n{"a": {"b": 2}}\nThanks.'
    assert safe_json_extract(raw) == '{"a": {"b": 2}}'


def test_safe_json_extract_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        safe_json_extract("nothing useful")


def test_trim_to_sentence_boundary_short_unchanged() -> None:
    assert trim_to_sentence_boundary("Kısa.", 100) == "Kısa."


def test_trim_to_sentence_boundary_cuts_at_sentence() -> None:
    text = "Birinci cümle. " + "İkinci cümle çok uzun olabilir. " * 20
    out = trim_to_sentence_boundary(text, 80)
    assert len(out) <= 80
    assert out.endswith(".") or out.endswith("…")


def test_coerce_str_list_handles_mixed() -> None:
    assert coerce_str_list(["a", " b ", "", None]) == ["a", "b"]
    assert coerce_str_list("only") == ["only"]
    assert coerce_str_list(None) == []
