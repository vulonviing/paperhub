"""Small utilities: logging, retry policy, text trimming.

Kept dependency-light; everything here works without provider extras.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_LOG = logging.getLogger("paperhub")
if not _LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _LOG.addHandler(handler)
_LOG.setLevel(logging.WARNING)


def get_logger(name: str = "paperhub") -> logging.Logger:
    """Return a configured logger; library-level default is WARNING."""

    return logging.getLogger(name)


def http_retry(*exceptions: type[BaseException]) -> AsyncRetrying:
    """Async retry policy used by HTTP-bound helpers.

    Exponential backoff: 4, 8, 16, 32 seconds, capped at 60. 4 attempts.
    """

    if not exceptions:
        exceptions = (Exception,)
    return AsyncRetrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=4, min=4, max=60),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )


_SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?…])\s+")


def trim_to_sentence_boundary(text: str, max_chars: int) -> str:
    """Cut `text` at a sentence boundary so the result is `<= max_chars`.

    Falls back to a hard cut when no boundary fits.
    """

    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    boundary_match = list(_SENTENCE_SPLIT.finditer(head))
    if boundary_match:
        cut = boundary_match[-1].end()
        if cut > max_chars * 0.5:
            return head[:cut].rstrip()
    return head.rstrip()


def safe_json_extract(raw: str) -> str:
    """Pull the first balanced JSON object out of a string.

    LLMs sometimes wrap JSON in code fences or add prose. We keep the inner
    block so callers can `json.loads` it.
    """

    if not raw:
        raise ValueError("empty LLM response")
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return fence.group(1)
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("unbalanced JSON object in response")


def coerce_str_list(value: Any) -> list[str]:
    """Best-effort: coerce mixed list/str inputs into a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value)]
