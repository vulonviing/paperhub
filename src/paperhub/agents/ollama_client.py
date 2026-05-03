"""Ollama LLM client (local provider).

Uses Ollama's native /api/chat endpoint with JSON schema enforcement (format
parameter). Ollama converts the schema to GBNF grammar and constrains token
sampling at inference time, so the response is always valid JSON that matches
the required schema — regardless of model size or instruction-following ability.

Falls back to the OpenAI-compatible endpoint (/v1/chat/completions) with basic
JSON mode if the native endpoint is unreachable or returns an error.

Install and start Ollama:  https://ollama.com
Pull a model:              ollama pull gemma4:e2b
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .base import DEFAULT_MODELS

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_TIMEOUT = 120.0

# JSON schema for the structured output expected from every paper summary.
# Passed as `format` to Ollama's native API — enforced at the token level.
PAPER_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "motivation": {"type": "string"},
        "method": {"type": "string"},
        "findings": {"type": "string"},
        "real_world_examples": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": ["motivation", "method", "findings", "real_world_examples", "summary"],
    "additionalProperties": False,
}


class OllamaAuthError(RuntimeError):
    """Raised when the Ollama server cannot be reached."""


class OllamaClient:
    """LLM client for local Ollama models.

    Primary path: Ollama's native /api/chat endpoint with `format` JSON schema
    enforcement (available since Ollama v0.5). This guarantees structured output
    even from small models that struggle to follow plain-language JSON rules.

    Fallback path: OpenAI-compatible /v1/chat/completions with json_object mode
    (original behaviour, kept for older Ollama versions).
    """

    provider = "ollama"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODELS["ollama"],
        base_url: str | None = None,
        timeout: float = OLLAMA_DEFAULT_TIMEOUT,
    ):
        self.model = model
        self._base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._openai_client: Any = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _native_base_url(self) -> str:
        """Return the Ollama root URL (strip /v1 if present)."""
        base = self._base_url
        if base.endswith("/v1"):
            base = base[:-3]
        return base

    def _ensure_openai_client(self) -> None:
        if self._openai_client is not None:
            return
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("The 'openai' package is required for OllamaClient.") from exc
        self._openai_client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",  # Ollama ignores the key; non-empty value required
            timeout=self._timeout,
        )

    async def _complete_native(
        self,
        msg_list: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Ollama native /api/chat with JSON schema format enforcement."""
        import httpx

        url = f"{self._native_base_url()}/api/chat"
        # Low temperature → more deterministic, consistent JSON output.
        effective_temp = min(temperature, 0.1)
        # Cap output tokens: Turkish and other non-Latin scripts tokenise into
        # ~2× more tokens than English, so a 2048-token budget can mean 5+ min
        # of generation time on a 4B model. 1200 tokens is enough for a complete
        # JSON response (~1000 chars of summary + all other fields).
        effective_max = min(max_tokens, 1200)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msg_list,
            "stream": False,
            "format": PAPER_SUMMARY_SCHEMA,
            "options": {
                "temperature": effective_temp,
                "num_predict": effective_max,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            resp = await http.post(url, json=payload)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        content: str = (data.get("message", {}).get("content", "") or "").strip()
        # Ollama native API returns raw JSON string; validate it parses correctly.
        json.loads(content)  # raises ValueError if malformed
        return content

    async def _complete_openai_compat(
        self,
        msg_list: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """OpenAI-compatible endpoint with basic JSON mode (fallback)."""
        self._ensure_openai_client()
        assert self._openai_client is not None
        request: dict[str, Any] = {
            "model": self.model,
            "messages": msg_list,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        response = await self._openai_client.chat.completions.create(**request)
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None)
        return (getattr(message, "content", "") or "").strip()

    # ------------------------------------------------------------------
    # Public interface (LLMClient protocol)
    # ------------------------------------------------------------------

    async def complete(
        self,
        *,
        system: str,
        messages: Iterable[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        msg_list: list[dict[str, str]] = [{"role": "system", "content": system}]
        for m in messages:
            msg_list.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        try:
            return await self._complete_native(msg_list, max_tokens, temperature)
        except Exception as native_exc:
            # Fall back to OpenAI-compatible endpoint (older Ollama, network quirks).
            try:
                return await self._complete_openai_compat(msg_list, max_tokens, temperature)
            except Exception as compat_exc:
                raise OllamaAuthError(
                    f"Could not reach Ollama at {self._base_url}. "
                    "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                    f"(`ollama pull {self.model}`). "
                    f"Native API error: {native_exc}; "
                    f"OpenAI-compat error: {compat_exc}"
                ) from compat_exc
