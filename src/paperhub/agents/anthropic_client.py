"""Anthropic LLM client (optional provider).

Uses the official `anthropic` SDK. Imported lazily by `build_llm` so this file
is only loaded when the user actually picks Anthropic.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .base import DEFAULT_MODELS


class AnthropicAuthError(RuntimeError):
    """Raised when no Anthropic API key is configured at call time."""


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, *, model: str = DEFAULT_MODELS["anthropic"], api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise AnthropicAuthError(
                "ANTHROPIC_API_KEY is not set. Set the env var or pass api_key=... ."
            )
        try:
            from anthropic import AsyncAnthropic
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("The 'anthropic' package is required for AnthropicClient.") from exc
        self._client = AsyncAnthropic(api_key=self._api_key)

    async def complete(
        self,
        *,
        system: str,
        messages: Iterable[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        self._ensure_client()
        assert self._client is not None
        msg_list = [
            {"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages
        ]
        response = await self._client.messages.create(
            model=self.model,
            system=system,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()
