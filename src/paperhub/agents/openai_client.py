"""OpenAI LLM client (default provider)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .base import DEFAULT_MODELS, DEFAULT_OPENAI_REASONING_EFFORT


class OpenAIAuthError(RuntimeError):
    """Raised when no OpenAI API key is configured at call time."""


class OpenAIClient:
    provider = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODELS["openai"],
        api_key: str | None = None,
        reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.reasoning_effort = (reasoning_effort or "").strip() or None
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise OpenAIAuthError(
                "OPENAI_API_KEY is not set. Set the env var or pass api_key=... ."
            )
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("The 'openai' package is required for OpenAIClient.") from exc
        self._client = AsyncOpenAI(api_key=self._api_key)

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
        msg_list: list[dict[str, str]] = [{"role": "system", "content": system}]
        for m in messages:
            msg_list.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        request: dict[str, Any] = {
            "model": self.model,
            "messages": msg_list,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        response = await self._client.chat.completions.create(**request)
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return (getattr(message, "content", "") or "").strip()
