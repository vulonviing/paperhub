"""OpenAI LLM client (default provider)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .base import DEFAULT_MODELS, DEFAULT_OPENAI_REASONING_EFFORT

OPENAI_REASONING_MIN_COMPLETION_TOKENS = 25000


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
            "max_completion_tokens": _completion_token_budget(self.model, max_tokens),
            "response_format": {"type": "json_object"},
        }
        if _supports_reasoning_effort(self.model) and self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if _supports_custom_temperature(self.model):
            request["temperature"] = temperature

        response = await self._client.chat.completions.create(**request)
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = (getattr(message, "content", "") or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise RuntimeError(
                f"OpenAI returned empty content (finish_reason={finish_reason}). "
                "Try a lower reasoning effort or a model with a larger completion budget."
            )
        return content


def _supports_reasoning_effort(model: str) -> bool:
    """Return whether a Chat Completions model supports reasoning_effort."""

    lowered = model.lower()
    return lowered.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_custom_temperature(model: str) -> bool:
    """Return whether PaperHub should send a custom temperature value."""

    return not _supports_reasoning_effort(model)


def _completion_token_budget(model: str, requested: int) -> int:
    """Give reasoning models enough budget for hidden reasoning plus JSON output."""

    if _supports_reasoning_effort(model):
        return max(requested, OPENAI_REASONING_MIN_COMPLETION_TOKENS)
    return requested
