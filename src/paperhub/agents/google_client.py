"""Google Gemini LLM client (optional extra)."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


class GoogleAuthError(RuntimeError):
    """Raised when no Google API key is configured at call time."""


class GoogleClient:
    provider = "google"

    def __init__(self, *, model: str = "gemini-2.5-pro", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise GoogleAuthError(
                "GOOGLE_API_KEY is not set. Set the env var or pass api_key=... ."
            )
        try:
            from google import genai
        except Exception as exc:  # pragma: no cover - extra not installed
            raise RuntimeError(
                "Install the 'google' extra to use GoogleClient: pip install paperhub[google]"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)

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
        joined = "\n\n".join(m.get("content", "") for m in messages)
        prompt = f"{system}\n\n{joined}"

        types = None
        try:
            from google.genai import types as _types

            types = _types
        except Exception:  # pragma: no cover
            types = None

        config = None
        if types is not None:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                )
            except Exception:
                config = None

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", None)
        return (text or "").strip()
