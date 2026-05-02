"""LLM client protocol and provider factory.

`PaperAgent` only depends on this `LLMClient` protocol — the agent never
imports a provider SDK directly. Provider selection happens here, lazily, so
that missing optional extras (anthropic, google) never break import.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.4-mini",
    "google": "gemini-3-flash-preview",
}
DEFAULT_OPENAI_REASONING_EFFORT = "medium"
DEFAULT_PROVIDER = "openai"


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async LLM interface used by `PaperAgent`."""

    model: str
    provider: str

    async def complete(
        self,
        *,
        system: str,
        messages: Iterable[dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Return a single string response (expected to contain JSON)."""
        ...


def infer_provider_from_model(model: str | None) -> str | None:
    """Infer a provider from common model id prefixes."""

    if not model:
        return None
    lowered = model.lower()
    if lowered.startswith(("claude", "anthropic")):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "openai", "ft:gpt")):
        return "openai"
    if lowered.startswith(("gemini", "google")):
        return "google"
    return None


def default_model_for_provider(provider: str | None) -> str:
    """Return the built-in default model for a provider."""

    normalized = (provider or DEFAULT_PROVIDER).lower()
    return DEFAULT_MODELS.get(normalized, DEFAULT_MODELS[DEFAULT_PROVIDER])


def _normalize_provider(model: str | None, provider: str | None) -> str:
    if provider:
        return provider.lower()
    return infer_provider_from_model(model) or DEFAULT_PROVIDER


def build_llm(
    model: str | None = None,
    provider: str | None = None,
    *,
    api_key: str | None = None,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
) -> LLMClient:
    """Construct the LLM client for the given model/provider.

    The actual SDK import happens inside this function so the base package
    can be imported without optional providers installed.
    """

    chosen = _normalize_provider(model, provider)
    resolved_model = model or default_model_for_provider(chosen)
    if chosen == "anthropic":
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model=resolved_model, api_key=api_key)
    if chosen == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(
            model=resolved_model,
            api_key=api_key,
            reasoning_effort=openai_reasoning_effort,
        )
    if chosen == "google":
        from .google_client import GoogleClient

        return GoogleClient(model=resolved_model, api_key=api_key)
    raise ValueError(f"unknown provider: {chosen}")
