"""LLM client protocol and provider factory.

`PaperAgent` only depends on this `LLMClient` protocol — the agent never
imports a provider SDK directly. Provider selection happens here, lazily, so
that missing optional extras (anthropic, google) never break import.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

# Providers that shell out to a locally installed, already-signed-in agent CLI
# instead of calling an HTTP API with a key. Mapped to the API provider whose
# model ids they understand.
CLI_PROVIDERS = {
    "claude-cli": "anthropic",
    "codex-cli": "openai",
}
# Executable name looked up on PATH for each CLI provider.
CLI_EXECUTABLES = {
    "claude-cli": "claude",
    "codex-cli": "codex",
}

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-5.4-mini",
    "google": "gemini-3-flash-preview",
    "ollama": "gemma4:e2b",
    # Resolved by `default_model_for_provider` (Codex's default is read from its config).
    "claude-cli": "sonnet",
    "codex-cli": "codex-cli-default",
}

# Cheaper OpenAI alternative — no reasoning budget, standard chat pricing.
OPENAI_CHEAPER_ALTERNATIVE = "gpt-4.1-mini"

DEFAULT_OPENAI_REASONING_EFFORT = "medium"
DEFAULT_PROVIDER = "openai"

# Auto-detection order when no provider is configured: installed CLIs first
# (no key needed), then providers that already have an API key.
AUTO_PROVIDER_ORDER = ("claude-cli", "codex-cli", "openai", "anthropic", "google")
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


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


def is_cli_provider(provider: str | None) -> bool:
    """True when `provider` drives a local agent CLI rather than an HTTP API."""

    return (provider or "").lower() in CLI_PROVIDERS


def cli_installed(provider: str) -> bool:
    """True when the executable behind a CLI provider is on PATH."""

    exe = CLI_EXECUTABLES.get(provider)
    return bool(exe and shutil.which(exe))


def provider_accepts_model(provider: str, inferred: str | None) -> bool:
    """Whether a model id inferred as `inferred` can be used with `provider`."""

    return inferred is None or inferred == provider or CLI_PROVIDERS.get(provider) == inferred


def detect_provider(api_keys: Mapping[str, str | None] | None = None) -> str:
    """Pick the best available provider when none was configured explicitly.

    Installed agent CLIs win because they need no API key; otherwise the first
    provider with a key; otherwise `DEFAULT_PROVIDER`.
    """

    for provider in AUTO_PROVIDER_ORDER:
        if provider in CLI_PROVIDERS:
            if cli_installed(provider):
                return provider
        elif (api_keys or {}).get(provider):
            return provider
    return DEFAULT_PROVIDER


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
    # Common Ollama model families — checked last so cloud prefixes take priority.
    if lowered.startswith(("gemma", "llama", "mistral", "qwen", "phi", "deepseek", "nomic")):
        return "ollama"
    return None


def default_model_for_provider(provider: str | None) -> str:
    """Return the built-in default model for a provider."""

    normalized = (provider or DEFAULT_PROVIDER).lower()
    if normalized in CLI_PROVIDERS:
        from .cli_models import default_cli_model

        return default_cli_model(normalized)
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
    ollama_base_url: str | None = None,
    effort: str | None = None,
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
    if chosen == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(model=resolved_model, base_url=ollama_base_url)
    if chosen in CLI_PROVIDERS:
        from .cli_client import CLIClient

        return CLIClient(provider=chosen, model=resolved_model, effort=effort)
    raise ValueError(f"unknown provider: {chosen}")
