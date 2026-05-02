"""Application settings, sourced from environment variables.

Importing this module must never require any provider key. Missing keys are
allowed at construction time; clients only fail when they actually try to call
a remote service.
"""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agents.base import (
    DEFAULT_MODELS,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_PROVIDER,
    default_model_for_provider,
    infer_provider_from_model,
)


class Settings(BaseSettings):
    """Environment-backed configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    paperhub_provider: str = Field(default=DEFAULT_PROVIDER, alias="PAPERHUB_PROVIDER")
    paperhub_model: str | None = Field(default=None, alias="PAPERHUB_MODEL")
    paperhub_anthropic_model: str = Field(
        default=DEFAULT_MODELS["anthropic"],
        alias="PAPERHUB_ANTHROPIC_MODEL",
    )
    paperhub_openai_model: str = Field(
        default=DEFAULT_MODELS["openai"],
        alias="PAPERHUB_OPENAI_MODEL",
    )
    paperhub_openai_reasoning_effort: str | None = Field(
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        alias="PAPERHUB_OPENAI_REASONING_EFFORT",
    )
    paperhub_google_model: str = Field(
        default=DEFAULT_MODELS["google"],
        alias="PAPERHUB_GOOGLE_MODEL",
    )
    paperhub_concurrency: int = Field(default=5, alias="PAPERHUB_CONCURRENCY")
    paperhub_max_pdf_chars: int = Field(default=60_000, alias="PAPERHUB_MAX_PDF_CHARS")
    paperhub_cache_dir: str | None = Field(default=None, alias="PAPERHUB_CACHE_DIR")
    paperhub_request_timeout_s: float = Field(default=30.0, alias="PAPERHUB_REQUEST_TIMEOUT_S")

    def cache_dir(self) -> Path:
        """Return the path used for caching, creating it if missing."""
        if self.paperhub_cache_dir:
            path = Path(self.paperhub_cache_dir).expanduser()
        else:
            path = user_cache_path("paperhub", appauthor="paperhub")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_for_provider(self, provider: str | None = None) -> str:
        """Return the configured default model for a provider.

        `PAPERHUB_MODEL` is kept as a global override, but if it clearly
        belongs to a different provider we ignore it and use the selected
        provider's own default. This prevents, for example, sending a Claude
        model id to OpenAI when `PAPERHUB_PROVIDER=openai`.
        """

        chosen = (provider or self.paperhub_provider or DEFAULT_PROVIDER).lower()
        if self.paperhub_model:
            inferred = infer_provider_from_model(self.paperhub_model)
            if inferred is None or inferred == chosen:
                return self.paperhub_model
        provider_specific = {
            "anthropic": self.paperhub_anthropic_model,
            "openai": self.paperhub_openai_model,
            "google": self.paperhub_google_model,
        }.get(chosen)
        return provider_specific or default_model_for_provider(chosen)

    def openai_reasoning_effort(self) -> str | None:
        """Return the configured OpenAI reasoning effort, or None when disabled."""

        if self.paperhub_openai_reasoning_effort is None:
            return None
        value = self.paperhub_openai_reasoning_effort.strip()
        return value or None


def load_settings() -> Settings:
    """Build a `Settings` instance from the current environment."""

    return Settings()
