"""Application settings, sourced from environment variables.

Importing this module must never require any provider key. Missing keys are
allowed at construction time; clients only fail when they actually try to call
a remote service.
"""

from __future__ import annotations

import os
import re
from contextlib import suppress
from pathlib import Path

from platformdirs import user_cache_path, user_config_path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agents.base import (
    DEFAULT_MODELS,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_PROVIDER,
    default_model_for_provider,
    infer_provider_from_model,
)

APP_NAME = "paperhub"
APP_AUTHOR = "paperhub"
USER_CONFIG_FILENAME = ".env"


def user_config_dir(*, create: bool = False) -> Path:
    """Return PaperHub's per-user config directory."""

    path = user_config_path(APP_NAME, appauthor=APP_AUTHOR)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def user_config_env_path(*, create_parent: bool = False) -> Path:
    """Return the PaperHub-specific dotenv file path."""

    return user_config_dir(create=create_parent) / USER_CONFIG_FILENAME


def read_user_config_values(path: Path | None = None) -> dict[str, str]:
    """Read PaperHub's user config dotenv file."""

    config_path = path or user_config_env_path()
    if not config_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        values[key] = _unquote_dotenv_value(value.strip())
    return values


def save_user_config_value(key: str, value: str, path: Path | None = None) -> Path:
    """Write one value into PaperHub's user config dotenv file."""

    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
        raise ValueError(f"invalid config key: {key}")
    if "\n" in value or "\r" in value:
        raise ValueError("config values must be single-line strings")

    config_path = path or user_config_env_path(create_parent=True)
    values = read_user_config_values(config_path)
    values[key] = value.strip()

    lines = [
        "# PaperHub user config. Do not commit this file.",
        "# Managed by `paperhub set-key` and the interactive `/set-key` command.",
        "",
    ]
    for existing_key in sorted(values):
        lines.append(f"{existing_key}={_quote_dotenv_value(values[existing_key])}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with suppress(OSError):
        os.chmod(config_path, 0o600)
    return config_path


def _quote_dotenv_value(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@+\-]*", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _unquote_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


class Settings(BaseSettings):
    """Environment-backed configuration."""

    model_config = SettingsConfigDict(
        env_file=str(user_config_env_path()),
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
            path = user_cache_path(APP_NAME, appauthor=APP_AUTHOR)
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
