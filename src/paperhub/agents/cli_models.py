"""Model and effort metadata for the agent-CLI providers (claude-cli, codex-cli).

Kept free of imports from the rest of the package so both `base` and
`cli_client` can use it. Everything here is best-effort: unreadable config
simply yields "no information" and callers fall back to free text.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Aliases documented by `claude --help`; the CLI resolves them to the latest model.
CLAUDE_MODEL_ALIASES = ("fable", "opus", "sonnet")
# Cheap and fast, matching the API default (claude-sonnet-4-6) in spirit.
CLAUDE_DEFAULT_MODEL = "sonnet"
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Used when Codex does not list per-model levels.
CODEX_FALLBACK_EFFORT_LEVELS = ("low", "medium", "high", "xhigh")
# Codex also offers "ultra", which auto-delegates to sub-agents; that is far
# too expensive and unpredictable for one-paper summaries, so never offer it.
CODEX_HIDDEN_EFFORT_LEVELS = frozenset({"ultra"})

# What PaperHub uses unless the user picks otherwise.
DEFAULT_CLI_EFFORT = "medium"

# Placeholder meaning "let the CLI choose"; only used when Codex's own config
# cannot be read, so we cannot name a concrete model.
UNKNOWN_CODEX_MODEL = "codex-cli-default"


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _codex_models() -> list[dict]:
    try:
        data = json.loads((_codex_home() / "models_cache.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("models", []) if isinstance(data, dict) else []
    return [m for m in entries if isinstance(m, dict) and m.get("slug")]


def list_cli_models(provider: str) -> list[str]:
    """Return model ids worth offering in a picker for a CLI provider.

    Codex keeps the models available to the signed-in account in
    `$CODEX_HOME/models_cache.json`; when that file is missing or unreadable
    the list is empty and callers fall back to free text.
    """

    if provider == "claude-cli":
        return list(CLAUDE_MODEL_ALIASES)
    if provider == "codex-cli":
        return [m["slug"] for m in _codex_models() if m.get("visibility") == "list"]
    return []


def _codex_configured_model() -> str | None:
    try:
        text = (_codex_home() / "config.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^model\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def default_cli_model(provider: str) -> str:
    """Concrete default model for a CLI provider, so the UI never says "default"."""

    if provider == "claude-cli":
        return CLAUDE_DEFAULT_MODEL
    if provider == "codex-cli":
        listed = list_cli_models(provider)
        return _codex_configured_model() or (listed[0] if listed else UNKNOWN_CODEX_MODEL)
    raise ValueError(f"not a CLI provider: {provider}")


def is_unknown_model(model: str | None) -> bool:
    """True when `model` is the placeholder for "no concrete model known"."""

    return not model or model == UNKNOWN_CODEX_MODEL


def cli_effort_levels(provider: str, model: str | None = None) -> list[str]:
    """Reasoning-effort levels a CLI provider accepts for `model`."""

    if provider == "claude-cli":
        return list(CLAUDE_EFFORT_LEVELS)
    if provider == "codex-cli":
        for entry in _codex_models():
            if entry["slug"] == model:
                levels = [
                    str(lvl["effort"])
                    for lvl in entry.get("supported_reasoning_levels", [])
                    if isinstance(lvl, dict)
                    and lvl.get("effort")
                    and lvl["effort"] not in CODEX_HIDDEN_EFFORT_LEVELS
                ]
                if levels:
                    return levels
        return list(CODEX_FALLBACK_EFFORT_LEVELS)
    return []
