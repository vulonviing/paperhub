"""Small live checks for provider credentials and model availability."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..utils import safe_json_extract
from .base import (
    DEFAULT_OPENAI_REASONING_EFFORT,
    LLMClient,
    build_llm,
    default_model_for_provider,
)

HEALTH_CHECK_MAX_TOKENS = 512


@dataclass(frozen=True)
class LLMHealthCheck:
    """Result of a tiny provider request."""

    provider: str
    model: str
    ok: bool
    message: str
    response_preview: str = ""


async def check_llm(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    openai_reasoning_effort: str | None = DEFAULT_OPENAI_REASONING_EFFORT,
    llm: LLMClient | None = None,
    ollama_base_url: str | None = None,
) -> LLMHealthCheck:
    """Send a tiny JSON request and verify that the selected LLM responds."""

    normalized_provider = provider.lower()
    resolved_model = model or default_model_for_provider(normalized_provider)
    is_ollama = normalized_provider == "ollama"
    expected_fields = {"motivation", "method", "findings", "real_world_examples", "summary"}

    try:
        client = llm or build_llm(
            model=resolved_model,
            provider=normalized_provider,
            api_key=api_key,
            openai_reasoning_effort=openai_reasoning_effort,
            ollama_base_url=ollama_base_url,
        )
        if is_ollama:
            system = (
                "You are a PaperHub provider health check. Return JSON only, "
                "using the required PaperHub paper-summary schema."
            )
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Return a minimal valid paper summary JSON object for a health check. "
                        "Use short non-empty strings for motivation, method, findings, and "
                        "summary, and include one real_world_examples item."
                    ),
                }
            ]
        else:
            system = "You are a PaperHub provider health check. Return JSON only."
            messages = [
                {
                    "role": "user",
                    "content": 'Return exactly {"ok": true} and no other text.',
                }
            ]
        raw = await client.complete(
            system=system,
            messages=messages,
            max_tokens=HEALTH_CHECK_MAX_TOKENS,
            temperature=0.0,
        )
        preview = " ".join(raw.split())[:180]
        body = safe_json_extract(raw)
        payload = json.loads(body)
    except Exception as exc:
        return LLMHealthCheck(
            provider=normalized_provider,
            model=resolved_model,
            ok=False,
            message=f"{exc.__class__.__name__}: {exc}",
        )

    if is_ollama and isinstance(payload, dict):
        has_summary_schema = expected_fields.issubset(payload)
        content_fields = ("motivation", "method", "findings", "summary")
        has_content = all(str(payload.get(field, "")).strip() for field in content_fields)
        examples = payload.get("real_world_examples")
        if has_summary_schema and has_content and isinstance(examples, list):
            return LLMHealthCheck(
                provider=normalized_provider,
                model=resolved_model,
                ok=True,
                message="LLM check passed.",
                response_preview=preview,
            )

    if isinstance(payload, dict) and payload.get("ok") is True:
        return LLMHealthCheck(
            provider=normalized_provider,
            model=resolved_model,
            ok=True,
            message="LLM check passed.",
            response_preview=preview,
        )
    message = (
        "Provider responded, but the health-check JSON did not match the PaperHub summary schema."
        if is_ollama
        else "Provider responded, but the health-check JSON did not contain ok=true."
    )
    return LLMHealthCheck(
        provider=normalized_provider,
        model=resolved_model,
        ok=False,
        message=message,
        response_preview=preview,
    )
