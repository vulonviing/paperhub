"""Provider health-check behavior."""

from __future__ import annotations

import pytest

from paperhub.agents.health import check_llm


@pytest.mark.asyncio
async def test_check_llm_passes_when_provider_returns_ok_json(fake_llm) -> None:
    result = await check_llm("openai", llm=fake_llm([{"ok": True}]))

    assert result.ok is True
    assert result.provider == "openai"
    assert result.message == "LLM check passed."


@pytest.mark.asyncio
async def test_check_llm_fails_on_empty_response(fake_llm) -> None:
    result = await check_llm("openai", llm=fake_llm([""]))

    assert result.ok is False
    assert "empty LLM response" in result.message


@pytest.mark.asyncio
async def test_check_llm_fails_when_json_does_not_confirm_ok(fake_llm) -> None:
    result = await check_llm("openai", llm=fake_llm([{"ok": False}]))

    assert result.ok is False
    assert "ok=true" in result.message
