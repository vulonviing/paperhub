"""OpenAI client request-shaping tests."""

from __future__ import annotations

from typing import Any

import pytest

from paperhub.agents.openai_client import OPENAI_REASONING_MIN_COMPLETION_TOKENS, OpenAIClient


class _Message:
    def __init__(self, content: str = '{"summary": "ok"}') -> None:
        self.content = content


class _Choice:
    def __init__(
        self,
        content: str = '{"summary": "ok"}',
        finish_reason: str | None = "stop",
    ) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, choice: _Choice | None = None) -> None:
        self.choices = [choice if choice is not None else _Choice()]


class _Completions:
    def __init__(self, response: _Response | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._response = response or _Response()

    async def create(self, **kwargs: Any) -> _Response:
        self.requests.append(kwargs)
        return self._response


class _Chat:
    def __init__(self, response: _Response | None = None) -> None:
        self.completions = _Completions(response)


class _OpenAIStub:
    def __init__(self, response: _Response | None = None) -> None:
        self.chat = _Chat(response)


@pytest.mark.asyncio
async def test_reasoning_models_omit_custom_temperature() -> None:
    stub = _OpenAIStub()
    client = OpenAIClient(model="gpt-5.4-mini", api_key="test-key")
    client._client = stub

    await client.complete(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.3,
    )

    request = stub.chat.completions.requests[0]
    assert request["model"] == "gpt-5.4-mini"
    assert request["reasoning_effort"] == "medium"
    assert request["max_completion_tokens"] == OPENAI_REASONING_MIN_COMPLETION_TOKENS
    assert "temperature" not in request


@pytest.mark.asyncio
async def test_non_reasoning_models_keep_temperature_and_skip_reasoning_effort() -> None:
    stub = _OpenAIStub()
    client = OpenAIClient(model="gpt-4o-mini", api_key="test-key")
    client._client = stub

    await client.complete(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.3,
    )

    request = stub.chat.completions.requests[0]
    assert request["model"] == "gpt-4o-mini"
    assert request["max_completion_tokens"] == 2048
    assert request["temperature"] == 0.3
    assert "reasoning_effort" not in request


@pytest.mark.asyncio
async def test_reasoning_models_keep_larger_explicit_completion_budget() -> None:
    stub = _OpenAIStub()
    client = OpenAIClient(model="gpt-5.4-mini", api_key="test-key")
    client._client = stub

    await client.complete(
        system="system",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32000,
    )

    request = stub.chat.completions.requests[0]
    assert request["max_completion_tokens"] == 32000


@pytest.mark.asyncio
async def test_finish_reason_length_with_empty_content_raises() -> None:
    """finish_reason=length + empty content must raise, not silently return ''."""
    truncated_choice = _Choice(content="", finish_reason="length")
    stub = _OpenAIStub(response=_Response(choice=truncated_choice))
    client = OpenAIClient(model="gpt-5.4-mini", api_key="test-key")
    client._client = stub

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        await client.complete(
            system="system",
            messages=[{"role": "user", "content": "summarize this paper"}],
        )


@pytest.mark.asyncio
async def test_finish_reason_length_with_partial_content_raises() -> None:
    """Partial JSON truncated by length limit must also raise."""
    truncated_choice = _Choice(content='{"summary": "cut off here...', finish_reason="length")
    stub = _OpenAIStub(response=_Response(choice=truncated_choice))
    client = OpenAIClient(model="gpt-5.4-mini", api_key="test-key")
    client._client = stub

    with pytest.raises(RuntimeError, match="finish_reason=length"):
        await client.complete(
            system="system",
            messages=[{"role": "user", "content": "summarize this paper"}],
        )


@pytest.mark.asyncio
async def test_reasoning_model_token_budget_exceeds_minimum() -> None:
    """Reasoning models must get at least OPENAI_REASONING_MIN_COMPLETION_TOKENS (25000)."""
    assert OPENAI_REASONING_MIN_COMPLETION_TOKENS >= 25000, (
        "Budget too low: reasoning models need space for hidden reasoning + visible JSON. "
        f"Got {OPENAI_REASONING_MIN_COMPLETION_TOKENS}, need at least 25000."
    )
