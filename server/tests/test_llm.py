"""OpenRouter client tests. Uses httpx.MockTransport so no network or key is
needed and the HTTP-status → error-class mapping is exercised directly."""

from __future__ import annotations

import httpx
import pytest

from aimnis import llm


def _transport(handler):
    return httpx.MockTransport(handler)


def test_build_messages_numbers_sources():
    msgs = llm._build_messages(
        "how to X",
        [{"title": "A", "url": "http://a", "snippet": "sa"},
         {"title": "B", "url": "http://b", "snippet": "sb"}],
    )
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "how to X" in user
    assert "[1] A" in user and "[2] B" in user and "http://b" in user


async def test_distill_success_parses_answer_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "choices": [{"message": {"content": "Use register_vector [1]."}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18},
            },
        )

    out = await llm.distill(
        "q", [{"title": "T", "url": "http://x", "snippet": "s"}],
        api_key="test-key", transport=_transport(handler),
    )
    assert out.answer_text == "Use register_vector [1]."
    assert out.prompt_tokens == 120 and out.completion_tokens == 18
    assert "llama" in out.model


async def test_distill_maps_429_to_rate_limited():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    with pytest.raises(llm.LLMRateLimited) as ei:
        await llm.distill("q", [{"title": "T", "url": "u", "snippet": "s"}],
                          api_key="k", transport=_transport(handler))
    assert ei.value.http_status == 429


async def test_distill_maps_500_to_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(llm.LLMError) as ei:
        await llm.distill("q", [{"title": "T", "url": "u", "snippet": "s"}],
                          api_key="k", transport=_transport(handler))
    assert ei.value.http_status == 500


async def test_distill_empty_answer_is_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    with pytest.raises(llm.LLMError):
        await llm.distill("q", [{"title": "T", "url": "u", "snippet": "s"}],
                          api_key="k", transport=_transport(handler))


async def test_distill_without_key_raises(monkeypatch):
    # Force no key even when one is present in the environment (.env).
    from aimnis.config import settings
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    with pytest.raises(llm.LLMError):
        await llm.distill("q", [{"title": "T", "url": "u", "snippet": "s"}], api_key=None)
