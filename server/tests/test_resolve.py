"""Resolution-engine tests: miss → live+pool, then repeat → cache hit.

Requires both Postgres (the `clean` fixture) and a reachable SearXNG; skips if
SearXNG or the embedding model are unavailable.
"""

from __future__ import annotations

import httpx
import pytest

from aimnis import resolve
from aimnis.config import settings


async def _searxng_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(
                f"{settings.searxng_url}/search", params={"q": "ping", "format": "json"}
            )
            return r.status_code == 200
    except Exception:
        return False


async def test_miss_then_exact_cache_hit(clean, monkeypatch):
    if not await _searxng_up():
        pytest.skip("SearXNG not reachable")
    try:
        from aimnis.embedding import embed  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")

    # Pin to SearXNG and disable distillation so the test spends no Brave/LLM quota.
    monkeypatch.setattr(settings, "search_backend", "searxng")
    monkeypatch.setattr(settings, "distill_enabled", False)

    query = "python resolve ImportError no module named requests"

    first = await resolve.resolve_search(clean, query)
    if first.get("source") == "error" or not first.get("results"):
        pytest.skip("SearXNG returned no results (upstream engines rate-limited / CAPTCHA)")
    assert first["source"] == "live"
    assert len(first["results"]) > 0

    # Identical query → served from cache (exact hash match), no live call.
    second = await resolve.resolve_search(clean, query)
    assert second["source"] == "cache"
    assert second["match"] == "exact"
    assert second["entry_id"] == first["entry_id"]


async def test_format_for_agent_shapes():
    live = resolve.format_for_agent(
        {"source": "live", "results": [{"title": "T", "url": "http://x", "snippet": "s"}]}
    )
    assert "live results" in live and "http://x" in live

    cached = resolve.format_for_agent(
        {"source": "cache", "match": "semantic", "answer": None,
         "results": [{"title": "T", "url": "http://x", "snippet": "s"}]}
    )
    assert "cache hit" in cached

    err = resolve.format_for_agent({"source": "error", "error": "boom", "results": []})
    assert "unavailable" in err


def test_humanize_age():
    assert resolve._humanize_age(None) is None
    assert resolve._humanize_age(10) == "just now"
    assert resolve._humanize_age(600) == "10 min ago"
    assert resolve._humanize_age(3 * 3600) == "3 hours ago"
    assert resolve._humanize_age(5 * 86400) == "5 days ago"


async def test_format_shows_cache_age():
    out = resolve.format_for_agent({
        "source": "cache", "match": "exact", "answer": "do X",
        "cached_at": "2026-06-28T12:00:00+00:00", "age_seconds": 5 * 86400,
        "results": [],
    })
    assert "cached 2026-06-28" in out
    assert "5 days ago" in out


async def test_format_echoes_matched_question_for_polarity_check():
    out = resolve.format_for_agent({
        "source": "cache", "match": "semantic",
        "matched_query": "how do I disable telemetry in dotnet",
        "answer": "set the env var", "results": [],
    })
    # The agent must see the cached question to judge intent/polarity itself.
    assert "how do I disable telemetry in dotnet" in out
    assert "request a live search" in out


async def test_format_semantic_hit_offers_reject_entry():
    out = resolve.format_for_agent({
        "source": "cache", "match": "semantic", "answer": "cached answer",
        "matched_query": "enable http2 in nginx", "entry_id": "abc-123",
        "results": [],
    })
    # The sanctioned escape hatch: retrying with reject_entry both bypasses the
    # mis-hit and labels it for the satisfaction metric.
    assert 'reject_entry="abc-123"' in out

    # Without an entry id (unpooled), fall back to the plain disregard wording.
    out2 = resolve.format_for_agent({
        "source": "cache", "match": "semantic", "answer": "cached answer",
        "matched_query": "enable http2 in nginx", "entry_id": None,
        "results": [],
    })
    assert "reject_entry" not in out2 and "disregard" in out2
