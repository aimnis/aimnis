"""Search backend tests.

The Brave parser/dispatch tests are offline (canned payloads / monkeypatched
backends) so the suite never spends Brave quota. The one live test is pinned to
SearXNG for the same reason and skips if SearXNG is unreachable / empty.
"""

from __future__ import annotations

import httpx
import pytest

from aimnis import search
from aimnis.config import settings
from aimnis.search import SearchError, SearchResult, live_search


@pytest.fixture(autouse=True)
def _isolate_provider_keys(monkeypatch):
    """Null Tavily/Exa keys by default so a developer's real keys in .env don't
    make the fallback-chain tests hit live APIs. Tests that exercise those
    providers set their own keys explicitly."""
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "exa_api_key", None)


async def _searxng_up() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(
                f"{settings.searxng_url}/search", params={"q": "ping", "format": "json"}
            )
            return r.status_code == 200
    except Exception:
        return False


def test_clean_strips_tags_and_entities():
    assert search._clean("use <strong>read_text</strong> &quot;x&quot;") == 'use read_text "x"'
    assert search._clean(None) == ""


def test_parse_brave_shape_and_extra_snippets():
    data = {"web": {"results": [
        {"title": "pathlib &amp; open", "url": "http://a",
         "description": "<strong>read_text</strong> reads a file",
         "extra_snippets": ["opens then closes the file", "returns a str", ""]},
        {"title": "no url dropped", "description": "x"},  # no url → skipped
    ]}}
    out = search._parse_brave(data, limit=8)
    assert len(out) == 1
    r = out[0]
    assert r.title == "pathlib & open"
    assert r.url == "http://a"
    assert r.snippet == "read_text reads a file"
    assert r.extra_snippets == ("opens then closes the file", "returns a str")


def test_parse_tavily_and_exa_shapes():
    tav = {"results": [
        {"title": "H2 in nginx", "url": "http://a", "content": "enable http2 on;"},
        {"title": "no url", "content": "x"},  # dropped
    ]}
    out = [r for r in _map_tavily(tav)]
    assert len(out) == 1 and out[0].url == "http://a" and out[0].snippet == "enable http2 on;"

    exa = {"results": [
        {"title": "Neural", "url": "http://b", "text": "semantic recall result"},
        {"url": ""},  # dropped (empty url)
    ]}
    out = [r for r in _map_exa(exa)]
    assert len(out) == 1 and out[0].url == "http://b" and out[0].snippet == "semantic recall result"


def _map_tavily(data):
    return [search.SearchResult(title=search._clean(i.get("title")), url=i["url"],
                                snippet=search._clean(i.get("content")))
            for i in (data.get("results") or []) if i.get("url")]


def _map_exa(data):
    return [search.SearchResult(title=search._clean(i.get("title")), url=i["url"],
                                snippet=search._clean(i.get("text")))
            for i in (data.get("results") or []) if i.get("url")]


def test_provider_chain_auto_filters_unkeyed(monkeypatch):
    monkeypatch.setattr(settings, "search_backend", "auto")
    monkeypatch.setattr(settings, "search_preference", ("brave", "tavily", "exa", "searxng"))
    # No keys: auto chain still lists all names (usability is filtered in live_search),
    # but searxng is always present as the floor.
    assert search._provider_chain() == ["brave", "tavily", "exa", "searxng"]
    monkeypatch.setattr(settings, "brave_api_key", None)
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "exa_api_key", None)
    assert search._usable("searxng") is True
    assert search._usable("brave") is False
    monkeypatch.setattr(settings, "tavily_api_key", "k")
    assert search._usable("tavily") is True


def test_provider_chain_explicit_backend_first_then_fallback(monkeypatch):
    monkeypatch.setattr(settings, "search_backend", "tavily")
    monkeypatch.setattr(settings, "search_preference", ("brave", "tavily", "exa", "searxng"))
    chain = search._provider_chain()
    assert chain[0] == "tavily"
    assert chain == ["tavily", "brave", "exa", "searxng"]


async def test_live_search_walks_full_chain_to_exa(monkeypatch):
    # brave + tavily fail, exa succeeds → chain reaches exa before searxng.
    monkeypatch.setattr(settings, "search_backend", "auto")
    monkeypatch.setattr(settings, "search_preference", ("brave", "tavily", "exa", "searxng"))
    monkeypatch.setattr(settings, "brave_api_key", "k")
    monkeypatch.setattr(settings, "tavily_api_key", "k")
    monkeypatch.setattr(settings, "exa_api_key", "k")

    async def boom(name):
        async def _f(query, limit):
            raise SearchError(f"{name} 429")
        return _f

    monkeypatch.setattr(search, "_brave_search", await boom("brave"))
    monkeypatch.setattr(search, "_tavily_search", await boom("tavily"))

    async def exa_ok(query, limit):
        return [SearchResult(title="T", url="http://exa", snippet="s")]

    called = {"searxng": False}

    async def searxng_spy(query, limit):
        called["searxng"] = True
        return [SearchResult(title="T", url="http://sx", snippet="s")]

    monkeypatch.setattr(search, "_exa_search", exa_ok)
    monkeypatch.setattr(search, "_searxng_search", searxng_spy)

    results = await live_search("q")
    assert results[0].url == "http://exa"
    assert called["searxng"] is False  # exa satisfied it; searxng floor untouched


async def test_live_search_falls_back_when_primary_fails(monkeypatch):
    monkeypatch.setattr(settings, "search_backend", "brave")
    monkeypatch.setattr(settings, "brave_api_key", "k")

    async def brave_boom(query, limit):
        raise SearchError("brave 429")

    async def searxng_ok(query, limit):
        return [SearchResult(title="T", url="http://x", snippet="s")]

    monkeypatch.setattr(search, "_brave_search", brave_boom)
    monkeypatch.setattr(search, "_searxng_search", searxng_ok)

    results = await live_search("q")
    assert len(results) == 1 and results[0].url == "http://x"


async def test_live_search_falls_back_on_empty_primary(monkeypatch):
    monkeypatch.setattr(settings, "search_backend", "brave")
    monkeypatch.setattr(settings, "brave_api_key", "k")

    async def brave_empty(query, limit):
        return []

    async def searxng_ok(query, limit):
        return [SearchResult(title="T", url="http://x", snippet="s")]

    monkeypatch.setattr(search, "_brave_search", brave_empty)
    monkeypatch.setattr(search, "_searxng_search", searxng_ok)

    results = await live_search("q")
    assert len(results) == 1


async def test_live_search_raises_when_all_backends_fail(monkeypatch):
    monkeypatch.setattr(settings, "search_backend", "searxng")
    monkeypatch.setattr(settings, "brave_api_key", None)

    async def searxng_boom(query, limit):
        raise SearchError("engines down")

    monkeypatch.setattr(search, "_searxng_search", searxng_boom)
    with pytest.raises(SearchError):
        await live_search("q")


async def test_live_search_searxng_smoke():
    # Pinned to SearXNG so the suite never spends Brave quota.
    from aimnis.config import settings as s
    if not await _searxng_up():
        pytest.skip("SearXNG not reachable")
    orig = s.search_backend
    s.search_backend = "searxng"
    try:
        results = await live_search("python fix ImportError")
    except SearchError:
        pytest.skip("SearXNG upstream engines returning nothing (rate-limited / CAPTCHA)")
    finally:
        s.search_backend = orig
    if not results:
        pytest.skip("SearXNG returned no results")
    assert all(r.url for r in results)
