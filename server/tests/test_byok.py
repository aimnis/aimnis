"""BYOK: encrypted credential storage, key-override plumbing, ToS provenance tags,
and both edges (REST + MCP) delivering the client's keys to the resolution engine."""

from __future__ import annotations

import httpx
import pytest

from aimnis import api, apikeys, db, resolve, search
from aimnis.config import settings
from aimnis.mcp_http import mcp_edge
from aimnis.search import SearchResult


@pytest.fixture(autouse=True)
def _byok_secret(monkeypatch):
    monkeypatch.setattr(settings, "byok_secret", "test-byok-secret")


def _client(pool, monkeypatch) -> httpx.AsyncClient:
    async def fake_get_pool():
        return pool
    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    # mcp_server imports get_pool directly; patch its module-local reference too.
    from aimnis import mcp_server
    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t")


CK = apikeys.ClientKeys(
    openrouter_api_key="sk-or-user-1",
    search_provider="tavily",
    search_api_key="tvly-user-1",
)


# --- storage ---------------------------------------------------------------- #

async def test_issue_with_byok_boosts_caps_and_encrypts(clean):
    issued = await apikeys.issue(clean, email="byok@example.com", client_keys=CK)
    assert issued.byok
    assert issued.rpm_limit == settings.byok_rpm
    assert issued.rpd_limit == settings.byok_rpd

    # Stored ciphertext, not plaintext; decrypt round-trips.
    row = await clean.fetchrow(
        "SELECT openrouter_key_enc, search_key_enc, search_provider FROM api_client WHERE id=$1",
        issued.id,
    )
    assert row["search_provider"] == "tavily"
    assert b"sk-or-user-1" not in bytes(row["openrouter_key_enc"])
    assert b"tvly-user-1" not in bytes(row["search_key_enc"])

    loaded = await apikeys.load_client_keys(clean, issued.id)
    assert loaded == CK


async def test_byok_requires_secret(clean, monkeypatch):
    monkeypatch.setattr(settings, "byok_secret", None)
    with pytest.raises(RuntimeError, match="BYOK is disabled"):
        await apikeys.issue(clean, email="x@example.com", client_keys=CK)


async def test_rotate_wipes_old_creds_and_revoke_wipes(clean):
    first = await apikeys.issue(clean, email="rot@example.com", client_keys=CK)
    second = await apikeys.issue(clean, email="rot@example.com")  # rotate, no keys

    old = await clean.fetchrow(
        "SELECT status, openrouter_key_enc, search_key_enc FROM api_client WHERE id=$1", first.id
    )
    assert old["status"] == "revoked"
    assert old["openrouter_key_enc"] is None and old["search_key_enc"] is None
    # The new key has no creds and default caps.
    assert not second.byok and second.rpd_limit == settings.client_default_rpd
    assert await apikeys.load_client_keys(clean, second.id) is None

    third = await apikeys.issue(clean, email="rev@example.com", client_keys=CK)
    await apikeys.revoke(clean, email="rev@example.com")
    gone = await clean.fetchrow(
        "SELECT openrouter_key_enc, search_key_enc FROM api_client WHERE id=$1", third.id
    )
    assert gone["openrouter_key_enc"] is None and gone["search_key_enc"] is None
    assert await apikeys.load_client_keys(clean, third.id) is None


# --- search override ---------------------------------------------------------#

async def test_live_search_tries_client_provider_first(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    async def fake_tavily(query, limit, api_key=None):
        calls.append(("tavily", api_key))
        return [SearchResult(title="t", url="https://t.example/x", snippet="s")]

    monkeypatch.setattr(search, "_tavily_search", fake_tavily)
    out = await search.live_search("q", limit=3, client_keys=CK)
    assert out and calls == [("tavily", "tvly-user-1")]  # client key, tried first


async def test_live_search_falls_back_to_service_chain(monkeypatch):
    calls: list[str] = []

    async def fail_tavily(query, limit, api_key=None):
        calls.append(f"tavily:{api_key}")
        raise search.SearchError("client key 429")

    async def ok_searxng(query, limit):
        calls.append("searxng")
        return [SearchResult(title="s", url="https://s.example/y", snippet="s")]

    monkeypatch.setattr(search, "_tavily_search", fail_tavily)
    monkeypatch.setattr(search, "_searxng_search", ok_searxng)
    monkeypatch.setattr(settings, "brave_api_key", None)
    monkeypatch.setattr(settings, "tavily_api_key", None)
    monkeypatch.setattr(settings, "exa_api_key", None)

    out = await search.live_search("q", limit=3, client_keys=CK)
    assert out
    assert calls[0] == "tavily:tvly-user-1"   # client key first…
    assert "searxng" in calls                  # …then the service floor


# --- distill override --------------------------------------------------------#

async def test_byok_distill_uses_client_key_and_skips_ledger(clean, monkeypatch):
    from aimnis import llm

    seen: dict = {}

    async def fake_distill(query, results, **kw):
        seen.update(kw)
        return llm.DistillResult(answer_text="a" * 60, model="m", prompt_tokens=1,
                                 completion_tokens=1)

    monkeypatch.setattr(llm, "distill", fake_distill)
    monkeypatch.setattr(settings, "openrouter_api_key", None)  # service key ABSENT

    out = await resolve._distill(clean, "q", "h", [{"title": "t", "url": "u", "snippet": "s"}],
                                 client_keys=CK)
    assert out is not None and seen.get("api_key") == "sk-or-user-1"
    # No ledger reservation was taken — their quota, not ours.
    assert await clean.fetchval("SELECT count(*) FROM upstream_call") == 0


async def test_byok_distill_failure_degrades_without_service_fallback(clean, monkeypatch):
    from aimnis import llm

    async def limited(query, results, **kw):
        raise llm.LLMRateLimited("their key is out", http_status=429)

    monkeypatch.setattr(llm, "distill", limited)
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-SERVICE")  # present but must not be used

    out = await resolve._distill(clean, "q", "h", [{"title": "t", "url": "u", "snippet": "s"}],
                                 client_keys=CK)
    assert out is None
    assert await clean.fetchval("SELECT count(*) FROM upstream_call") == 0


# --- provenance tagging ------------------------------------------------------#

async def test_pool_entry_provenance_tags_byok(clean, monkeypatch):
    async def fake_live_search(query, *, limit=None, client_keys=None):
        return [SearchResult(title="t", url="https://t.example/z", snippet="s")]

    from aimnis import llm

    async def fake_distill(query, results, **kw):
        return llm.DistillResult(answer_text="answer " * 12, model="m",
                                 prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(resolve, "live_search", fake_live_search)
    monkeypatch.setattr(llm, "distill", fake_distill)

    res = await resolve.resolve_search(clean, "unique byok provenance q", client_keys=CK)
    assert res["pooled"]
    import json
    prov = json.loads(await clean.fetchval(
        "SELECT provenance FROM pool_entry WHERE id=$1", res["entry_id"]))
    assert prov["byok_search"] == "tavily"
    assert prov["byok_distill"] is True


# --- edges -------------------------------------------------------------------#

async def test_rest_edge_delivers_client_keys(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="edge@example.com", client_keys=CK)

    captured: dict = {}

    async def fake_resolve(pool, query, *, niche=None, client_keys=None, client_id=None, reject_entry=None):
        captured["ck"] = client_keys
        return {"source": "cache", "match": "exact", "distance": 0.0,
                "answer": "ok", "results": [], "model": "m", "entry_id": 1}
    monkeypatch.setattr(resolve, "resolve_search", fake_resolve)

    async with _client(clean, monkeypatch) as c:
        r = await c.post("/v1/search", json={"query": "hi"},
                         headers={"Authorization": f"Bearer {issued.key}"})
    assert r.status_code == 200
    assert captured["ck"] == CK


async def test_mcp_edge_delivers_client_keys(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcpbyok@example.com", client_keys=CK)

    captured: dict = {}

    async def fake_resolve(pool, query, *, niche=None, client_keys=None, client_id=None, reject_entry=None):
        captured["ck"] = client_keys
        return {"source": "cache", "match": "exact", "distance": 0.0,
                "answer": "ok", "results": [], "model": "m", "entry_id": None}
    monkeypatch.setattr(resolve, "resolve_search", fake_resolve)

    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "search", "arguments": {"query": "hi"}}}
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2025-06-18",
               "X-API-Key": issued.key}
    async with mcp_edge.run():
        async with _client(clean, monkeypatch) as c:
            r = await c.post("/mcp", headers=headers, json=payload)
    assert r.status_code == 200
    assert captured["ck"] == CK  # contextvar crossed the MCP task boundary
