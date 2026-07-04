"""Hosted /mcp edge: auth, metering, and an end-to-end MCP tool call over HTTP.

The edge needs its MCP session manager running (normally started by api.py's
lifespan); tests enter `mcp_edge.run()` explicitly since the ASGI test client
skips lifespan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx

from aimnis import api, apikeys, db
from aimnis.config import settings
from aimnis.mcp_http import mcp_edge

_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    # Stateless server: no initialize handshake needed, but the protocol version
    # header must be present on direct calls.
    "MCP-Protocol-Version": "2025-06-18",
}


def _tools_call(query: str) -> dict:
    return {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "stats", "arguments": {}},
    }


_TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


@asynccontextmanager
async def _client(pool, monkeypatch):
    async def fake_get_pool():
        return pool
    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    # mcp_server imports get_pool directly (`from .db import get_pool`), so the
    # module-local reference must be patched too or the MCP tools grab a REAL pool.
    from aimnis import mcp_server
    monkeypatch.setattr(mcp_server, "get_pool", fake_get_pool)
    async with mcp_edge.run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://t"
        ) as c:
            yield c


async def test_mcp_keyless_handshake_succeeds(clean, monkeypatch):
    # A key-less connection must be able to complete the handshake: a 401 during
    # initialize dies inside the MCP client library where nobody reads it. The
    # agent should connect, see the tools, and learn about keys on first call.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS, json=_TOOLS_LIST)
    assert r.status_code == 200
    assert {t["name"] for t in r.json()["result"]["tools"]} >= {"search", "stats"}
    # Nothing metered, nothing recorded for anonymous chatter.
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_keyless_tool_call_gets_onboarding_message(clean, monkeypatch):
    # The key-less tools/call is the onboarding surface: a JSON-RPC tool result
    # (isError) whose text lands in the model's context and points at /register.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS, json=_tools_call("x"))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1  # echoes the request id
    assert body["result"]["isError"] is True
    text = body["result"]["content"][0]["text"]
    assert "aimnis.com/register" in text and "Bearer aim_" in text
    # No search ran, nothing was metered.
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0
    assert await clean.fetchval("SELECT count(*) FROM lookup_event") == 0


async def test_mcp_presented_bad_key_still_401(clean, monkeypatch):
    # Key-less is onboarding; a PRESENTED but invalid key is refused outright —
    # with the self-serve hint in the body and WWW-Authenticate for MCP clients.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        for payload in (_TOOLS_LIST, _tools_call("x")):
            r = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": "nope"},
                             json=payload)
            assert r.status_code == 401
            assert "aimnis.com/register" in r.json().get("hint", "")
            assert r.headers["www-authenticate"].startswith("Bearer")


async def test_mcp_tools_list_with_client_key_is_unmetered(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcp@example.com")
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                         json=_TOOLS_LIST)
    assert r.status_code == 200
    tools = {t["name"] for t in r.json()["result"]["tools"]}
    assert {"search", "stats"} <= tools
    # Protocol chatter doesn't burn quota.
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_tools_call_is_metered_and_works(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcp2@example.com")
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                         json=_tools_call("x"))
    assert r.status_code == 200
    body = r.json()
    text = body["result"]["content"][0]["text"]
    assert "hit rate" in text.lower() or "pool" in text.lower()
    # Exactly one metered request logged.
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 1


async def test_mcp_tools_call_rate_limited(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcp3@example.com", rpm_limit=1, rpd_limit=100)
    async with _client(clean, monkeypatch) as c:
        first = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                             json=_tools_call("x"))
        second = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                              json=_tools_call("x"))
    assert first.status_code == 200
    assert second.status_code == 429


async def test_mcp_admin_key_unmetered(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", ["adminsecret"])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp",
                         headers={**_MCP_HEADERS, "Authorization": "Bearer adminsecret"},
                         json=_tools_call("x"))
    assert r.status_code == 200
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_revoked_key_rejected(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcp4@example.com")
    await apikeys.revoke(clean, email="mcp4@example.com")
    async with _client(clean, monkeypatch) as c:
        call = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                            json=_tools_call("x"))
        chatter = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                               json=_TOOLS_LIST)
    assert call.status_code == 401
    assert chatter.status_code == 401


async def test_mcp_malformed_body_still_metered(clean, monkeypatch):
    # An unparseable body must be treated as a tool call (fail-closed metering),
    # not slip through the unmetered chatter path.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="mcp5@example.com", rpm_limit=1, rpd_limit=100)
    async with _client(clean, monkeypatch) as c:
        await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                     content=b"{not json")
        r = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": issued.key},
                         json=_tools_call("x"))
    assert r.status_code == 429  # first (malformed) request consumed the 1/min budget
