"""Hosted /mcp edge: auth, metering, and an end-to-end MCP tool call over HTTP.

The edge needs its MCP session manager running (normally started by api.py's
lifespan); tests enter `mcp_edge.run()` explicitly since the ASGI test client
skips lifespan.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager

import httpx

from aimnis import api, apikeys, db, flags
from aimnis.config import settings
from aimnis.mcp_http import _anon_minute, mcp_edge

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


def _register_call(email: str) -> dict:
    return {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "register", "arguments": {"email": email}},
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
    # The keyless minute throttle is in-process state keyed by IP; every test
    # shares the ASGI test client's IP, so clear it between tests.
    _anon_minute.clear()
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
    # The hosted edge carries the in-band signup tool alongside the search tools.
    assert {t["name"] for t in r.json()["result"]["tools"]} >= {"search", "stats", "register"}
    # Nothing metered, nothing recorded for anonymous chatter.
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_keyless_tool_call_runs_free(clean, monkeypatch):
    # The free tier: a key-less tools/call actually EXECUTES (stats here; the
    # search miss budget is exercised at the resolve layer). Nothing is metered
    # against client keys.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS, json=_tools_call("x"))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 1  # echoes the request id
    assert body["result"].get("isError") is not True
    assert "hit rate" in body["result"]["content"][0]["text"].lower()
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_keyless_kill_switch_restores_onboarding(clean, monkeypatch):
    # anon_search_enabled=False reverts key-less tool calls to the onboarding
    # message (isError result pointing at register) — the pre-free-tier behavior.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    monkeypatch.setattr(settings, "anon_search_enabled", False)
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS, json=_tools_call("x"))
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["isError"] is True
    text = body["result"]["content"][0]["text"]
    assert "aimnis.com/register" in text and "register" in text
    assert await clean.fetchval("SELECT count(*) FROM lookup_event") == 0


async def test_mcp_keyless_minute_throttle(clean, monkeypatch):
    # In-process per-IP throttle on keyless tool calls (cache lookups burn CPU).
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    monkeypatch.setattr(settings, "anon_rpm", 1)
    async with _client(clean, monkeypatch) as c:
        first = await c.post("/mcp", headers=_MCP_HEADERS, json=_tools_call("x"))
        second = await c.post("/mcp", headers=_MCP_HEADERS, json=_tools_call("x"))
        # Chatter is NOT throttled — only tool calls.
        chatter = await c.post("/mcp", headers=_MCP_HEADERS, json=_TOOLS_LIST)
    assert first.status_code == 200
    assert second.status_code == 429
    assert "aimnis.com/register" in second.json()["error"]
    assert chatter.status_code == 200


async def test_mcp_register_tool_issues_key_in_band(clean, monkeypatch):
    # The whole point of the in-band funnel: one tool call → a working key in the
    # tool result, no email round-trip (email is a best-effort copy).
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS,
                         json=_register_call("newuser@example.com"))
        assert r.status_code == 200
        text = r.json()["result"]["content"][0]["text"]
        key = re.search(r"aim_[A-Za-z0-9_-]+", text).group(0)
        # The issued key authenticates and meters like any portal-issued key.
        keyed = await c.post("/mcp", headers={**_MCP_HEADERS, "X-API-Key": key},
                             json=_tools_call("x"))
    row = await clean.fetchrow("SELECT email, status, key_hash FROM api_client")
    assert row["email"] == "newuser@example.com" and row["status"] == "active"
    assert row["key_hash"] == apikeys.hash_key(key)
    assert keyed.status_code == 200
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 1


async def test_mcp_register_tool_rejects_bad_email(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS, json=_register_call("not-an-email"))
    text = r.json()["result"]["content"][0]["text"]
    assert "valid email" in text and "aim_" not in text
    assert await clean.fetchval("SELECT count(*) FROM api_client") == 0


async def test_mcp_register_tool_at_capacity_waitlists(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    await flags.set_flag(clean, "registration_open", False)
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp", headers=_MCP_HEADERS,
                         json=_register_call("waitme@example.com"))
    text = r.json()["result"]["content"][0]["text"]
    assert "waitlist" in text.lower() and "aim_" not in text
    assert await clean.fetchval("SELECT count(*) FROM api_client") == 0
    assert await clean.fetchval(
        "SELECT count(*) FROM waitlist WHERE email='waitme@example.com'") == 1


async def test_mcp_register_tool_per_ip_daily_cap(clean, monkeypatch):
    # Key farming from one network is bounded per day; the refusal points at the
    # portal path instead of dead-ending.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    monkeypatch.setattr(settings, "anon_reg_rpd", 1)
    async with _client(clean, monkeypatch) as c:
        r1 = await c.post("/mcp", headers=_MCP_HEADERS, json=_register_call("a@example.com"))
        r2 = await c.post("/mcp", headers=_MCP_HEADERS, json=_register_call("b@example.com"))
    assert "aim_" in r1.json()["result"]["content"][0]["text"]
    t2 = r2.json()["result"]["content"][0]["text"]
    assert "aim_" not in t2 and "aimnis.com/register" in t2
    assert await clean.fetchval("SELECT count(*) FROM api_client") == 1


async def test_mcp_url_api_key_works_and_is_metered(clean, monkeypatch):
    # Some MCP clients can only be configured with a bare URL — `?api_key=aim_...`
    # must authenticate exactly like the header (and headers win when both exist).
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="urlkey@example.com")
    async with _client(clean, monkeypatch) as c:
        r = await c.post(f"/mcp?api_key={issued.key}", headers=_MCP_HEADERS,
                         json=_tools_call("x"))
    assert r.status_code == 200
    assert "hit rate" in r.json()["result"]["content"][0]["text"].lower()
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 1


async def test_mcp_foreign_url_key_stays_on_keyless_path(clean, monkeypatch):
    # A non-aim_ query value (e.g. a gateway's own key leaking through a proxy)
    # is ignored — the caller stays on the keyless FREE path (the call runs),
    # not a 401 for a key they never meant to present to US.
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/mcp?api_key=8a58cfe0-daf2-4dca-8b38-6266ae7bdead",
                         headers=_MCP_HEADERS, json=_tools_call("x"))
    assert r.status_code == 200
    assert r.json()["result"].get("isError") is not True
    assert await clean.fetchval("SELECT count(*) FROM api_request") == 0


async def test_mcp_header_key_wins_over_url_key(clean, monkeypatch):
    # A presented header is authoritative: a bad header 401s even if the URL
    # carries a valid key (never silently fall back past explicit credentials).
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    issued = await apikeys.issue(clean, email="urlkey2@example.com")
    async with _client(clean, monkeypatch) as c:
        r = await c.post(f"/mcp?api_key={issued.key}",
                         headers={**_MCP_HEADERS, "X-API-Key": "nope"},
                         json=_tools_call("x"))
    assert r.status_code == 401


def test_access_log_redacts_api_keys():
    import logging
    from aimnis.api import _RedactKeysInAccessLog
    rec = logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                            '%s - "%s %s HTTP/%s" %d', None, None)
    rec.args = ("1.2.3.4:1", "POST", "/mcp?api_key=aim_SECRET&x=1", "1.1", 200)
    assert _RedactKeysInAccessLog().filter(rec) is True
    assert "aim_SECRET" not in rec.getMessage()
    assert "api_key=[redacted]" in rec.getMessage()


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
