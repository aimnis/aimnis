"""Portal tests — landing, self-serve registration, waitlist, pause flag, and that an
issued key authenticates through the gateway. Drives the ASGI app in-process."""

from __future__ import annotations

import httpx
import pytest

from aimnis import api, apikeys, db, email as email_mod, flags, portal, resolve
from aimnis.config import settings


@pytest.fixture(autouse=True)
def _fresh_throttle():
    # The per-IP form throttle is in-process state; every test starts clean.
    portal._form_hits.clear()
    yield
    portal._form_hits.clear()


def _client(pool, monkeypatch) -> httpx.AsyncClient:
    async def fake_get_pool():
        return pool
    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t")


async def test_landing_and_terms(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        home = await c.get("/")
        terms = await c.get("/terms")
    assert home.status_code == 200 and "Collaborative search" in home.text
    assert "Get an eval API key" in home.text
    assert terms.status_code == 200
    assert "revoke any key at any time" in terms.text
    assert "AI-generated" in terms.text


async def test_setup_page_covers_all_agents(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/setup")
    assert r.status_code == 200
    body = r.text
    # Every supported agent has a section, and the hosted MCP endpoint is shown.
    for agent in ("OpenCode", "OpenClaw", "Hermes", "Pi", "Claude Code"):
        assert agent in body
    assert "/mcp" in body
    assert "streamable" in body.lower()
    assert "Bearer aim_YOUR_KEY" in body        # copy-paste snippets present
    assert "/v1/search" in body                  # REST fallback documented


async def test_register_form_shown_when_open(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/register")
    assert r.status_code == 200 and "Create my key" in r.text


async def test_register_issues_key_and_persists_active_client(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/register", data={"email": "user@example.com", "use_case": "cli"})
    assert r.status_code == 200
    assert "Your eval API key" in r.text
    assert "aim_" in r.text  # dev/self-host (no email provider): shown once on-screen

    row = await clean.fetchrow(
        "SELECT status, label FROM api_client WHERE lower(email)='user@example.com'"
    )
    assert row["status"] == "active" and row["label"] == "cli"


async def test_register_email_only_in_production(clean, monkeypatch):
    # With an email provider configured, the key travels ONLY by email — the page
    # must not contain it (on-screen delivery would enable throwaway-email farming).
    monkeypatch.setattr(settings, "resend_api_key", "re_test")
    sent: dict = {}

    async def fake_send(to, subject, html):
        sent.update(to=to, subject=subject, html=html)
        return True
    monkeypatch.setattr(email_mod, "send_email", fake_send)

    async with _client(clean, monkeypatch) as c:
        r = await c.post("/register", data={"email": "prod@example.com"})
    assert r.status_code == 200
    assert "Check your email" in r.text
    import re
    key = re.search(r"aim_[A-Za-z0-9_-]{20,}", sent["html"]).group(0)
    assert key not in r.text  # the full key never appears on-page (prefix hint is fine)
    assert sent["to"] == "prod@example.com"
    assert await clean.fetchval(
        "SELECT count(*) FROM api_client WHERE lower(email)='prod@example.com' "
        "AND status='active'"
    ) == 1


async def test_register_send_failure_revokes_key(clean, monkeypatch):
    monkeypatch.setattr(settings, "resend_api_key", "re_test")

    async def failing_send(to, subject, html):
        return False
    monkeypatch.setattr(email_mod, "send_email", failing_send)

    async with _client(clean, monkeypatch) as c:
        r = await c.post("/register", data={"email": "lost@example.com"})
    assert r.status_code == 502 and "couldn't send" in r.text.lower()
    # The undeliverable key must not remain usable.
    assert await clean.fetchval(
        "SELECT count(*) FROM api_client WHERE lower(email)='lost@example.com' "
        "AND status='active'"
    ) == 0


async def test_register_per_ip_throttle(clean, monkeypatch):
    monkeypatch.setattr(settings, "portal_ip_hourly", 2)
    async with _client(clean, monkeypatch) as c:
        first = await c.post("/register", data={"email": "t1@example.com"})
        second = await c.post("/register", data={"email": "t2@example.com"})
        third = await c.post("/register", data={"email": "t3@example.com"})
        wl = await c.post("/waitlist", data={"email": "t4@example.com"})  # shared budget
    assert first.status_code == 200 and second.status_code == 200
    assert third.status_code == 429 and "slow down" in third.text.lower()
    assert wl.status_code == 429
    assert await clean.fetchval(
        "SELECT count(*) FROM api_client WHERE lower(email)='t3@example.com'"
    ) == 0


async def test_register_rejects_bad_email(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/register", data={"email": "not-an-email"})
    assert r.status_code == 400 and "valid email" in r.text
    assert await clean.fetchval("SELECT count(*) FROM api_client") == 0


async def test_issued_key_authenticates_through_gateway(clean, monkeypatch):
    # No env gateway keys → auth falls through to the DB-backed client key.
    monkeypatch.setattr(settings, "gateway_api_keys", [])

    async def fake_resolve(pool, query, *, niche=None, client_keys=None):
        return {"source": "cache", "match": "exact", "distance": 0.0,
                "answer": "ok", "results": [], "model": "m", "entry_id": 1}
    monkeypatch.setattr(resolve, "resolve_search", fake_resolve)

    issued = await apikeys.issue(clean, email="agent@example.com")
    async with _client(clean, monkeypatch) as c:
        r = await c.post("/v1/search", json={"query": "hi"},
                         headers={"Authorization": f"Bearer {issued.key}"})
    assert r.status_code == 200 and r.json()["answer"] == "ok"


async def test_register_byok_flow(clean, monkeypatch):
    monkeypatch.setattr(settings, "byok_secret", "test-secret")
    async with _client(clean, monkeypatch) as c:
        form = (await c.get("/register")).text
        assert "Bring your own keys" in form  # section visible when enabled

        # Missing acknowledgement → rejected, nothing stored.
        bad = await c.post("/register", data={
            "email": "byok@example.com", "openrouter_key": "sk-or-x", "byok_ack": ""})
        assert bad.status_code == 400 and "checkbox" in bad.text
        assert await clean.fetchval("SELECT count(*) FROM api_client") == 0

        # Acknowledged → issued with BYOK caps; creds stored encrypted.
        ok = await c.post("/register", data={
            "email": "byok@example.com", "openrouter_key": "sk-or-x",
            "search_provider": "brave", "search_key": "BSA-x", "byok_ack": "yes"})
    assert ok.status_code == 200
    assert f"{settings.byok_rpd:,}" in ok.text  # boosted cap shown
    row = await clean.fetchrow(
        "SELECT rpd_limit, search_provider, openrouter_key_enc FROM api_client "
        "WHERE lower(email)='byok@example.com' AND status='active'")
    assert row["rpd_limit"] == settings.byok_rpd
    assert row["search_provider"] == "brave"
    assert row["openrouter_key_enc"] is not None


async def test_register_byok_hidden_and_rejected_when_disabled(clean, monkeypatch):
    monkeypatch.setattr(settings, "byok_secret", None)
    async with _client(clean, monkeypatch) as c:
        form = (await c.get("/register")).text
        assert "Bring your own keys" not in form
        r = await c.post("/register", data={
            "email": "x@example.com", "openrouter_key": "sk-or-x", "byok_ack": "yes"})
    assert r.status_code == 400 and "not available" in r.text


async def test_paused_shows_capacity_and_captures_waitlist(clean, monkeypatch):
    await flags.set_flag(clean, flags.REGISTRATION_OPEN, False)
    async with _client(clean, monkeypatch) as c:
        form = await c.get("/register")
        # Registering while paused routes to the waitlist, no key is issued.
        reg = await c.post("/register", data={"email": "wait@example.com"})
        wl = await c.post("/waitlist", data={"email": "wait@example.com"})
    assert "at capacity" in form.text.lower()
    assert "at capacity" in reg.text.lower()
    assert await clean.fetchval("SELECT count(*) FROM api_client") == 0
    assert wl.status_code == 200 and "on the list" in wl.text.lower()
    assert await clean.fetchval(
        "SELECT count(*) FROM waitlist WHERE lower(email)='wait@example.com'"
    ) == 1


async def test_waitlist_dedupes_email(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        await c.post("/waitlist", data={"email": "dup@example.com"})
        await c.post("/waitlist", data={"email": "DUP@example.com"})
    assert await clean.fetchval("SELECT count(*) FROM waitlist") == 1


async def test_admin_toggle_requires_key(clean, monkeypatch):
    # Disabled (fail-closed) when no admin key configured.
    monkeypatch.setattr(settings, "admin_api_key", None)
    async with _client(clean, monkeypatch) as c:
        assert (await c.post("/admin/registration", data={"open": "false"})).status_code == 404

    monkeypatch.setattr(settings, "admin_api_key", "adm1n")
    async with _client(clean, monkeypatch) as c:
        # Wrong / missing key rejected.
        assert (await c.post("/admin/registration", data={"open": "false"})).status_code == 401
        bad = await c.post("/admin/registration", data={"open": "false"},
                           headers={"X-Admin-Key": "wrong"})
        assert bad.status_code == 401
        # Correct key flips the flag live.
        ok = await c.post("/admin/registration", data={"open": "false"},
                          headers={"X-Admin-Key": "adm1n"})
        assert ok.status_code == 200 and ok.json() == {"registration_open": False}
    assert await flags.registration_open(clean) is False
