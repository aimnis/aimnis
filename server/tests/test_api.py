"""Dashboard/API tests. Drives the ASGI app in-process (no server, no lifespan)
against the test DB by monkeypatching db.get_pool to the `clean` pool."""

from __future__ import annotations

import httpx

from aimnis import api, citations, db, resolve, stats
from aimnis.config import settings


async def _insert_entry(pool, sources) -> str:
    """Minimal servable pool_entry with the given sources; returns its id."""
    import json
    row = await pool.fetchrow(
        "INSERT INTO pool_entry (query_text, query_norm, query_hash, sources) "
        "VALUES ('q','q','h', $1::jsonb) RETURNING id",
        json.dumps(sources),
    )
    return str(row["id"])


def _client(pool, monkeypatch) -> httpx.AsyncClient:
    async def fake_get_pool():
        return pool
    monkeypatch.setattr(db, "get_pool", fake_get_pool)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://t")


async def test_healthz(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/healthz")
        assert r.status_code == 200 and r.json() == {"ok": True}


async def test_api_stats_shape(clean, monkeypatch):
    for _ in range(2):
        await stats.record_event(clean, query_hash="m", outcome="miss")
    for _ in range(3):
        await stats.record_event(clean, query_hash="h", outcome="hit_exact")

    async with _client(clean, monkeypatch) as c:
        r = await c.get("/api/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["lookups_total"] == 5
    assert abs(d["hit_rate"] - 0.6) < 1e-9
    assert d["target_hit_rate"] == 0.30
    assert len(d["series"]) == 5
    assert d["series"][-1]["unique_queries"] == 2  # two misses = two unique queries
    assert d["series"][-1]["lookups"] == 5


async def test_dashboard_html(clean, monkeypatch):
    await stats.record_event(clean, query_hash="m", outcome="miss")
    await stats.record_event(clean, query_hash="h", outcome="hit_exact")

    async with _client(clean, monkeypatch) as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Flywheel" in body
    assert "<svg" in body and "Gate 1 target" in body
    assert "50%" in body  # 1 hit / 2 lookups
    assert "Pool storage" in body  # storage tile renders


async def test_dashboard_empty_pool_renders(clean, monkeypatch):
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "No lookups yet" in r.text  # empty-series message, no crash


# --- Citation redirect (/r) ------------------------------------------------- #

async def test_citation_redirect_302s_and_logs_click(clean, monkeypatch):
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    monkeypatch.setattr(settings, "citation_public_base_url", "http://t")

    eid = await _insert_entry(clean, [
        {"title": "A", "url": "https://docs.python.org/3/os.html"},
        {"title": "B", "url": "https://nginx.org/en/docs/http2.html"},
    ])
    tok = citations.token_for(eid, "https://nginx.org/en/docs/http2.html")

    async with _client(clean, monkeypatch) as c:
        r = await c.get(f"/r/{tok}", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "https://nginx.org/en/docs/http2.html"

    # One aggregate click row: entry + found position + host + stable URL label.
    row = await clean.fetchrow(
        "SELECT entry_id, source_idx, host, source_url FROM citation_click"
    )
    assert str(row["entry_id"]) == eid
    assert row["source_idx"] == 1  # position it was found at
    assert row["host"] == "nginx.org"
    assert row["source_url"] == "https://nginx.org/en/docs/http2.html"


async def test_citation_link_survives_reordering(clean, monkeypatch):
    # A token minted before the entry's sources are re-ordered still resolves to
    # the same source afterward (URL-hash identity, not position).
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    url = "https://nginx.org/en/docs/http2.html"
    eid = await _insert_entry(clean, [{"title": "A", "url": "https://a.example/x"},
                                      {"title": "B", "url": url}])
    tok = citations.token_for(eid, url)  # url is at index 1
    # Re-order: put url first.
    import json
    await clean.execute("UPDATE pool_entry SET sources = $2::jsonb WHERE id = $1", eid,
                        json.dumps([{"title": "B", "url": url},
                                    {"title": "A", "url": "https://a.example/x"}]))
    async with _client(clean, monkeypatch) as c:
        r = await c.get(f"/r/{tok}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == url
    row = await clean.fetchrow("SELECT source_idx FROM citation_click")
    assert row["source_idx"] == 0  # now found at the new position


async def test_click_detail_is_gated_behind_api_key(clean, monkeypatch):
    # Public surfaces (dashboard + /api/stats) expose only AGGREGATE click counts;
    # the per-host / per-entry detail — which carries raw query text a scrubber miss
    # could expose — is served only from the authenticated /v1/stats.
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    monkeypatch.setattr(settings, "citation_public_base_url", "http://t")
    monkeypatch.setattr(settings, "gateway_api_keys", ["secret"])

    eid = await _insert_entry(clean, [{"title": "os", "url": "https://docs.python.org/3/os.html"}])
    await stats.record_event(clean, query_hash="h", outcome="hit_exact", entry_id=eid)
    tok = citations.token_for(eid, "https://docs.python.org/3/os.html")

    async with _client(clean, monkeypatch) as c:
        await c.get(f"/r/{tok}", follow_redirects=False)
        await c.get(f"/r/{tok}", follow_redirects=False)
        page = (await c.get("/")).text
        pub = (await c.get("/api/stats")).json()
        authed = (await c.get("/v1/stats", headers={"X-API-Key": "secret"})).json()

    # Public: the aggregate count is shown; the raw host/query detail is withheld.
    assert "docs.python.org" not in page
    assert "Most-followed" not in page
    assert pub["click_analytics"]["clicks_total"] == 2
    assert "top_hosts" not in pub["click_analytics"]
    assert "top_queries" not in pub

    # Authenticated: the per-host / per-entry detail is present for key holders.
    ca = authed["click_analytics"]
    assert ca["clicks_total"] == 2
    assert [list(x) for x in ca["top_hosts"]][0] == ["docs.python.org", 2]
    assert ca["top_entries"][0][1] == 2  # clicks on the top entry


async def test_citation_redirect_rejects_forged_token(clean, monkeypatch):
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/r/deadbeef-forged", follow_redirects=False)
    assert r.status_code == 404
    assert await clean.fetchval("SELECT count(*) FROM citation_click") == 0


async def test_citation_redirect_unknown_url_404s(clean, monkeypatch):
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    eid = await _insert_entry(clean, [{"title": "A", "url": "https://a.example/x"}])
    tok = citations.token_for(eid, "https://not-in-this-entry.example/y")
    async with _client(clean, monkeypatch) as c:
        r = await c.get(f"/r/{tok}", follow_redirects=False)
    assert r.status_code == 404  # no source in the entry matches the hash


async def test_citation_redirect_refuses_non_http_scheme(clean, monkeypatch):
    monkeypatch.setattr(settings, "citation_signing_secret", "s3cr3t")
    eid = await _insert_entry(clean, [{"title": "evil", "url": "javascript:alert(1)"}])
    tok = citations.token_for(eid, "javascript:alert(1)")
    async with _client(clean, monkeypatch) as c:
        r = await c.get(f"/r/{tok}", follow_redirects=False)
    assert r.status_code == 404  # never emit a non-http(s) redirect


# --- HTTP gateway (/v1) ----------------------------------------------------- #

async def test_gateway_fail_closed_without_keys(clean, monkeypatch):
    """No keys configured ⇒ the /v1 routes refuse (503), never open to the world."""
    monkeypatch.setattr(settings, "gateway_api_keys", [])
    async with _client(clean, monkeypatch) as c:
        assert (await c.get("/v1/stats")).status_code == 503
        assert (await c.post("/v1/search", json={"query": "x"})).status_code == 503


async def test_gateway_rejects_bad_key(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", ["secret"])
    async with _client(clean, monkeypatch) as c:
        assert (await c.get("/v1/stats")).status_code == 401                       # none
        r = await c.get("/v1/stats", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401                                                # wrong


async def test_gateway_stats_with_key(clean, monkeypatch):
    monkeypatch.setattr(settings, "gateway_api_keys", ["secret"])
    await stats.record_event(clean, query_hash="h", outcome="hit_exact")
    async with _client(clean, monkeypatch) as c:
        r = await c.get("/v1/stats", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    assert r.json()["lookups_total"] == 1


async def test_gateway_search_with_key(clean, monkeypatch):
    """Accepts the X-API-Key header and returns the resolution dict + rendered text."""
    monkeypatch.setattr(settings, "gateway_api_keys", ["secret"])

    async def fake_resolve(pool, query, *, niche=None):
        return {"source": "cache", "match": "exact", "distance": 0.0,
                "answer": "42.", "results": [], "model": "m", "entry_id": 1}
    monkeypatch.setattr(resolve, "resolve_search", fake_resolve)

    async with _client(clean, monkeypatch) as c:
        r = await c.post("/v1/search", json={"query": "meaning of life"},
                         headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    d = r.json()
    assert d["answer"] == "42." and d["source"] == "cache"
    assert "formatted" in d and "42." in d["formatted"]
