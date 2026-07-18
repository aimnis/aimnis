"""request_log telemetry tests. Drive telemetry.record_request + reach directly
(no network) so the adoption rollup is verified deterministically."""

from __future__ import annotations

import pytest

from aimnis import telemetry
from aimnis.config import settings


async def _rec(pool, **kw):
    base = dict(surface="mcp", method="POST", path="/mcp",
                ip_hash="aaa", user_agent="claude-code/1.0", tool="search", auth="keyless")
    base.update(kw)
    await telemetry.record_request(pool, **base)


async def test_reach_empty(clean):
    r = await telemetry.reach(clean)
    assert r.requests_total == 0
    assert r.sources_total == 0
    assert r.tool_calls_total == 0
    assert r.top_user_agents == []
    assert r.daily == []


async def test_record_and_reach(clean):
    await _rec(clean, ip_hash="aaa", user_agent="claude-code/1.0", tool="search", auth="keyless")
    await _rec(clean, ip_hash="aaa", user_agent="claude-code/1.0", tool=None, auth="keyless")
    await _rec(clean, ip_hash="bbb", user_agent="node", tool="search", auth="keyed")

    r = await telemetry.reach(clean)
    assert r.requests_total == 3
    assert r.tool_calls_total == 2            # two tools/call rows (tool IS NOT NULL)
    assert r.sources_total == 2               # two distinct ip_hash
    assert r.keyless_sources == 1
    assert r.keyed_sources == 1
    # most-frequent UA first: claude-code/1.0 = 2 requests from 1 distinct source
    assert r.top_user_agents[0] == ("claude-code/1.0", 2, 1)
    assert ("node", 1, 1) in r.top_user_agents
    # today's daily bucket carries all three
    assert r.daily[0][1] == 3                 # (day, requests, tool_calls, sources)
    assert r.daily[0][2] == 2
    assert r.daily[0][3] == 2


async def test_keyed_and_admin_both_count_as_keyed_sources(clean):
    await _rec(clean, ip_hash="k1", auth="keyed")
    await _rec(clean, ip_hash="a1", auth="admin")
    await _rec(clean, ip_hash="n1", auth="keyless")
    r = await telemetry.reach(clean)
    assert r.keyed_sources == 2                # keyed + admin
    assert r.keyless_sources == 1


async def test_record_disabled_is_a_no_op(clean, monkeypatch):
    monkeypatch.setattr(settings, "request_log_enabled", False)
    await _rec(clean)
    r = await telemetry.reach(clean)
    assert r.requests_total == 0


async def test_record_is_best_effort_on_db_failure():
    """A broken pool must not raise — observability never breaks a request."""
    class BadPool:
        async def execute(self, *a, **k):
            raise RuntimeError("db down")

    # Must not raise (request_log_enabled defaults True, so the insert is attempted).
    await telemetry.record_request(
        BadPool(), surface="mcp", method="GET", path="/mcp",
        ip_hash="x", user_agent="ua",
    )


async def test_long_user_agent_and_path_are_truncated(clean):
    await _rec(clean, ip_hash="z", user_agent="U" * 900, path="/" + "p" * 900, tool=None)
    row = await clean.fetchrow("SELECT path, user_agent FROM request_log WHERE ip_hash = 'z'")
    assert len(row["user_agent"]) == 512
    assert len(row["path"]) == 256


@pytest.mark.parametrize("auth", ["keyless", "keyed", "admin"])
async def test_auth_values_round_trip(clean, auth):
    await _rec(clean, ip_hash=f"ip-{auth}", auth=auth)
    got = await clean.fetchval("SELECT auth FROM request_log WHERE ip_hash = $1", f"ip-{auth}")
    assert got == auth
