"""Quota-ledger enforcement tests."""

from __future__ import annotations

from aimnis import quota


async def test_unknown_key(clean):
    res = await quota.reserve(clean, "background_precompute", label="does-not-exist")
    assert not res.granted
    assert res.reason == "unknown_key"
    assert res.call_id is None


async def test_inactive_key(clean):
    res = await quota.reserve(clean, "background_precompute", label="test-inactive")
    assert not res.granted
    assert res.reason == "key_inactive"


async def test_grant_and_record(clean):
    res = await quota.reserve(clean, "background_precompute", label="test-day")
    assert res.granted
    assert res.call_id is not None

    await quota.record_outcome(clean, res.call_id, "success", http_status=200)
    row = await clean.fetchrow(
        "SELECT status, http_status FROM upstream_call WHERE id = $1", res.call_id
    )
    assert row["status"] == "success"
    assert row["http_status"] == 200


async def test_rpm_limit(clean):
    a = await quota.reserve(clean, "p", label="test-rpm")
    b = await quota.reserve(clean, "p", label="test-rpm")
    c = await quota.reserve(clean, "p", label="test-rpm")
    assert a.granted and b.granted
    assert not c.granted
    assert c.reason == "rate_minute"


async def test_abandon_frees_a_slot(clean):
    a = await quota.reserve(clean, "p", label="test-rpm")
    b = await quota.reserve(clean, "p", label="test-rpm")
    assert a.granted and b.granted

    await quota.abandon(clean, b.call_id)  # never sent → stops counting
    c = await quota.reserve(clean, "p", label="test-rpm")
    assert c.granted


async def test_failed_call_still_counts(clean):
    """A 429 consumes quota, so a recorded rate_limited call keeps counting."""
    a = await quota.reserve(clean, "p", label="test-rpm")
    await quota.record_outcome(clean, a.call_id, "rate_limited", http_status=429)
    b = await quota.reserve(clean, "p", label="test-rpm")
    c = await quota.reserve(clean, "p", label="test-rpm")
    assert b.granted
    assert not c.granted
    assert c.reason == "rate_minute"


async def test_purpose_budget(clean):
    a = await quota.reserve(clean, "background_precompute", label="test-day")
    b = await quota.reserve(clean, "background_precompute", label="test-day")
    c = await quota.reserve(clean, "background_precompute", label="test-day")
    assert a.granted and b.granted
    assert not c.granted
    assert c.reason == "purpose_budget"

    # A purpose without a configured budget is still allowed (until the day cap).
    d = await quota.reserve(clean, "stale_refresh", label="test-day")
    assert d.granted


async def test_day_limit(clean):
    reserved = [await quota.reserve(clean, "stale_refresh", label="test-day") for _ in range(3)]
    assert all(r.granted for r in reserved)

    over = await quota.reserve(clean, "stale_refresh", label="test-day")
    assert not over.granted
    assert over.reason == "rate_day"


async def test_usage_snapshot(clean):
    await quota.reserve(clean, "stale_refresh", label="test-day")
    u = await quota.usage(clean, label="test-day")
    assert u is not None
    assert u.day_used == 1
    assert u.day_limit == 3
    assert u.minute_limit == 100


async def test_reserved_call_context_manager(clean):
    async with quota.reserved_call(clean, "background_precompute", label="test-day") as res:
        assert res.granted
    row = await clean.fetchrow(
        "SELECT status FROM upstream_call WHERE id = $1", res.call_id
    )
    assert row["status"] == "success"


async def test_reserved_call_records_error(clean):
    try:
        async with quota.reserved_call(clean, "background_precompute", label="test-day") as res:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    row = await clean.fetchrow(
        "SELECT status, error FROM upstream_call WHERE id = $1", res.call_id
    )
    assert row["status"] == "error"
    assert "boom" in row["error"]
