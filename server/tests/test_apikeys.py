"""DB-backed client keys: issuance, hashing, atomic reserve + rate limits, revoke."""

from __future__ import annotations

from aimnis import apikeys


async def test_issue_creates_active_key_and_reserve_grants(clean):
    issued = await apikeys.issue(clean, email="a@example.com", label="eval")
    assert issued.key.startswith("aim_")
    assert issued.prefix == issued.key[:12]

    res = await apikeys.reserve(clean, issued.key)
    assert res.granted and res.reason == "granted"
    assert res.client_id == issued.id

    # The plaintext is never stored — only its hash.
    stored = await clean.fetchval("SELECT key_hash FROM api_client WHERE id=$1", issued.id)
    assert stored == apikeys.hash_key(issued.key)
    assert issued.key not in (stored or "")


async def test_reserve_unknown_key_denied(clean):
    res = await apikeys.reserve(clean, "aim_definitely-not-issued")
    assert not res.granted and res.reason == "unknown_key" and res.client_id is None


async def test_reserve_revoked_key_denied(clean):
    issued = await apikeys.issue(clean, email="b@example.com")
    n = await apikeys.revoke(clean, prefix=issued.prefix)
    assert n == 1
    res = await apikeys.reserve(clean, issued.key)
    assert not res.granted and res.reason == "revoked"


async def test_per_minute_rate_limit(clean):
    issued = await apikeys.issue(clean, email="c@example.com", rpm_limit=2, rpd_limit=1000)
    assert (await apikeys.reserve(clean, issued.key)).granted
    assert (await apikeys.reserve(clean, issued.key)).granted
    third = await apikeys.reserve(clean, issued.key)
    assert not third.granted and third.reason == "rate_minute"


async def test_per_day_rate_limit(clean):
    issued = await apikeys.issue(clean, email="d@example.com", rpm_limit=100, rpd_limit=2)
    assert (await apikeys.reserve(clean, issued.key)).granted
    assert (await apikeys.reserve(clean, issued.key)).granted
    third = await apikeys.reserve(clean, issued.key)
    assert not third.granted and third.reason == "rate_day"


async def test_reissue_rotates_the_active_key_per_email(clean):
    first = await apikeys.issue(clean, email="e@example.com")
    second = await apikeys.issue(clean, email="e@example.com")

    # Exactly one active key for the email; the old one is revoked and no longer works.
    active = await clean.fetch(
        "SELECT key_prefix FROM api_client WHERE lower(email)='e@example.com' AND status='active'"
    )
    assert len(active) == 1 and active[0]["key_prefix"] == second.prefix
    assert not (await apikeys.reserve(clean, first.key)).granted
    assert (await apikeys.reserve(clean, second.key)).granted


async def test_revoke_by_email(clean):
    issued = await apikeys.issue(clean, email="f@example.com")
    assert await apikeys.revoke(clean, email="F@Example.com") == 1  # case-insensitive
    assert not (await apikeys.reserve(clean, issued.key)).granted


# --------------------------------------------------------------------------- #
# Anonymous (keyless free-tier) per-IP daily budgets
# --------------------------------------------------------------------------- #

async def test_reserve_anon_miss_caps_per_ip_per_day(clean, monkeypatch):
    from aimnis.config import settings
    monkeypatch.setattr(settings, "anon_miss_rpd", 2)
    h = apikeys.hash_ip("203.0.113.9")
    assert await apikeys.reserve_anon_miss(clean, h)
    assert await apikeys.reserve_anon_miss(clean, h)
    assert not await apikeys.reserve_anon_miss(clean, h)
    # Independent budget per IP…
    assert await apikeys.reserve_anon_miss(clean, apikeys.hash_ip("203.0.113.10"))
    # …and per kind: the exhausted-miss IP can still register.
    assert await apikeys.reserve_anon_registration(clean, h)


async def test_reserve_anon_zero_cap_denies(clean, monkeypatch):
    from aimnis.config import settings
    monkeypatch.setattr(settings, "anon_miss_rpd", 0)
    assert not await apikeys.reserve_anon_miss(clean, apikeys.hash_ip("203.0.113.11"))


def test_hash_ip_is_salted_and_short():
    h = apikeys.hash_ip("203.0.113.9")
    assert len(h) == 16
    # Deterministic (budgets accumulate) but not the bare sha256 of the IP.
    import hashlib
    assert h == apikeys.hash_ip("203.0.113.9")
    assert h != hashlib.sha256(b"203.0.113.9").hexdigest()[:16]
