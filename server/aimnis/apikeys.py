"""DB-backed client API keys — issue, verify+meter, revoke.

These are the eval keys the self-serve portal hands out. They live in `api_client`
(not the env-var allowlist) so they can be issued/revoked without a redeploy, and
each carries its own per-minute + per-day request cap. Verification and metering are
one atomic step (`reserve_client_request`) so concurrent requests can't race past a cap.

The plaintext key is shown to the registrant ONCE at issue and never stored — we keep
only its sha256 hash (and a short prefix for operator listings). A leaked DB row can't
be turned back into a usable key.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import asyncpg

from .config import settings

KEY_PREFIX = "aim_"


def generate_key() -> str:
    """A fresh opaque client key: `aim_<43 url-safe chars>`."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """Stable sha256 hex of a presented key — what we store and look up by."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IssuedKey:
    key: str          # plaintext — return to the user once, then it's unrecoverable
    id: str
    prefix: str
    email: str | None
    rpm_limit: int
    rpd_limit: int


@dataclass(frozen=True)
class Reservation:
    granted: bool
    reason: str            # granted | unknown_key | revoked | rate_minute | rate_day
    client_id: str | None


async def issue(
    pool: asyncpg.Pool,
    *,
    email: str | None = None,
    label: str | None = None,
    rpm_limit: int | None = None,
    rpd_limit: int | None = None,
) -> IssuedKey:
    """Mint a new active key. If an active key already exists for this email it is
    revoked first (re-registering rotates the key — a lost-key self-serve path),
    keeping the one-active-key-per-email invariant."""
    key = generate_key()
    key_hash = hash_key(key)
    prefix = key[: len(KEY_PREFIX) + 8]  # e.g. "aim_A1b2C3d4"
    rpm = rpm_limit if rpm_limit is not None else settings.client_default_rpm
    rpd = rpd_limit if rpd_limit is not None else settings.client_default_rpd

    async with pool.acquire() as conn, conn.transaction():
        if email:
            await conn.execute(
                "UPDATE api_client SET status='revoked', revoked_at=now() "
                "WHERE lower(email)=lower($1) AND status='active'",
                email,
            )
        row = await conn.fetchrow(
            "INSERT INTO api_client (key_hash, key_prefix, label, email, rpm_limit, rpd_limit) "
            "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id, email, rpm_limit, rpd_limit",
            key_hash, prefix, label, email, rpm, rpd,
        )
    return IssuedKey(
        key=key,
        id=str(row["id"]),
        prefix=prefix,
        email=row["email"],
        rpm_limit=row["rpm_limit"],
        rpd_limit=row["rpd_limit"],
    )


async def reserve(pool: asyncpg.Pool, presented_key: str) -> Reservation:
    """Authenticate + rate-limit + log a request in one atomic DB call."""
    row = await pool.fetchrow(
        "SELECT granted, reason, client_id FROM reserve_client_request($1)",
        hash_key(presented_key),
    )
    return Reservation(
        granted=row["granted"],
        reason=row["reason"],
        client_id=str(row["client_id"]) if row["client_id"] is not None else None,
    )


async def revoke(pool: asyncpg.Pool, *, prefix: str | None = None, email: str | None = None) -> int:
    """Revoke matching active key(s) by prefix or email. Returns rows affected."""
    if prefix:
        result = await pool.execute(
            "UPDATE api_client SET status='revoked', revoked_at=now() "
            "WHERE key_prefix=$1 AND status='active'",
            prefix,
        )
    elif email:
        result = await pool.execute(
            "UPDATE api_client SET status='revoked', revoked_at=now() "
            "WHERE lower(email)=lower($1) AND status='active'",
            email,
        )
    else:
        raise ValueError("revoke requires prefix or email")
    # asyncpg returns e.g. "UPDATE 1"
    return int(result.split()[-1])


async def list_clients(pool: asyncpg.Pool, *, active_only: bool = False) -> list[dict]:
    where = "WHERE status='active'" if active_only else ""
    rows = await pool.fetch(
        f"SELECT id, key_prefix, label, email, status, rpm_limit, rpd_limit, "
        f"created_at, revoked_at FROM api_client {where} ORDER BY created_at DESC"
    )
    return [dict(r) for r in rows]
