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
from contextvars import ContextVar
from dataclasses import dataclass

import asyncpg

from .config import settings

KEY_PREFIX = "aim_"

# Request-scoped BYOK credentials. The MCP edge (mcp_http) can't pass parameters
# into tool functions, so it sets this before delegating a metered tool call and
# the tool reads it (see mcp_server.search). The REST gateway passes ClientKeys
# explicitly instead. Default None ⇒ service keys (stdio/local mode included).
current_client_keys: ContextVar["ClientKeys | None"] = ContextVar(
    "aimnis_client_keys", default=None
)

# Request-scoped caller identity for the MCP edge, same mechanism as above: the
# DB client uuid, or "admin" for env admin keys. Feeds hit-satisfaction sequencing
# (hashed before it touches the lookup log). Default None ⇒ untracked (stdio/local).
current_client_id: ContextVar[str | None] = ContextVar(
    "aimnis_client_id", default=None
)

# Request-scoped ANONYMOUS caller marker: the hashed client IP, set by the MCP
# edge ONLY on keyless tool calls. Tools use it to budget what costs money (live
# misses, key issuance) per IP per day — cache hits stay free. Default None ⇒
# keyed/admin/stdio, no anon budgeting.
current_anon_ip: ContextVar[str | None] = ContextVar(
    "aimnis_anon_ip", default=None
)

# Request-scoped calling application (User-Agent), set by the hosted edges so the
# search tool can attribute each lookup to the app that made it. Default None ⇒
# stdio/local (no UA). Recorded on lookup_event.user_agent.
current_user_agent: ContextVar[str | None] = ContextVar(
    "aimnis_user_agent", default=None
)


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
    byok: bool = False


SEARCH_PROVIDERS = ("brave", "tavily", "exa")


@dataclass(frozen=True)
class ClientKeys:
    """A client's own upstream credentials (BYOK). Used EXCLUSIVELY for that
    client's requests — never to serve other users' misses (the line between
    legitimate BYOK and ToS-violating quota pooling). Never logged, never
    returned by any API."""

    openrouter_api_key: str | None = None
    search_provider: str | None = None
    search_api_key: str | None = None

    def __bool__(self) -> bool:
        return bool(self.openrouter_api_key or (self.search_provider and self.search_api_key))


def byok_enabled() -> bool:
    """BYOK is fail-closed: without an encryption secret nothing is stored or read."""
    return bool(settings.byok_secret)


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
    client_keys: ClientKeys | None = None,
) -> IssuedKey:
    """Mint a new active key. If an active key already exists for this email it is
    revoked first (re-registering rotates the key — a lost-key self-serve path, and
    also the way to attach/update BYOK credentials), keeping the one-active-key-
    per-email invariant. BYOK: providing `client_keys` stores them encrypted and
    grants the (much higher) BYOK caps — their misses spend their own quota."""
    byok = bool(client_keys)
    if byok and not byok_enabled():
        raise RuntimeError("BYOK is disabled (AIMNIS_BYOK_SECRET is not set)")
    if byok and client_keys.search_provider and client_keys.search_provider not in SEARCH_PROVIDERS:
        raise ValueError(f"unknown search provider {client_keys.search_provider!r}")

    key = generate_key()
    key_hash = hash_key(key)
    prefix = key[: len(KEY_PREFIX) + 8]  # e.g. "aim_A1b2C3d4"
    default_rpm = settings.byok_rpm if byok else settings.client_default_rpm
    default_rpd = settings.byok_rpd if byok else settings.client_default_rpd
    rpm = rpm_limit if rpm_limit is not None else default_rpm
    rpd = rpd_limit if rpd_limit is not None else default_rpd

    async with pool.acquire() as conn, conn.transaction():
        if email:
            # Rotate: revoke the previous key AND wipe its stored credentials — a
            # revoked row must never keep decryptable third-party keys around.
            await conn.execute(
                "UPDATE api_client SET status='revoked', revoked_at=now(), "
                "openrouter_key_enc=NULL, search_provider=NULL, search_key_enc=NULL "
                "WHERE lower(email)=lower($1) AND status='active'",
                email,
            )
        row = await conn.fetchrow(
            "INSERT INTO api_client (key_hash, key_prefix, label, email, rpm_limit, rpd_limit) "
            "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id, email, rpm_limit, rpd_limit",
            key_hash, prefix, label, email, rpm, rpd,
        )
        if byok:
            await conn.execute(
                "UPDATE api_client SET "
                "  openrouter_key_enc = CASE WHEN $2::text IS NULL THEN NULL "
                "                            ELSE pgp_sym_encrypt($2, $5) END, "
                "  search_provider    = $3, "
                "  search_key_enc     = CASE WHEN $4::text IS NULL THEN NULL "
                "                            ELSE pgp_sym_encrypt($4, $5) END "
                "WHERE id = $1",
                row["id"], client_keys.openrouter_api_key,
                client_keys.search_provider if client_keys.search_api_key else None,
                client_keys.search_api_key, settings.byok_secret,
            )
    return IssuedKey(
        key=key,
        id=str(row["id"]),
        prefix=prefix,
        email=row["email"],
        rpm_limit=row["rpm_limit"],
        rpd_limit=row["rpd_limit"],
        byok=byok,
    )


async def load_client_keys(pool: asyncpg.Pool, client_id: str) -> ClientKeys | None:
    """Decrypt a client's BYOK credentials for use in THEIR request. Returns None
    when the client has none, the key is no longer active, or BYOK is disabled."""
    if not byok_enabled():
        return None
    row = await pool.fetchrow(
        "SELECT pgp_sym_decrypt(openrouter_key_enc, $2) AS openrouter_key, "
        "       search_provider, "
        "       pgp_sym_decrypt(search_key_enc, $2) AS search_key "
        "FROM api_client WHERE id=$1 AND status='active' "
        "AND (openrouter_key_enc IS NOT NULL OR search_key_enc IS NOT NULL)",
        client_id, settings.byok_secret,
    )
    if row is None:
        return None
    ck = ClientKeys(
        openrouter_api_key=row["openrouter_key"],
        search_provider=row["search_provider"],
        search_api_key=row["search_key"],
    )
    return ck or None


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


def hash_ip(ip: str) -> str:
    """Salted short hash of a client IP for anon budgeting — the raw IP never
    lands in the DB (same stance as client_hash in lookup_event)."""
    return hashlib.sha256(("aimnis-anon:" + ip).encode("utf-8")).hexdigest()[:16]


async def reserve_anon_miss(pool: asyncpg.Pool, ip_hash: str) -> bool:
    """Take one unit of today's keyless live-search (miss) budget for this IP.
    Cache hits are never metered — only misses spend upstream money."""
    return await pool.fetchval(
        "SELECT reserve_anon($1, 'miss', $2)", ip_hash, settings.anon_miss_rpd
    )


async def reserve_anon_registration(pool: asyncpg.Pool, ip_hash: str) -> bool:
    """Take one unit of today's in-band key-issuance budget for this IP."""
    return await pool.fetchval(
        "SELECT reserve_anon($1, 'registration', $2)", ip_hash, settings.anon_reg_rpd
    )


async def verify(pool: asyncpg.Pool, presented_key: str) -> bool:
    """Authenticate WITHOUT metering: is this an active client key? Used for MCP
    protocol chatter (initialize / tools/list) so handshakes don't burn quota —
    only actual tool calls go through `reserve`."""
    status = await pool.fetchval(
        "SELECT status FROM api_client WHERE key_hash=$1", hash_key(presented_key)
    )
    return status == "active"


async def revoke(pool: asyncpg.Pool, *, prefix: str | None = None, email: str | None = None) -> int:
    """Revoke matching active key(s) by prefix or email. Returns rows affected."""
    # Revocation also wipes stored BYOK credentials — a revoked row must never
    # keep decryptable third-party keys around.
    wipe = ("status='revoked', revoked_at=now(), "
            "openrouter_key_enc=NULL, search_provider=NULL, search_key_enc=NULL")
    if prefix:
        result = await pool.execute(
            f"UPDATE api_client SET {wipe} WHERE key_prefix=$1 AND status='active'",
            prefix,
        )
    elif email:
        result = await pool.execute(
            f"UPDATE api_client SET {wipe} WHERE lower(email)=lower($1) AND status='active'",
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
