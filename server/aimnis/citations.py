"""Signed citation redirect — turn cited source links into a click signal.

A cited source URL is rewritten to `{base}/r/{token}` where `token` HMAC-signs
`entry_id:source_idx`. The redirect handler verifies the token, looks the URL up
in OUR pool (never a caller-supplied URL — so this can't be an open redirector),
logs an aggregate click, and 302s to the destination.

FAIL-CLOSED: with no `citation_signing_secret` (or no resolvable base URL),
`route_url` returns None and callers emit the raw URL — no routing, no tracking.

Privacy: the click log stores entry + source index + destination host + time.
No IP, no user-agent, no session/user id. It's relevance telemetry on the pool,
not surveillance of the people using it.
"""

from __future__ import annotations

import base64
import hmac
from hashlib import sha256
from urllib.parse import urlsplit

import asyncpg

from .config import settings

_TAG_LEN = 12   # bytes of the HMAC-SHA256 tag kept (96-bit; ample for a non-secret id)
_URLH_LEN = 12  # hex chars of sha256(url) in the token (48-bit; collisions within one entry negligible)


def url_hash(url: str) -> str:
    """Stable short identity for a source URL. The token references a source by
    this hash, not by its position — so re-ordering an entry's sources (at ingest
    or refresh) never breaks already-emitted links or misattributes a click."""
    return sha256(url.encode("utf-8")).hexdigest()[:_URLH_LEN]


def _secret() -> bytes | None:
    s = settings.citation_signing_secret
    return s.encode("utf-8") if s else None


def _base() -> str | None:
    b = settings.citation_public_base_url or settings.gateway_url
    return b.rstrip("/") if b else None


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(tok: str) -> bytes:
    return base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4))


def token_for(entry_id: str, url: str) -> str | None:
    """Signed opaque token for (entry_id, url), or None if unsigned/disabled. The
    token carries the URL's hash, not a position, so it survives re-ordering."""
    secret = _secret()
    if secret is None:
        return None
    payload = f"{entry_id}:{url_hash(url)}".encode("utf-8")
    tag = hmac.new(secret, payload, sha256).digest()[:_TAG_LEN]
    return _b64e(payload + tag)


def route_url(entry_id: str | None, url: str | None) -> str | None:
    """The `/r/<token>` link for a cited source, or None when routing is off,
    unsigned, base-less, or the entry/source isn't routable (→ raw URL)."""
    if not settings.citation_routing_enabled or not entry_id or not url:
        return None
    base = _base()
    if base is None:
        return None
    tok = token_for(entry_id, url)
    return f"{base}/r/{tok}" if tok else None


def verify(token: str) -> tuple[str, str] | None:
    """Recover (entry_id, url_hash) from a token, or None if forged/malformed."""
    secret = _secret()
    if secret is None:
        return None
    try:
        raw = _b64d(token)
    except (ValueError, TypeError):
        return None
    if len(raw) <= _TAG_LEN:
        return None
    payload, tag = raw[:-_TAG_LEN], raw[-_TAG_LEN:]
    expected = hmac.new(secret, payload, sha256).digest()[:_TAG_LEN]
    if not hmac.compare_digest(tag, expected):
        return None
    try:
        entry_id, uh = payload.decode("utf-8").rsplit(":", 1)
        return entry_id, uh
    except (ValueError, UnicodeDecodeError):
        return None


def host_of(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


async def resolve_click(pool: asyncpg.Pool, token: str) -> str | None:
    """Verify a token, find the matching source in the entry BY URL HASH, log one
    click, and return the URL to redirect to. Returns None if the token is invalid,
    the entry is gone, or no current source matches the hash (e.g. it was dropped
    in a re-selection). Matching by hash — not index — is what makes old links
    survive re-ordering.

    Only http(s) destinations are returned — a stored non-http scheme is refused so
    the redirect can never emit javascript:/data: etc."""
    parsed = verify(token)
    if parsed is None:
        return None
    entry_id, want_hash = parsed

    row = await pool.fetchrow("SELECT sources FROM pool_entry WHERE id = $1", entry_id)
    if row is None:
        return None
    sources = row["sources"]
    if isinstance(sources, str):
        import json
        sources = json.loads(sources)
    if not isinstance(sources, list):
        return None

    # Find the source whose URL hashes to the token's hash; record the position it
    # currently sits at (a soft position signal for the selector's click debiasing).
    for idx, src in enumerate(sources):
        url = src.get("url") if isinstance(src, dict) else None
        if url and url_hash(url) == want_hash:
            if urlsplit(url).scheme not in ("http", "https"):
                return None
            await _record_click(pool, entry_id, idx, host_of(url), url)
            return url
    return None


async def _record_click(pool: asyncpg.Pool, entry_id: str, source_idx: int,
                        host: str | None, source_url: str | None) -> None:
    """Append one click. Best-effort: telemetry must never break the redirect.
    `source_url` is the stable click label (survives re-ordering); `source_idx` is
    the position it was shown at, for position-bias debiasing."""
    try:
        await pool.execute(
            "INSERT INTO citation_click (entry_id, source_idx, host, source_url) "
            "VALUES ($1,$2,$3,$4)",
            entry_id, source_idx, host, source_url,
        )
    except Exception:  # noqa: BLE001 — a logging failure must not 500 the redirect
        pass
