"""Quota ledger API — a thin async wrapper over the SQL reservation functions.

Flow for any upstream (OpenRouter :free) call:

    res = await reserve(pool, "background_precompute")
    if not res.granted:
        ...  # back off; the reason says which limit was hit
    else:
        try:
            <make the HTTP call>
            await record_outcome(pool, res.call_id, "success", http_status=200, ...)
        except RateLimited:
            await record_outcome(pool, res.call_id, "rate_limited", http_status=429)

The reservation is taken BEFORE the call because failed attempts (incl. 429s)
consume real quota, so they must count. Use `abandon` only when the call was
reserved but never actually sent.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg

from .config import settings


class QuotaExceeded(Exception):
    """Raised when a reservation is denied. `reason` is the limit that was hit."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"quota denied: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class Reservation:
    granted: bool
    reason: str
    call_id: str | None


@dataclass(frozen=True)
class Usage:
    minute_used: int
    minute_limit: int
    day_used: int
    day_limit: int


async def reserve(
    pool: asyncpg.Pool,
    purpose: str,
    *,
    label: str | None = None,
    model: str | None = None,
    query_hash: str | None = None,
) -> Reservation:
    row = await pool.fetchrow(
        "SELECT granted, reason, call_id FROM reserve_upstream_call($1, $2, $3, $4)",
        label or settings.quota_key_label,
        purpose,
        model,
        query_hash,
    )
    return Reservation(
        granted=row["granted"],
        reason=row["reason"],
        call_id=str(row["call_id"]) if row["call_id"] is not None else None,
    )


async def record_outcome(
    pool: asyncpg.Pool,
    call_id: str,
    status: str,
    *,
    http_status: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: str | None = None,
) -> None:
    await pool.execute(
        "SELECT record_upstream_outcome($1, $2, $3, $4, $5, $6)",
        call_id,
        status,
        http_status,
        prompt_tokens,
        completion_tokens,
        error,
    )


async def abandon(pool: asyncpg.Pool, call_id: str) -> None:
    """Mark a reserved-but-never-sent call so it stops counting toward quota."""
    await record_outcome(pool, call_id, "abandoned")


async def usage(pool: asyncpg.Pool, *, label: str | None = None) -> Usage | None:
    row = await pool.fetchrow(
        "SELECT minute_used, minute_limit, day_used, day_limit FROM quota_usage($1)",
        label or settings.quota_key_label,
    )
    if row is None:
        return None
    return Usage(
        minute_used=row["minute_used"],
        minute_limit=row["minute_limit"],
        day_used=row["day_used"],
        day_limit=row["day_limit"],
    )


@asynccontextmanager
async def reserved_call(
    pool: asyncpg.Pool,
    purpose: str,
    *,
    label: str | None = None,
    model: str | None = None,
    query_hash: str | None = None,
):
    """Reserve, run the block, and auto-record success/error.

    Raises QuotaExceeded if the reservation is denied. On a clean exit records
    'success'; on exception records 'error' and re-raises. Callers needing a
    finer outcome (e.g. 'rate_limited' with tokens) should call record_outcome
    explicitly inside the block and let this skip the default.
    """
    res = await reserve(pool, purpose, label=label, model=model, query_hash=query_hash)
    if not res.granted:
        raise QuotaExceeded(res.reason)
    assert res.call_id is not None
    try:
        yield res
    except Exception as exc:  # noqa: BLE001 — record then re-raise
        await record_outcome(pool, res.call_id, "error", error=str(exc)[:500])
        raise
    else:
        # Only default to success if the caller didn't already finalize it.
        await pool.execute(
            "SELECT record_upstream_outcome($1, 'success', NULL, NULL, NULL, NULL) "
            "WHERE (SELECT status FROM upstream_call WHERE id = $1) = 'in_flight'",
            res.call_id,
        )
