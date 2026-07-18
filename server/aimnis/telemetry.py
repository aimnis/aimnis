"""Durable per-request telemetry for the hosted edges (`/mcp`, `/v1`).

Why this exists: Railway containers are ephemeral and platform log retention is
short, so the stdout `mcp ... ua= ip=` lines that show WHO is using the service
age out within a couple of days (learned 2026-07-18 while trying to attribute a
traffic burst — the raw IPs were already gone). This records the same signal to
Postgres so adoption is a query, not a log-forensics hunt.

Privacy: only a salted IP hash is stored, never the raw IP — the same stance as
anon_usage.ip_hash and lookup_event.client_hash. Writes are best-effort: a
telemetry failure must never break or slow a real request (observability is not
allowed to turn a working search into an error).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg

from .config import settings

log = logging.getLogger("aimnis.telemetry")


async def record_request(
    pool: asyncpg.Pool,
    *,
    surface: str,
    method: str | None,
    path: str | None,
    ip_hash: str | None,
    user_agent: str | None,
    tool: str | None = None,
    auth: str = "keyless",
) -> None:
    """Append one edge request. Best-effort — every error is swallowed."""
    if not settings.request_log_enabled:
        return
    try:
        await pool.execute(
            "INSERT INTO request_log (surface, method, path, tool, auth, ip_hash, user_agent) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7)",
            surface,
            method,
            (path or "")[:256] or None,
            tool,
            auth,
            ip_hash,
            (user_agent or "")[:512] or None,
        )
    except Exception:  # noqa: BLE001 — telemetry must never surface to the caller
        log.debug("request_log insert failed", exc_info=True)


@dataclass(frozen=True)
class Reach:
    """Aggregate adoption signal from request_log. A "source" is a distinct salted
    IP hash (never a raw IP). Every field is an aggregate — no per-source row is
    exposed; `top_user_agents` is gated to the authenticated /v1/stats."""

    requests_total: int
    tool_calls_total: int
    sources_total: int          # distinct ip_hash all-time
    sources_7d: int
    sources_24h: int
    keyless_sources: int        # distinct ip_hash seen only via the keyless free tier
    keyed_sources: int          # distinct ip_hash that presented a key (incl. admin)
    top_user_agents: list       # [(ua, requests, sources), ...] — gated to /v1/stats
    daily: list                 # [(day, requests, tool_calls, sources), ...] newest first


async def reach(pool: asyncpg.Pool, *, ua_top_n: int = 10, days: int = 30) -> Reach:
    """Roll request_log up into the dashboard's reach view. Reads only."""
    agg = await pool.fetchrow(
        """SELECT
             count(*)                                                                 AS requests_total,
             count(*) FILTER (WHERE tool IS NOT NULL)                                  AS tool_calls_total,
             count(DISTINCT ip_hash)                                                   AS sources_total,
             count(DISTINCT ip_hash) FILTER (WHERE ts > now() - interval '7 days')     AS sources_7d,
             count(DISTINCT ip_hash) FILTER (WHERE ts > now() - interval '24 hours')   AS sources_24h,
             count(DISTINCT ip_hash) FILTER (WHERE auth = 'keyless')                    AS keyless_sources,
             count(DISTINCT ip_hash) FILTER (WHERE auth IN ('keyed', 'admin'))          AS keyed_sources
           FROM request_log"""
    )
    uas = await pool.fetch(
        "SELECT user_agent, count(*) AS requests, count(DISTINCT ip_hash) AS sources "
        "FROM request_log GROUP BY user_agent ORDER BY requests DESC, user_agent LIMIT $1",
        ua_top_n,
    )
    daily = await pool.fetch(
        "SELECT ts::date AS day, count(*) AS requests, "
        "count(*) FILTER (WHERE tool IS NOT NULL) AS tool_calls, "
        "count(DISTINCT ip_hash) AS sources "
        "FROM request_log WHERE ts > now() - ($1 * interval '1 day') "
        "GROUP BY 1 ORDER BY 1 DESC",
        days,
    )
    return Reach(
        requests_total=agg["requests_total"] or 0,
        tool_calls_total=agg["tool_calls_total"] or 0,
        sources_total=agg["sources_total"] or 0,
        sources_7d=agg["sources_7d"] or 0,
        sources_24h=agg["sources_24h"] or 0,
        keyless_sources=agg["keyless_sources"] or 0,
        keyed_sources=agg["keyed_sources"] or 0,
        top_user_agents=[(r["user_agent"], r["requests"], r["sources"]) for r in uas],
        daily=[(str(r["day"]), r["requests"], r["tool_calls"], r["sources"]) for r in daily],
    )
