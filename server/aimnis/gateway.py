"""HTTP gateway — the remote REST edge over the shared knowledge pool.

This is what makes the pool *communal* once hosted: a user's local MCP client (or
any HTTP client) calls this over the network instead of connecting to Postgres
directly. Postgres is never exposed; the gateway holds the OpenRouter/Brave keys
and does the resolution server-side.

    POST /v1/search   {"query": "..."}   -> resolution result (JSON)
    GET  /v1/stats                        -> flywheel stats (JSON)

Auth is FAIL-CLOSED — no request is served without a valid key. Two key sources:
  * env allowlist (`AIMNIS_GATEWAY_API_KEYS`) — the ADMIN/bootstrap path: unlimited,
    unmetered, constant-time compared. This is the operator's own key.
  * DB-backed client keys (`api_client`) — the self-serve EVAL keys the portal issues:
    metered per key (per-minute + per-day caps) and revocable at any time without a
    redeploy. A cap breach returns 429; an unknown/revoked key returns 401.
Callers send a key as a bearer token (`Authorization: Bearer <key>`) or `X-API-Key`.
"""

from __future__ import annotations

import hmac
from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from . import apikeys, db, resolve, stats
from .config import settings

router = APIRouter(prefix="/v1")


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """Fail-closed bearer/X-API-Key auth: env admin keys, then metered DB client keys."""
    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented:
        raise HTTPException(status_code=401, detail="invalid or missing API key")

    # Admin/bootstrap path: env allowlist, unlimited + unmetered. Constant-time compare
    # (a plain `in` short-circuits byte-by-byte, leaking key material via timing);
    # hmac.compare_digest is fixed-time and we OR over the allowlist so the number of
    # comparisons doesn't depend on which key (if any) matches.
    if settings.gateway_api_keys and any(
        hmac.compare_digest(presented, k) for k in settings.gateway_api_keys
    ):
        return presented

    # Metered self-serve path: authenticate + rate-limit + log in one atomic DB call.
    pool = await db.get_pool()
    res = await apikeys.reserve(pool, presented)
    if res.granted:
        return presented
    if res.reason in ("rate_minute", "rate_day"):
        raise HTTPException(status_code=429, detail=f"rate limit exceeded ({res.reason})")
    raise HTTPException(status_code=401, detail="invalid or missing API key")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    niche: str | None = None


@router.post("/search")
async def search(req: SearchRequest, _key: str = Depends(require_api_key)) -> dict:
    pool = await db.get_pool()
    result = await resolve.resolve_search(pool, req.query, niche=req.niche)
    # `format_for_agent` gives the ready-to-render text; include it so a thin client
    # doesn't need to replicate rendering, while keeping the structured fields too.
    result = dict(result)
    result["formatted"] = resolve.format_for_agent(result)
    return result


@router.get("/stats")
async def stats_endpoint(_key: str = Depends(require_api_key)) -> dict:
    # The authenticated view carries the per-query / per-host detail that the
    # public dashboard deliberately withholds (raw query text could echo a secret
    # the best-effort scrubber missed — see api.py). Only key holders see it.
    pool = await db.get_pool()
    s = await stats.gather(pool)
    clicks = await stats.click_analytics(pool)
    return {**asdict(s), "click_analytics": asdict(clicks)}
