"""HTTP gateway — the remote REST edge over the shared knowledge pool.

This is what makes the pool *communal* once hosted: a user's local MCP client (or
any HTTP client) calls this over the network instead of connecting to Postgres
directly. Postgres is never exposed; the gateway holds the OpenRouter/Brave keys
and does the resolution server-side.

    POST /v1/search   {"query": "..."}   -> resolution result (JSON)
    GET  /v1/stats                        -> flywheel stats (JSON)

Auth is FAIL-CLOSED: if `settings.gateway_api_keys` is empty the routes refuse with
503, so a public deploy can't accidentally serve unauthenticated, quota-spending
search to the whole internet. With keys configured, callers send one as a bearer
token (`Authorization: Bearer <key>`) or `X-API-Key: <key>`.
"""

from __future__ import annotations

import hmac
from dataclasses import asdict

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from . import db, resolve, stats
from .config import settings

router = APIRouter(prefix="/v1")


async def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """Fail-closed bearer/X-API-Key auth against the configured key allowlist."""
    if not settings.gateway_api_keys:
        # No keys configured → the gateway is intentionally not open. This protects
        # the upstream quota; set AIMNIS_GATEWAY_API_KEYS to expose it.
        raise HTTPException(status_code=503, detail="gateway disabled (no API keys configured)")

    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    # Constant-time compare against every configured key: a plain `in` test both
    # short-circuits and compares byte-by-byte, leaking key material via response
    # timing. hmac.compare_digest is fixed-time; we OR over the allowlist so the
    # number of comparisons doesn't depend on which key (if any) matches.
    ok = bool(presented) and any(
        hmac.compare_digest(presented, k) for k in settings.gateway_api_keys
    )
    if not ok:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return presented


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
