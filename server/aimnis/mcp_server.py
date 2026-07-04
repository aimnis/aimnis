"""Aimnis MCP server (stdio) — exposes a `search` tool for coding agents.

Register with Claude Code:

    claude mcp add --transport stdio aimnis-search -- \
        /path/to/server/.venv/bin/python -m aimnis.mcp_server

Then force the model to use it (deny the built-in tool) in .claude/settings.json:

    { "permissions": { "deny": ["WebSearch"] } }

The tool appears to the model as `mcp__aimnis_search__search`.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import resolve
from . import stats as stats_mod
from .config import settings
from .db import get_pool

# One FastMCP instance serves BOTH transports: `python -m aimnis.mcp_server` runs it
# over stdio on a user's machine (local or remote mode), and the hosted gateway
# serves the same tools over streamable HTTP at /mcp (see mcp_http.py) so agents can
# connect with just a URL + API key — no local install.
mcp = FastMCP("aimnis-search")


async def _remote_search(query: str, reject_entry: str | None = None) -> str:
    """Call a hosted Aimnis gateway over HTTP (remote mode)."""
    import httpx

    headers = {}
    if settings.gateway_client_api_key:
        headers["Authorization"] = f"Bearer {settings.gateway_client_api_key}"
    payload: dict = {"query": query}
    if reject_entry:
        payload["reject_entry"] = reject_entry
    async with httpx.AsyncClient(timeout=settings.gateway_timeout_seconds) as client:
        r = await client.post(
            f"{settings.gateway_url.rstrip('/')}/v1/search",
            json=payload,
            headers=headers,
        )
        r.raise_for_status()
        data = r.json()
    # The gateway already renders the agent-facing text server-side.
    return data.get("formatted") or resolve.format_for_agent(data)


async def _remote_stats() -> str:
    import dataclasses
    import httpx

    headers = {}
    if settings.gateway_client_api_key:
        headers["Authorization"] = f"Bearer {settings.gateway_client_api_key}"
    async with httpx.AsyncClient(timeout=settings.gateway_timeout_seconds) as client:
        r = await client.get(f"{settings.gateway_url.rstrip('/')}/v1/stats", headers=headers)
        r.raise_for_status()
        data = r.json()
    # /v1/stats carries extra key-holder detail (e.g. click_analytics) beyond the
    # Stats dataclass fields; keep only the known fields so added keys never break
    # the client.
    field_names = {f.name for f in dataclasses.fields(stats_mod.Stats)}
    return stats_mod.format_for_agent(
        stats_mod.Stats(**{k: v for k, v in data.items() if k in field_names})
    )


@mcp.tool()
async def search(query: str, reject_entry: str | None = None) -> str:
    """Search the web via Aimnis.

    Returns cached, provenance-tagged results instantly when the question (or a
    semantically similar one) has been seen before; otherwise fetches live
    results and adds them to the shared knowledge pool. Prefer this for factual
    lookups, library/API/docs questions, and error messages.

    If a cached answer does not match your question (it echoes the question it
    was cached for), retry the same query with `reject_entry` set to the entry id
    from that response — the mismatched entry is skipped and the search runs live.
    """
    # Remote mode: talk to the hosted gateway. Local mode: resolve against the
    # local pool. The switch is AIMNIS_GATEWAY_URL.
    if settings.gateway_url:
        return await _remote_search(query, reject_entry)
    db = await get_pool()
    # BYOK: on the hosted /mcp edge the caller's own upstream credentials arrive
    # via this contextvar (set by mcp_http per metered call); stdio/local runs get
    # the default None ⇒ service keys. current_client_id travels the same way and
    # feeds hit-satisfaction sequencing.
    from . import apikeys

    result = await resolve.resolve_search(
        db, query, client_keys=apikeys.current_client_keys.get(),
        client_id=apikeys.current_client_id.get(), reject_entry=reject_entry,
    )
    return resolve.format_for_agent(result)


@mcp.tool()
async def stats() -> str:
    """Report Aimnis flywheel statistics: knowledge-pool (cache) size, cache hit
    rate (all-time and recent), and the most-reused queries.

    This is the Gate 1 pass/kill metric — cache hit rate should climb as the
    corpus grows. Call it to see whether the compounding-pool thesis is holding.
    """
    if settings.gateway_url:
        return await _remote_stats()
    db = await get_pool()
    s = await stats_mod.gather(db)
    return stats_mod.format_for_agent(s)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
