"""Hosted MCP edge — the stdio MCP server's tools, served over streamable HTTP at /mcp.

This is what lets any remote-MCP-capable agent (OpenCode, OpenClaw, Hermes, …) use
Aimnis with just a URL and an API key — no local package install:

    url:     https://aimnis.com/mcp
    header:  Authorization: Bearer <key>   (or X-API-Key)

api.py mounts the module-level `mcp_edge` at /mcp and runs `mcp_edge.run()` in its
lifespan. We drive the SDK's StreamableHTTPSessionManager directly (rather than
mounting FastMCP's own Starlette app) for two reasons: its `.run()` is once-per-
instance, so owning construction lets each lifespan get a fresh manager (prod
restarts, tests); and handle_request() takes the raw ASGI scope, which avoids
nested-router path gymnastics under a FastAPI mount.

Auth enforces the SAME key model as the REST gateway (see gateway.py): env admin
keys pass unmetered; DB client keys are authenticated on every message but METERED
only on `tools/call` — MCP protocol chatter (initialize, tools/list, pings)
shouldn't burn a caller's daily quota. A cap breach ⇒ 429. Stateless + JSON
responses: every POST is self-contained, which suits a metered public edge (no
server-side session state to leak or exhaust).

KEY-LESS connections get the FREE TIER — search is free for agents the way Google
is free for humans. The handshake is open, and keyless tool calls actually run:
cache hits cost ~nothing to serve so they're free and unlimited; only MISSES
(live upstream search + distill spend) draw from a small per-IP daily budget
(settings.anon_miss_rpd, enforced inside resolve via miss_gate — see
mcp_server.search). Out of budget ⇒ an in-band message pointing at the `register`
tool / portal, never a transport error mid-conversation (a 401/exhausted-quota
error dies inside the MCP client library — a tool result lands in the model's
context, the one place guaranteed to be read). A per-IP per-minute in-process
throttle (settings.anon_rpm) bounds keyless call volume overall, and
settings.anon_search_enabled=False is the kill switch back to onboarding-only.

Fail-closed still holds where it matters: unbudgeted upstream spend never runs
without a valid key, and a PRESENTED but invalid/revoked key is refused outright
(401) on every message — keyless is a deliberate grant, a bad key is a refusal.

The `register` tool (defined here — hosted-edge only, stdio never imports this
module) closes the funnel in-band: an agent can fetch its user a free key without
the human ever leaving the conversation. The key is returned in the tool result
(email gets a copy); BYOK stays portal-only — upstream secrets must never be
pasted into a chat context.
"""

from __future__ import annotations

import hmac
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import parse_qs

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import ToolAnnotations
from pydantic import Field

from . import apikeys, db, email as email_mod, flags
from .config import settings
from .mcp_server import mcp

log = logging.getLogger("aimnis.mcp")


def _presented_key(scope) -> str | None:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1")
               for k, v in scope.get("headers", [])}
    key = headers.get("x-api-key")
    auth = headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    if not key:
        # URL fallback for clients that can only be configured with a bare URL
        # (Smithery-style `?api_key=...`). Only OUR key format is read: a
        # foreign-format value (e.g. a gateway's own key leaking through a
        # proxy) is ignored, so that caller stays on the keyless onboarding
        # path instead of being hard-401'd for a key they never meant to send.
        # Keys are scrubbed from access logs — see the redaction filter in api.py.
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        for name in ("api_key", "apiKey"):
            for candidate in qs.get(name, []):
                if candidate.startswith("aim_"):
                    return candidate
    return key or None


def _is_admin(presented: str) -> bool:
    # Constant-time over the allowlist — same rationale as gateway.require_api_key.
    return bool(settings.gateway_api_keys) and any(
        hmac.compare_digest(presented, k) for k in settings.gateway_api_keys
    )


_ANON_CALL_TEXT = (
    "You are connected to Aimnis without an API key, so search is not enabled yet. "
    "Get a free key by calling the `register` tool with your user's email address "
    "(ask them first — the key comes back immediately), or at "
    "https://aimnis.com/register. Then send it on this MCP connection as "
    "'Authorization: Bearer aim_...' (or 'X-API-Key') — per-agent setup guides: "
    "https://aimnis.com/setup. Please relay this to the user so they can add the key."
)


def _anon_call_response(body: bytes) -> bytes:
    """JSON-RPC reply for a key-less tools/call: a successful transport response
    whose tool result carries the onboarding message (isError, so the agent
    treats it as 'this call needs fixing', not as an answer)."""
    try:
        msg = json.loads(body)
    except Exception:  # noqa: BLE001
        msg = {}
    msgs = msg if isinstance(msg, list) else [msg]
    call = next((m for m in msgs if isinstance(m, dict) and m.get("method") == "tools/call"), {})
    return json.dumps({
        "jsonrpc": "2.0",
        "id": call.get("id", 0),
        "result": {"content": [{"type": "text", "text": _ANON_CALL_TEXT}],
                   "isError": True},
    }).encode()


def _wants_tool_call(body: bytes) -> bool:
    """Does this JSON-RPC payload invoke a tool? Anything unparseable is treated as
    a tool call, so a malformed body can never dodge metering."""
    try:
        msg = json.loads(body)
    except Exception:  # noqa: BLE001
        return True
    msgs = msg if isinstance(msg, list) else [msg]
    return any(isinstance(m, dict) and m.get("method") == "tools/call" for m in msgs)


def _called_tools(body: bytes) -> set[str]:
    """Tool names a JSON-RPC payload invokes (empty for unparseable bodies)."""
    try:
        msg = json.loads(body)
    except Exception:  # noqa: BLE001
        return set()
    msgs = msg if isinstance(msg, list) else [msg]
    return {
        (m.get("params") or {}).get("name")
        for m in msgs
        if isinstance(m, dict) and m.get("method") == "tools/call"
    } - {None}


# In-process per-IP minute throttle for KEYLESS tool calls (cache lookups burn
# real CPU: embedding + rerank). Single-instance deploy — same stance as
# portal_ip_hourly. ip -> deque of monotonic call times within the last minute.
_anon_minute: dict[str, deque] = {}


def _anon_minute_ok(ip: str) -> bool:
    now = time.monotonic()
    dq = _anon_minute.setdefault(ip, deque())
    while dq and now - dq[0] > 60.0:
        dq.popleft()
    if len(dq) >= settings.anon_rpm:
        return False
    dq.append(now)
    if len(_anon_minute) > 10_000:  # bound memory against a rotating-IP sweep
        stale = [k for k, v in _anon_minute.items() if not v or now - v[-1] > 60.0]
        for k in stale:
            _anon_minute.pop(k, None)
    return True


# --------------------------------------------------------------------------- #
# `register` tool — in-band self-serve key issuance, HOSTED EDGE ONLY.
#
# Defined here (not mcp_server.py) so it exists exactly where it makes sense:
# api.py imports this module, so the hosted /mcp edge lists it; a local stdio run
# (`python -m aimnis.mcp_server`) never imports mcp_http and never sees it.
#
# The key is returned IN the tool result — the human never has to leave the
# conversation (the email round-trip was where the signup funnel died). Email
# still gets a copy (durable storage; the chat scrolls away), but delivery
# failure no longer voids the key the way it does on the portal: in-band IS the
# primary channel here. Farming pressure is bounded by the per-IP daily
# registration cap + the one-active-key-per-email rotation invariant — and a
# farmed key buys little anyway now that keyless search works.
#
# Deliberately NO BYOK parameters: upstream provider secrets must never be
# pasted into an agent conversation (they'd land in the model context and every
# log of it). BYOK stays on the portal form.
# --------------------------------------------------------------------------- #
@mcp.tool(
    structured_output=True,
    annotations=ToolAnnotations(
        title="Get a free Aimnis API key",
        # Mints a key server-side; safe to repeat (same email ⇒ rotation).
        readOnlyHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def register(
    email: Annotated[
        str,
        Field(description="Your USER'S email address — ask them for it first, never "
                          "invent or guess one. The key is returned in this result "
                          "and a copy is emailed to this address."),
    ],
    use_case: Annotated[
        str | None,
        Field(description="Optional one-line note on what the key will be used for."),
    ] = None,
) -> str:
    """Get your user a free Aimnis API key (no credit card, takes one call).

    Ask your user for their email address first. The key comes back in this tool
    result — relay it to the user so they can save it and add it to your MCP
    connection ('Authorization: Bearer aim_...'). A key raises the daily
    live-search limits; cached answers are always free, with or without a key.
    Re-registering with the same email rotates (replaces) that email's key.
    """
    addr = (email or "").strip()
    if not email_mod.valid_address(addr):
        return ("That doesn't look like a valid email address. Ask your user for "
                "their email address and call `register` again with it.")

    pool = await db.get_pool()
    if not await flags.registration_open(pool):
        # At capacity: park them on the waitlist (idempotent) instead of a dead end.
        await pool.execute(
            "INSERT INTO waitlist (email) VALUES ($1) "
            "ON CONFLICT (lower(email)) DO NOTHING",
            addr,
        )
        return (f"Aimnis is at evaluation capacity right now, so no key was issued. "
                f"{addr} has been added to the waitlist and will be emailed the "
                f"moment capacity opens up.")

    # Keyless callers spend their per-IP daily registration budget; keyed/admin
    # callers (rotating a key) have no anon marker and skip it.
    anon = apikeys.current_anon_ip.get()
    if anon is not None and not await apikeys.reserve_anon_registration(pool, anon):
        return ("Too many keys were issued from your network today. Register at "
                "https://aimnis.com/register instead (the key arrives by email), "
                "or try again tomorrow. Keyless cached search keeps working.")

    issued = await apikeys.issue(
        pool, email=addr, label=(use_case or "").strip()[:500] or None
    )

    emailed = False
    if settings.resend_api_key:
        # Best-effort copy — reuse the portal's key email so there is one source
        # of truth for that copy. In-band delivery is primary on this path, so a
        # send failure does NOT void the key (unlike the portal's email-only flow).
        from . import portal

        emailed = await email_mod.send_email(
            addr, "Your Aimnis API key",
            portal._key_email_html(issued, settings.portal_base_url.rstrip("/")),
        )

    return "\n".join([
        f"Aimnis API key issued to {addr}:",
        "",
        f"    {issued.key}",
        "",
        "IMPORTANT: show this key to your user and have them save it — it is shown "
        "once and never again (re-registering with the same email issues a "
        "replacement)." + (" A copy was emailed." if emailed else
                           " Email delivery is unavailable, so this message is the "
                           "only copy."),
        "",
        "Use it on this MCP connection as 'Authorization: Bearer <key>' (or "
        f"'X-API-Key'). Limits: {issued.rpm_limit} requests/min, "
        f"{issued.rpd_limit:,} requests/day — cached answers are free, only live "
        "misses count. Per-agent setup guides: https://aimnis.com/setup. Higher "
        "limits with your own upstream keys: https://aimnis.com/register (never "
        "paste upstream provider keys into this chat).",
    ])


class McpEdge:
    """ASGI app: API-key auth + metering in front of a streamable-HTTP MCP session
    manager. Mount anywhere; enter `.run()` for the app's lifetime."""

    def __init__(self) -> None:
        self._manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def run(self):
        # A fresh manager per lifespan: the SDK forbids re-running one instance.
        # `mcp._mcp_server` is the low-level Server carrying the registered tools —
        # the same object FastMCP would hand its own manager.
        manager = StreamableHTTPSessionManager(
            app=mcp._mcp_server, json_response=True, stateless=True
        )
        self._manager = manager
        try:
            async with manager.run():
                yield
        finally:
            self._manager = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return

        # Who's actually hitting /mcp: the access log only shows Railway's proxy IP,
        # so a spider and a real MCP client look identical there. The User-Agent (and
        # the real client via x-forwarded-for's first hop) is what tells them apart —
        # registry validators, agent runtimes, and crawlers each announce themselves.
        # No secrets here: keys travel in Authorization / X-API-Key / ?api_key, none
        # of which is logged.
        _h = {k.decode("latin-1").lower(): v.decode("latin-1")
              for k, v in scope.get("headers", [])}
        # aimnis.com rides behind Cloudflare, so the X-Forwarded-For hop that
        # reaches the app is a CF EDGE IP, not the client (verified in prod logs
        # 2026-07-05) — keying the anon budgets on it would pool strangers'
        # quotas per edge. CF-Connecting-IP is the real visitor; prefer it.
        # (Forgeable only by callers bypassing Cloudflare via the railway.app
        # hostname — no worse than rotating real IPs; revisit if abused.)
        client_ip = (_h.get("cf-connecting-ip", "").strip()
                     or _h.get("x-forwarded-for", "").split(",")[0].strip()
                     or (scope.get("client") or ("-",))[0])
        log.info("mcp %s ua=%r ip=%s",
                 scope.get("method"), _h.get("user-agent", "-"), client_ip)

        async def respond(status: int, message: str = "", *, raw: bytes | None = None) -> None:
            extra = []
            if raw is None:
                payload = {"error": message}
                if status == 401:
                    # For direct callers (curl, scripts) the body is the onboarding
                    # surface; spec-following MCP clients look at WWW-Authenticate.
                    payload["hint"] = (
                        "Send a key as 'Authorization: Bearer aim_...' or 'X-API-Key'. "
                        "Free eval keys: https://aimnis.com/register — setup guides: "
                        "https://aimnis.com/setup"
                    )
                    extra.append((b"www-authenticate", b'Bearer realm="aimnis"'))
                raw = json.dumps(payload).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(raw)).encode()), *extra]})
            await send({"type": "http.response.body", "body": raw})

        if self._manager is None:
            await respond(503, "mcp edge not running")
            return

        presented = _presented_key(scope)

        if presented and _is_admin(presented):
            id_token = apikeys.current_client_id.set("admin")
            try:
                await self._manager.handle_request(scope, receive, send)
            finally:
                apikeys.current_client_id.reset(id_token)
            return

        # Buffer the request body so we can (a) decide whether this message is a
        # metered tool call and (b) replay it to the MCP layer afterwards.
        chunks: list[bytes] = []
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":  # e.g. http.disconnect
                return
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)

        is_tool_call = scope.get("method") == "POST" and _wants_tool_call(body)
        client_keys = None
        client_id = None
        anon_ip_hash = None
        if presented is None:
            # Key-less = the free tier. The handshake succeeds, and tool calls
            # actually run under an anon marker: cache hits are free/unmetered,
            # misses spend the per-IP daily budget (gated inside resolve — see
            # mcp_server.search), and `register` spends the per-IP issuance
            # budget. A per-minute in-process throttle bounds overall call volume.
            if is_tool_call:
                if not settings.anon_search_enabled and "register" not in _called_tools(body):
                    # Kill switch: back to onboarding-only — but never block the
                    # register tool itself (it has its own daily cap).
                    await respond(200, raw=_anon_call_response(body))
                    return
                if not _anon_minute_ok(client_ip):
                    await respond(429, "anonymous rate limit exceeded — free API "
                                       "keys raise it: https://aimnis.com/register")
                    return
                anon_ip_hash = apikeys.hash_ip(client_ip)
        elif is_tool_call:
            res = await apikeys.reserve(await db.get_pool(), presented)
            if not res.granted:
                if res.reason in ("rate_minute", "rate_day"):
                    await respond(429, f"rate limit exceeded ({res.reason})")
                else:
                    await respond(401, "invalid or missing API key")
                return
            # BYOK: load this client's own upstream credentials for THIS tool call.
            # Tool functions can't take extra parameters over MCP, so they travel
            # via a contextvar (tasks the MCP layer spawns inherit this context).
            if res.client_id:
                client_id = res.client_id
                client_keys = await apikeys.load_client_keys(await db.get_pool(), res.client_id)
        elif not await apikeys.verify(await db.get_pool(), presented):
            await respond(401, "invalid or missing API key")
            return

        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        token = apikeys.current_client_keys.set(client_keys)
        id_token = apikeys.current_client_id.set(client_id)
        anon_token = apikeys.current_anon_ip.set(anon_ip_hash)
        try:
            await self._manager.handle_request(scope, replay, send)
        finally:
            apikeys.current_anon_ip.reset(anon_token)
            apikeys.current_client_id.reset(id_token)
            apikeys.current_client_keys.reset(token)


mcp_edge = McpEdge()
