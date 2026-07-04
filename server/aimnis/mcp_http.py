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

KEY-LESS connections are allowed to handshake (initialize/tools/list — static,
free, unmetered even for keyed clients) but a key-less `tools/call` gets an
in-band onboarding message instead of results. Rationale: a 401 during
`initialize` dies inside the MCP client library — the human sees "connection
failed" and the model sees nothing — whereas a tool result lands in the model's
context, the one place guaranteed to be read. Fail-closed still holds where it
matters: nothing that costs money runs without a valid key, and a PRESENTED but
invalid/revoked key is still refused outright (401) on every message.
"""

from __future__ import annotations

import hmac
import json
from contextlib import asynccontextmanager

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from . import apikeys, db
from .config import settings
from .mcp_server import mcp


def _presented_key(scope) -> str | None:
    headers = {k.decode("latin-1").lower(): v.decode("latin-1")
               for k, v in scope.get("headers", [])}
    key = headers.get("x-api-key")
    auth = headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    return key or None


def _is_admin(presented: str) -> bool:
    # Constant-time over the allowlist — same rationale as gateway.require_api_key.
    return bool(settings.gateway_api_keys) and any(
        hmac.compare_digest(presented, k) for k in settings.gateway_api_keys
    )


_ANON_CALL_TEXT = (
    "You are connected to Aimnis without an API key, so search is not enabled yet. "
    "Get a free eval key by email at https://aimnis.com/register, then send it on "
    "this MCP connection as 'Authorization: Bearer aim_...' (or 'X-API-Key') — "
    "per-agent setup guides: https://aimnis.com/setup. Please relay this to the "
    "user so they can add the key."
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
        if presented is None:
            # Key-less: the handshake succeeds (so the agent sees the tools), but
            # a tool call answers with how to get a key instead of results.
            if is_tool_call:
                await respond(200, raw=_anon_call_response(body))
                return
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
        try:
            await self._manager.handle_request(scope, replay, send)
        finally:
            apikeys.current_client_id.reset(id_token)
            apikeys.current_client_keys.reset(token)


mcp_edge = McpEdge()
