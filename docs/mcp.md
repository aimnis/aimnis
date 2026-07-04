# Using Aimnis over MCP

Aimnis exposes two [MCP](https://modelcontextprotocol.io) tools to any MCP-capable
coding agent:

| Tool | Description |
| --- | --- |
| `search` | Search the web via Aimnis. Returns a cached, source-cited answer instantly when the question (or a semantically similar one) has been seen before; otherwise fetches live results, distills + cites them, and adds them to the shared pool. |
| `stats` | Report flywheel statistics — pool size, cache hit rate (all-time and recent), and the most-reused queries. |

There are two ways to connect:

- **Hosted (recommended for eval users)** — the gateway serves MCP over streamable
  HTTP at `/mcp`. Nothing to install: your agent connects with a URL and an API key.
- **Local stdio** — run `python -m aimnis.mcp_server` from a checkout, either against
  your own Postgres (self-host) or forwarding to a hosted gateway (remote mode).

## Hosted endpoint

```
URL:     https://aimnis.com/mcp          (MCP, streamable HTTP)
Auth:    Authorization: Bearer <key>     (or X-API-Key: <key>)
```

Get a key at [aimnis.com/register](https://aimnis.com/register). Metering counts only
`tools/call` — MCP protocol chatter (initialize, tools/list) is free.

### Claude Code

```bash
claude mcp add --transport http aimnis https://aimnis.com/mcp \
    --header "Authorization: Bearer aim_YOUR_KEY"
```

To make the model prefer Aimnis over the built-in web search, deny `WebSearch` in
`.claude/settings.json` (a bare deny removes the built-in tool from the model's
context so it reaches for the MCP one instead):

```json
{ "permissions": { "deny": ["WebSearch"] } }
```

### OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "aimnis": {
      "type": "remote",
      "url": "https://aimnis.com/mcp",
      "enabled": true,
      "headers": { "Authorization": "Bearer aim_YOUR_KEY" }
    }
  }
}
```

OpenCode's built-in `websearch` is only active with the OpenCode provider or
`OPENCODE_ENABLE_EXA` set — leave those off and the model uses Aimnis naturally.

### OpenClaw

In `~/.openclaw/openclaw.json` under `mcp.servers`:

```json
{
  "mcp": {
    "servers": {
      "aimnis": {
        "transport": "streamable-http",
        "url": "https://aimnis.com/mcp",
        "headers": { "Authorization": "Bearer aim_YOUR_KEY" }
      }
    }
  }
}
```

To make Aimnis the only web search, disable the managed one:
`{ "tools": { "web": { "search": { "enabled": false } } } }`.
Verify with `openclaw mcp doctor --probe`.

### Hermes Agent

In `~/.hermes/config.yaml` (put `AIMNIS_KEY=aim_YOUR_KEY` in `~/.hermes/.env` —
Hermes resolves `${VAR}` placeholders from it):

```yaml
mcp_servers:
  aimnis:
    url: "https://aimnis.com/mcp"
    headers:
      Authorization: "Bearer ${AIMNIS_KEY}"
```

Tools surface as `mcp_aimnis_search` / `mcp_aimnis_stats`. To route all search
through Aimnis, disable the built-in web toolset:

```yaml
agent:
  disabled_toolsets:
    - web
```

Reload a live session with `/reload-mcp`.

### Pi

Pi has no native MCP client; the `pi-mcp-tools` extension bridges it:

```bash
pi install npm:@zhafron/pi-mcp-tools
```

```json
{
  "mcp": {
    "aimnis": { "type": "remote", "url": "https://aimnis.com/mcp" }
  }
}
```

(in `~/.pi/agent/settings.json`, or `.pi/settings.json` per-project). If your
pi-mcp-tools version doesn't pass auth headers to remote servers, use Pi's idiomatic
path instead — a CLI the agent calls via bash. Note it in your project's `AGENTS.md`:

```bash
curl -s https://aimnis.com/v1/search \
  -H "Authorization: Bearer $AIMNIS_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question"}'
```

### Any other MCP client

Point it at `https://aimnis.com/mcp` (streamable HTTP) with the `Authorization`
header, or launch the stdio server below. Agents without MCP support can call the
REST endpoint (`POST /v1/search`) directly — the response includes a ready-to-render
`formatted` field.

---

## Local stdio server

The launch command is the same everywhere:

```
/path/to/server/.venv/bin/python -m aimnis.mcp_server
```

Point it at a Python that has the `aimnis` package installed editable (see the
[server quickstart](../server/README.md)). Two modes:

- **Local/self-host** (default): resolves against your own Postgres pool.
- **Remote**: set `AIMNIS_GATEWAY_URL` (+ `AIMNIS_GATEWAY_CLIENT_API_KEY`) and it
  forwards to a hosted gateway instead — useful if your agent only supports stdio
  MCP servers.

Register with Claude Code:

```bash
claude mcp add --transport stdio aimnis-search -- \
    /path/to/server/.venv/bin/python -m aimnis.mcp_server
```

OpenCode (`opencode.json`):

```json
{
  "mcp": {
    "aimnis-search": {
      "type": "local",
      "command": ["/path/to/server/.venv/bin/python", "-m", "aimnis.mcp_server"],
      "enabled": true,
      "environment": {
        "AIMNIS_GATEWAY_URL": "https://aimnis.com",
        "AIMNIS_GATEWAY_CLIENT_API_KEY": "aim_YOUR_KEY"
      }
    }
  }
}
```

(Omit the `environment` block to run against a local Postgres instead.)

---

## Notes

- **Freshness.** Every answer carries provenance (model, sources) and a staleness
  signal. Time-sensitive facts should still be verified against the cited sources —
  the tool output says so.
- **Privacy.** Common secret formats and emails are redacted before a query is
  embedded, sent to a provider, or pooled; secret-dense queries are served live-only
  and never stored. This is format-based scrubbing plus an entropy pass — treat it as
  defense-in-depth, not a guarantee it catches every secret or all PII. In **local
  mode** (self-host) scrubbing runs on your machine, so raw text never leaves it. In
  **hosted/remote mode**, the query is sent over TLS to the gateway and scrubbed on
  ingress before any provider call or storage.
- **Cost.** Cache hits cost nothing upstream. Misses spend a small, quota-metered
  amount to fetch and distill; without an OpenRouter key the server returns raw
  cited snippets and spends nothing.
