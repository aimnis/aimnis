# Using Aimnis over MCP

Aimnis exposes an [MCP](https://modelcontextprotocol.io) stdio server so any
MCP-capable coding agent can use it as its web-search tool. It provides two tools:

| Tool | Description |
| --- | --- |
| `search` | Search the web via Aimnis. Returns a cached, source-cited answer instantly when the question (or a semantically similar one) has been seen before; otherwise fetches live results, distills + cites them, and adds them to the shared pool. |
| `stats` | Report flywheel statistics — pool size, cache hit rate (all-time and recent), and the most-reused queries. |

The launch command is the same everywhere:

```
/path/to/server/.venv/bin/python -m aimnis.mcp_server
```

Point it at a Python that has the `aimnis` package installed editable (see the
[server quickstart](../server/README.md)). The server talks to the same Postgres
pool the dashboard reads from.

---

## Claude Code

```bash
claude mcp add --transport stdio aimnis-search -- \
    /path/to/server/.venv/bin/python -m aimnis.mcp_server
```

The tools appear as `mcp__aimnis_search__search` and `mcp__aimnis_search__stats`.

To make the model prefer Aimnis over the built-in web search, deny `WebSearch` in
`.claude/settings.json` (Claude Code has no allow-whitelist; a bare deny removes
the built-in tool from the model's context so it reaches for the MCP one instead):

```json
{ "permissions": { "deny": ["WebSearch"] } }
```

Restart Claude Code after registering — the stdio server is spawned once at
startup.

## OpenCode

OpenCode's built-in web search is disabled by default, which makes Aimnis a natural
drop-in. Add it to your OpenCode MCP config (`opencode.json`, or the `mcp` block in
your config file):

```json
{
  "mcp": {
    "aimnis-search": {
      "type": "local",
      "command": ["/path/to/server/.venv/bin/python", "-m", "aimnis.mcp_server"],
      "enabled": true
    }
  }
}
```

## Other MCP clients

Any client that can launch a stdio MCP server works — give it the launch command
above. Generic form:

```json
{
  "command": "/path/to/server/.venv/bin/python",
  "args": ["-m", "aimnis.mcp_server"]
}
```

---

## Notes

- **Freshness.** Every answer carries provenance (model, sources) and a staleness
  signal. Time-sensitive facts should still be verified against the cited sources —
  the tool output says so.
- **Privacy.** Common secret formats and emails are redacted before a query is
  embedded, sent to a provider, or pooled; secret-dense queries are served live-only
  and never stored. This is format-based scrubbing plus an entropy pass — treat it as
  defense-in-depth, not a guarantee it catches every secret or all PII. In **local
  mode** (default) scrubbing runs on your machine, so raw text never leaves it. In
  **remote mode** (`AIMNIS_GATEWAY_URL` set), the query is sent over TLS to the
  hosted gateway and scrubbed on ingress before any provider call or storage.
- **Cost.** Cache hits cost nothing upstream. Misses spend a small, quota-metered
  amount to fetch and distill; without an OpenRouter key the server returns raw
  cited snippets and spends nothing.
