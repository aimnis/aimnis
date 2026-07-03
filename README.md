# Aimnis

### Collaborative Search for Agents

**Search once. Answer everyone.** An open-source, cache-first web-search gateway for
coding agents: ask a question and get a distilled, source-cited answer instantly from
a shared, always-current knowledge pool — so every search makes the pool smarter and
cheaper for everyone. Like RAG, but over a communal live-web pool, not your stale
private docs.

> **Status: pre-release, building in public.** The core is working and dogfooded;
> we are proving one thing before launch — that the cache hit rate compounds as the
> pool grows (the [flywheel](#the-flywheel)). Follow along.

## Why

Every model has a training cutoff. The moment it ships, the world moves on — new
library versions, new APIs, new errors — and the model can't keep up without
searching the live web on every question. That's slow, expensive, and per-vendor.

Aimnis is the shared, always-current layer in front of that: the **first** thing an
agent checks. Ask a question; if it (or a semantically similar one) has been asked
before, you get a distilled, source-cited answer instantly for near-zero cost. If
it hasn't, Aimnis fetches it live, distills it, and adds it to the pool — so the
next agent to ask gets it free. The corpus captures what happened *after* every
model's cutoff, which no static training set can.

## How it works

```
query
  └─ scrub secrets/PII (redacted before embed, search, distill, or storage)
       └─ local embed + normalize
            └─ semantic cache lookup (exact hash → vector nearest-neighbour)
                 ├─ HIT  → return the pooled, cited answer instantly (no upstream cost)
                 └─ MISS → live search → distill into a cited answer → quality-gate → pool it
```

- **Cache-first.** The knowledge pool *is* the semantic cache (pgvector). A
  reworded question hits the same entry, so the pool compounds faster than exact
  matching would.
- **Grounded, cited answers.** Misses are distilled from live web results into a
  short answer with `[n]` citations back to sources — not raw links. The answer is
  **AI-generated** (a model distills the sources) and labeled as such in the tool
  output, so the agent always knows it's reading a machine-written summary.
- **Provenance & freshness by default.** Every answer carries its model, sources,
  and a **cache timestamp** (when the answer was produced or last re-distilled —
  not when it was last served), plus a relative age, so the agent (and our own
  ranking) can weigh staleness and decide when to escalate to live search.
- **Cited links can be routed for a relevance signal (opt-in, aggregate-only).**
  When enabled, a cited source link points at a signed `/r/…` redirect that logs
  which pooled answer earned a follow-through, then forwards to the source — this
  is how click-through improves ranking. The source's real host is shown inline so
  the agent still sees where it's going, and the log records only *entry + source +
  host + time* — **never IP, user-agent, or any user/session id**. It's telemetry on
  the pool, not on you. It's off unless a signing secret is configured, tokens are
  HMAC-signed (so `/r` can't be abused as an open redirector), and self-hosting
  points the redirect at *your own* gateway — so clicks stay on your box.
- **Privacy-conscious ingress.** Common secret formats (API keys, tokens, private
  keys, connection strings) and emails are redacted *before* a query is embedded,
  sent to a search/LLM provider, or pooled; secret-dense queries are served
  live-only and never stored. This is high-precision format-based scrubbing plus an
  entropy pass — it catches known shapes, not every possible secret or PII, so treat
  it as defense-in-depth, not a guarantee. The pipeline is open source precisely so
  you can audit and extend it. **Where it runs:** self-hosted, scrubbing happens
  locally, so raw text never leaves your machine. Against the *hosted* gateway, the
  query is sent over TLS and scrubbed on ingress (in memory, before any provider call
  or persistence) — if you need redaction to happen before the query leaves your
  machine, self-host or run the local MCP server in local mode.
- **Quality-gated pool.** A distilled answer must pass a quality gate before it can
  enter the pool — a bad answer degrades to raw snippets rather than poisoning the
  commons.

## The flywheel

The one metric that decides whether this works: **cache hit rate vs. corpus size.**
If it climbs as the pool grows, the thesis holds. It's public from day one — trust,
made measurable.

```bash
# run the live dashboard locally (see Quickstart)
aimnis-dashboard      # → http://127.0.0.1:8080   (hit-rate curve, corpus size, storage)
```

## Use it with your coding agent

Aimnis speaks [MCP](https://modelcontextprotocol.io), so any MCP-capable agent can
use it as its web-search tool.

> **Install:** not on PyPI yet — run it from source via the editable install in
> [Quickstart](#quickstart-dev) below, and point your agent at that checkout's
> `python`. A packaged one-command install lands once the flywheel is proven.

**Claude Code**

```bash
claude mcp add --transport stdio aimnis-search -- \
    /path/to/server/.venv/bin/python -m aimnis.mcp_server
```

Then, to make the model prefer Aimnis over the built-in tool, deny `WebSearch` in
`.claude/settings.json`:

```json
{ "permissions": { "deny": ["WebSearch"] } }
```

**OpenCode** (and other MCP clients) — see [`docs/mcp.md`](docs/mcp.md) for the
config-file snippet and the `search` / `stats` tool reference.

## Quickstart (dev)

```bash
cd server
docker compose up -d                    # Postgres+pgvector (:5432)
# want the keyless search path too? add SearXNG (:8888):
#   docker compose --profile keyless up -d
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"              # editable install (required — see docs)
python -m aimnis.migrate                # apply migrations/*.sql
cp .env.example .env                    # then fill in keys if you have them
pytest                                  # full suite against the compose DB
```

No OpenRouter key? Fine — Aimnis runs keyless (via SearXNG) and returns raw cited
snippets, spending zero upstream quota. Add `AIMNIS_OPENROUTER_API_KEY` to turn on
distillation. Live search is an ordered fallback chain — Brave → Tavily → Exa →
SearXNG — so setting any of `AIMNIS_BRAVE_API_KEY` / `AIMNIS_TAVILY_API_KEY` /
`AIMNIS_EXA_API_KEY` gives more reliable results, and a free tier that runs dry
rolls onto the next automatically. See [`server/README.md`](server/README.md) for
the component map.

## Licensing

The code is open. The corpus is a curated compilation, licensed as such.

| Part | License |
| --- | --- |
| Server core (`server/`) | **AGPLv3** — network copyleft; run a modified service, share your changes |
| SDKs & MCP client (`clients/`) | **Apache-2.0** — embed anywhere, including commercial products |
| Public knowledge-pool pages | **CC-BY-NC 4.0** applies to our *compilation* (curation, structure, presentation) |

**On the pool's contents, honestly:** pooled answers are distilled by third-party
models from third-party web results. We don't claim ownership of the underlying
facts or of upstream model outputs — CC-BY-NC covers our compilation, and any reuse
also remains subject to the terms of the original sources and the model providers.
Some provider outputs are excluded from redistribution and from any future
training-data feed wherever provider terms require it, and provider-mandated
attributions are carried through. A training-data feed is out of scope until the
flywheel is proven (see roadmap) — this repo makes no offer to license training data.

Anti-abuse thresholds (scrubbing/dedup/quality tuning) ship with safe example
defaults; production values are injected at deploy and are not part of this tree —
so publishing the ruleset doesn't hand it to poisoners.

## Status & roadmap

Gated roadmap, Gate 0 → Gate 4. **Gate 0 complete** (ToS audit, niche, naming,
licensing locked). **Gate 1 in progress** — gateway + semantic cache + public
flywheel dashboard are built and dogfooded; the pass/kill test is the hit-rate
curve bending upward. Ads, billing, community compute, and the training-data feed
are explicitly out of scope until the flywheel is proven.
