# Deploying Aimnis on Railway

Aimnis hosts as **two Railway services in one project**: a Postgres (with pgvector)
database and the Aimnis web service (dashboard + HTTP gateway in one process). They
talk over Railway's private network, so Postgres is never exposed publicly.

```
┌─────────────────────────────┐        ┌──────────────────────────┐
│ aimnis-web (this Dockerfile) │──────▶│ Postgres + pgvector       │
│  • GET  /            dashboard│ private│  (shared knowledge pool)  │
│  • GET  /v1/stats             │  net   └──────────────────────────┘
│  • POST /v1/search  (API key) │
└─────────────────────────────┘
        ▲ HTTPS (public)
        │
   agents' local MCP clients  (AIMNIS_GATEWAY_URL → this service)
```

## 1. Database service — Postgres **with pgvector**

Add a database from the **"Postgres with pgVector"** template (not stock Postgres —
migration `0001` runs `CREATE EXTENSION vector`, which needs the extension present).
Railway exposes its connection details as variables on that service.

## 2. Web service

- **New service → Deploy from your GitHub repo** (the public `aimnis/aimnis`).
- **Settings → Root Directory: `server`** so Railway finds this `Dockerfile` and
  `railway.json`. The build is the Dockerfile; healthcheck is `/healthz`.

### Variables (web service)

| Variable | Value | Notes |
| --- | --- | --- |
| `AIMNIS_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Reference the DB service over the **private** network. Match the service name if you renamed it. |
| `AIMNIS_SEARCH_BACKEND` | `brave` | Don't run SearXNG on Railway — keep the box lean. |
| `AIMNIS_BRAVE_API_KEY` | *your key* | Live-fallback search. |
| `AIMNIS_OPENROUTER_API_KEY` | *your key* | Distillation. Omit → raw cited snippets, zero upstream spend. |
| `AIMNIS_GATEWAY_API_KEYS` | `key1,key2` | **Required to expose `/v1`.** Comma-separated. With none set the gateway is fail-closed (503). Generate long random strings; hand one to each client. |

Migrations run automatically on every deploy (`python -m aimnis.migrate`, idempotent),
then uvicorn binds `0.0.0.0:$PORT` (Railway injects `$PORT`).

## 3. Point a coding agent at the hosted gateway

On the **client** machine, run the local MCP server in remote mode — it forwards to
your Railway URL instead of touching Postgres:

```bash
AIMNIS_GATEWAY_URL=https://<your-web-service>.up.railway.app \
AIMNIS_GATEWAY_CLIENT_API_KEY=<one-of-your-gateway-keys> \
    python -m aimnis.mcp_server
```

Register that command in Claude Code / OpenCode exactly as in [`mcp.md`](mcp.md); the
only change is those two env vars. The dashboard is public at the same URL (`/`).

## Cost & footprint

- **No GPU.** Embeddings are CPU ONNX; distillation is external (OpenRouter).
- Railway Hobby is **$5/mo including $5 of usage**; an always-on web service + small
  Postgres realistically runs **~$5–15/mo**. Watch the usage dashboard weekly.
- `fastembed` downloads its ~130 MB embedding model on the first cache-miss embed,
  and a ~80 MB cross-encoder (semantic-match reranker) on the first semantic
  lookup; the first of each after a cold deploy is slower. Cache-exact hits and the
  dashboard don't touch either.
- Back up the pool (it's the whole moat): Railway's database backups, or a scheduled
  `pg_dump`.
