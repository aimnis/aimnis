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

Add a database → **PostgreSQL**. Railway's standard Postgres image now **includes
pgvector**, so migration `0001`'s `CREATE EXTENSION vector` succeeds on the default
DB — there's no separate "Postgres with pgVector" template to hunt for anymore.

Verify (optional): copy the DB's `DATABASE_PUBLIC_URL` and run
`psql "<url>" -c "SELECT * FROM pg_available_extensions WHERE name='vector';"` — one
row means you're set. If it comes back empty (an old/edge image), deploy a database
from the Docker image `pgvector/pgvector:pg17` instead and point `AIMNIS_DATABASE_URL`
at it. Either way you land on a Postgres with `vector` available.

## 2. Web service

- **New service → Deploy from your GitHub repo** (the public `aimnis/aimnis`).
- **Settings → Root Directory: `server`** so Railway finds this `Dockerfile` and
  `railway.json`. The build is the Dockerfile; healthcheck is `/healthz`.

### Variables (web service)

| Variable | Value | Notes |
| --- | --- | --- |
| `AIMNIS_DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Add via **New Variable → Add Reference (🔗) → your Postgres service → `DATABASE_URL`** — do **not** type it. A hand-typed reference with a wrong service name silently doesn't resolve, the app falls back to its `localhost` default, and (because startup is `migrate && uvicorn`) the failure looks like a bare crash / 502. |
| `AIMNIS_SEARCH_BACKEND` | `brave` | Don't run SearXNG on Railway — keep the box lean. |
| `AIMNIS_BRAVE_API_KEY` | *your key* | Live-fallback search. |
| `AIMNIS_OPENROUTER_API_KEY` | *your key* | Distillation. Omit → raw cited snippets, zero upstream spend. |
| `AIMNIS_GATEWAY_API_KEYS` | `key1,key2` | **Admin/bootstrap keys** (unlimited, unmetered) — comma-separated. This is *your* operator key. Self-serve eval keys are issued by the portal into the DB (metered + revocable), so you no longer need to list every client here. The gateway is fail-closed: no request is served without a valid admin key **or** a valid DB client key. |
| `AIMNIS_ADMIN_API_KEY` | *your key* | Guards the operator endpoint `POST /admin/registration` (pause/resume eval intake). Unset ⇒ that endpoint is disabled (404); you can still flip the flag directly in the DB. |
| `AIMNIS_RESEND_API_KEY` | *your key* | Transactional email (issued-key delivery + waitlist). Omit → email is a logged no-op and the key is only shown on-screen. See **Email deliverability** below. |
| `AIMNIS_EMAIL_FROM` | `Aimnis <eval@aimnis.com>` | From address; the domain must be verified in Resend. |
| `AIMNIS_PORTAL_BASE_URL` | `https://aimnis.com` | Used in email links. |
| `AIMNIS_CLIENT_DEFAULT_RPM` / `_RPD` | `20` / `200` | Per-issued-key caps. Keep RPD well under the shared ~1,000/day upstream ceiling so one key can't drain the pool. Optional (these are the defaults). |
| `AIMNIS_BYOK_SECRET` | *long random string* | Enables **bring-your-own-keys**: clients may attach their own OpenRouter/search keys at registration (stored pgcrypto-encrypted under this secret); their misses then spend *their* quota and they get the higher `AIMNIS_BYOK_RPM/RPD` caps (defaults 60 / 5000). Unset ⇒ BYOK is hidden and disabled (fail-closed). Losing/rotating the secret orphans stored client keys (users just re-register). |

Migrations run automatically on every deploy (`python -m aimnis.migrate`, idempotent),
then uvicorn binds `0.0.0.0:8080` (or `$PORT` if Railway injects one).

> Leave `AIMNIS_EMBEDDING_MODEL` **unset** (defaults to `BAAI/bge-small-en-v1.5`).
> Only set it to another *supported* fastembed model — a bad value now fails the
> deploy at startup with a clear message rather than 500-ing on the first search.
> Take care not to paste a key/secret into the wrong variable field.

### Expose the domain (the 502 gotcha)

**Settings → Networking → Generate Domain**, then make sure its **target port is
`8080`** (matching the port uvicorn logs on startup). If the domain targets a
different port, the deploy shows "success" — Railway's *internal* healthcheck reaches
the app fine — but every *public* request returns **502**. Setting a `PORT=8080`
variable and matching the domain's target port removes the ambiguity.

## 3. Point a coding agent at the hosted gateway

On the **client** machine, run the local MCP server in remote mode — it forwards to
your Railway URL instead of touching Postgres:

```bash
AIMNIS_GATEWAY_URL=https://<your-web-service>.up.railway.app \
AIMNIS_GATEWAY_CLIENT_API_KEY=<one-of-your-gateway-keys> \
    python -m aimnis.mcp_server
```

Register that command in Claude Code / OpenCode exactly as in [`mcp.md`](mcp.md); the
only change is those two env vars.

The web service serves, on one process: the public **portal** at `/` (landing +
self-serve eval-key registration + terms + waitlist + `/setup` per-agent guides),
the **flywheel dashboard** at `/flywheel`, the REST **gateway** at `/v1/*`, and a
hosted **MCP endpoint** at `/mcp` (streamable HTTP, same keys/metering as `/v1` —
remote-MCP-capable agents connect with just the URL + an `Authorization` header,
no local install; see [`mcp.md`](mcp.md)).

## 4. Self-serve eval portal (aimnis.com)

The portal lets people register for a metered, revocable eval key without you touching
Railway. Keys live in the `api_client` table (issued/revoked live, no redeploy); each is
rate-limited per the `AIMNIS_CLIENT_DEFAULT_RPM/RPD` caps.

- **Custom domain**: Settings → Networking → Custom Domain → add `aimnis.com`, then add
  the CNAME Railway shows to your DNS. Keep the generated `*.up.railway.app` domain too
  (target port 8080, per the gotcha above).
- **Pause / resume intake**: registration defaults to open. To show "at capacity" +
  the waitlist form instead, flip the flag (needs `AIMNIS_ADMIN_API_KEY`):
  ```bash
  curl -X POST https://aimnis.com/admin/registration \
    -H "X-Admin-Key: $AIMNIS_ADMIN_API_KEY" -d 'open=false'   # resume: open=true
  ```
  (Or `UPDATE service_flag SET enabled=false WHERE name='registration_open';`.)
- **Revoke a key**: no redeploy —
  `UPDATE api_client SET status='revoked', revoked_at=now() WHERE key_prefix='aim_XXXX';`
  (or by `email`). The operator's env `AIMNIS_GATEWAY_API_KEYS` are unaffected.
- **Waitlist / registered emails** are in the `waitlist` and `api_client` tables.
  Personal-data removal is free: delete the rows on request.

### Email deliverability (spam clearance)

Don't send mail from the Railway box — port 25 is blocked and the IPs are on shared
ranges, so it lands in spam. Use **Resend** (HTTP API) with domain authentication:

1. Create a Resend account and **add the `aimnis.com` domain**.
2. Add the **DKIM + SPF (+ optionally DMARC)** DNS records Resend prints to `aimnis.com`'s
   DNS. This is what "spam clearance" actually is — it authenticates your mail so inboxes
   trust it. Wait for Resend to show the domain **Verified**.
3. Set `AIMNIS_RESEND_API_KEY` (and `AIMNIS_EMAIL_FROM=Aimnis <eval@aimnis.com>`) in Railway.

Free tier (3k emails/mo, 100/day) covers an eval preview. Without the key the portal still
works — it just shows the key on-screen instead of emailing it.

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
