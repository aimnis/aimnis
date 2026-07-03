# Aimnis server core

The gateway + cache-first resolution engine. Licensed **AGPLv3** (see root `LICENSE`).

Components:
- Edge: MCP server (`search` + `stats` tools) + REST, per-key metering
- Ingress secret/PII scrub (redact-in-place)
- Local embedding + pgvector semantic cache (== the knowledge pool index)
- Resolution engine: hit / stale-serve+refresh / single-flight dedup / live-fallback vs OpenRouter `:free` queue
- Quality scorer → opt-in pool write (provenance, TTL, compliance flags)
- Quota ledger + conservative retry + model registry

> Anti-abuse thresholds (scrub/dedup/quality parameters) are read from config with safe example defaults here; tuned production values are injected at deploy and are **not** part of this public tree.

## Quickstart (dev)

```bash
docker compose up -d                      # Postgres + pgvector on :5432
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
python -m aimnis.migrate                  # apply migrations/*.sql
pytest                                    # runs against the compose DB
```

Config is read from the environment (`AIMNIS_` prefix) or `.env` — see `.env.example`.

## Entry points

- `aimnis-search` (`python -m aimnis.mcp_server`) — the MCP stdio server (`search` + `stats` tools). See [`../docs/mcp.md`](../docs/mcp.md).
- `aimnis-dashboard` (`python -m aimnis.api`) — the public flywheel dashboard (needs the `[api]` extra).
- `python -m aimnis.migrate` — apply `migrations/*.sql`.
- `python -m aimnis.refresh` — background re-distill/upgrade pass over snippet-only and low-quality pool entries (reuses stored sources; no new search spend).

## Status

- ✅ Schema: `pool_entry` (knowledge pool == semantic cache) + quota-ledger tables + append-only `lookup_event` (the flywheel curve's data source)
- ✅ Quota ledger: atomic `reserve_upstream_call` (per-minute / per-day / per-purpose budgets; failed 429s count), `record_upstream_outcome`, `quota_usage`
- ✅ Semantic cache: local ONNX embeddings (`fastembed`, bge-small 384d), normalize+hash, `pool.lookup` (exact hash → vector NN within `cache_max_distance`) / `pool.insert`
- ✅ Live fallback: ordered provider chain (Brave → Tavily → Exa → SearXNG) with per-provider key-guarding and cross-backend degradation
- ✅ MCP server: `search` + `stats` tools over stdio
- ✅ Distillation: OpenRouter `:free` with a provider-diverse fallback chain; grounded `[n]`-cited answers
- ✅ Quality gate: heuristic scorer (+ opt-in LLM judge) — a failing answer degrades to snippets, never pooled
- ✅ Background refresh queue: upgrades snippet-only / low-quality entries in place
- ✅ Public flywheel dashboard: hit-rate-vs-corpus curve, served as self-contained HTML
- ✅ Ingress scrubbing: secret/PII redaction before embed/search/distill/persist
