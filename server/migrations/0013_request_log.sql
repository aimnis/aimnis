-- 0013_request_log.sql — durable per-request telemetry for the hosted edges.
--
-- Why: Railway containers are ephemeral and platform log retention is short, so
-- the stdout `mcp ... ua= ip=` lines that answer "who is actually using this"
-- age out within a couple of days (learned 2026-07-18 while trying to attribute
-- a traffic burst — the raw IPs were already gone). This keeps that signal
-- durably and queryably, independent of log retention.
--
-- Privacy: the raw IP NEVER lands here — only a salted sha256 prefix
-- (apikeys.hash_ip), identical to anon_usage.ip_hash / lookup_event.client_hash.
-- One tiny row per request to the /mcp and /v1 edges.
CREATE TABLE IF NOT EXISTS request_log (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    surface     text NOT NULL,                    -- 'mcp' | 'rest'
    method      text,                             -- HTTP method (GET/POST/...)
    path        text,                             -- request path
    tool        text,                             -- MCP tool name(s) on a tools/call; NULL otherwise
    auth        text NOT NULL DEFAULT 'keyless',   -- 'keyless' | 'keyed' | 'admin'
    ip_hash     text,                             -- salted sha256[:16] of CF-Connecting-IP (never raw)
    user_agent  text
);

CREATE INDEX IF NOT EXISTS request_log_ts_idx    ON request_log (ts);
CREATE INDEX IF NOT EXISTS request_log_ip_ts_idx ON request_log (ip_hash, ts);
